"""Unified occupied-volume reconstruction: method='sdf' or 'density'.

Required: numpy, scipy, pyvista; PyMCubes or scikit-image.
Optional: torch for CUDA Gaussian filtering; torchmcubes for CUDA extraction.
The file is safe to import: examples do not access external notebook variables.

Examples
--------
    from clay_mesh_unified import clay_mesh
    mesh = clay_mesh(points, method="sdf", smooth=(1.5, 1.5, 2.0))
    mesh = clay_mesh(points, method="density", density_level=0.3)
    mesh = clay_mesh(warped_points, method="sdf", slice_axis=None)
    meshes = clay_mesh(points, label=labels, label_mode="separate")
"""

from pathlib import Path
import warnings

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree



def _gaussian(field, sigma, device):
    """Separable Gaussian, with the same nearest boundary rule on CPU/CUDA."""
    sigma = np.asarray(sigma, dtype=float)
    if str(device) == "cpu":
        return ndi.gaussian_filter(field, sigma, mode="nearest", truncate=3.0)
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        v = torch.as_tensor(np.ascontiguousarray(field), device=device)[None, None]
        for axis, s in enumerate(sigma):
            r = int(3.0 * s + 0.5)
            if r == 0:
                continue
            t = torch.arange(-r, r + 1, device=device, dtype=v.dtype)
            kernel = torch.exp(-0.5 * (t / s).square())
            kernel /= kernel.sum()
            shape = [1, 1, 1, 1, 1]
            shape[axis + 2] = 2 * r + 1
            pad = [0] * 6
            pad[2 * (2 - axis):2 * (2 - axis) + 2] = [r, r]
            v = F.conv3d(F.pad(v, pad, mode="replicate"), kernel.reshape(shape))
        return v[0, 0].cpu().numpy()


def _denoise(points, groups, levels, axis, outlier, max_gap, k):
    """Estimate spacing, then simultaneously reject local/cross-section outliers."""
    plane = [a for a in range(3) if a != axis] if axis is not None else [0, 1, 2]
    order = np.argsort(groups, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)
    keep = np.ones(len(points), dtype=bool)
    estimates = []
    for ids in blocks:
        if len(ids) < 4:
            continue
        ki = min(int(k), len(ids) - 1)
        distances = cKDTree(points[ids][:, plane]).query(
            points[ids][:, plane], k=[ki + 1], workers=-1
        )[0][:, 0]
        med = np.median(distances)
        estimates.append(med / ki ** (1.0 / len(plane)))
        if outlier > 0:
            spread = max(1.4826 * np.median(abs(distances - med)), 0.1 * med)
            keep[ids] = distances <= med + outlier * spread
    if not estimates or np.median(estimates) <= 0:
        raise ValueError("Not enough distinct points to estimate sampling spacing.")
    d = float(np.median(estimates))
    if axis is None or outlier <= 0:
        return keep, d

    # All queries use the same pre-rejection trees: no sequential slice bias.
    ids_by_slice = {int(groups[b[0]]): b[keep[b]] for b in blocks if keep[b].any()}
    trees = {g: cKDTree(points[ids][:, plane]) for g, ids in ids_by_slice.items()}
    keys = sorted(trees)
    for j, g in enumerate(keys):
        ids = ids_by_slice[g]
        support = np.full(len(ids), np.inf)
        used = False
        for neighbor in keys[max(0, j - 1):j] + keys[j + 1:j + 2]:
            dz = abs(levels[g] - levels[neighbor])
            if dz > max_gap:
                continue
            distance = trees[neighbor].query(points[ids][:, plane], workers=-1)[0]
            support = np.minimum(support, distance / np.hypot(d, 0.35 * dz))
            used = True
        if used:
            keep[ids] &= support <= outlier
    return keep, d


def _axial_weights(z, levels, sigma):
    """Sparse Gaussian regression; sigma=0 gives physical-distance linear interpolation."""
    rows, cols, values = [], [], []
    n = len(levels)
    widths = np.ones(n) if n == 1 else np.r_[
        levels[1] - levels[0], (levels[2:] - levels[:-2]) / 2,
        levels[-1] - levels[-2]
    ]
    for i, value in enumerate(z):
        right = np.searchsorted(levels, value)
        pair = np.unique(np.clip([right - 1, right], 0, n - 1))
        if sigma == 0:
            ids = pair
            if len(ids) == 1:
                w = np.ones(1)
            else:
                a = np.clip((value - levels[ids[0]]) / np.diff(levels[ids])[0], 0, 1)
                w = np.array([1 - a, a])
        else:
            lo, hi = np.searchsorted(levels, [value - 3 * sigma, value + 3 * sigma])
            ids = np.union1d(np.arange(lo, hi), pair)
            logits = -0.5 * ((value - levels[ids]) / sigma) ** 2
            w = np.exp(logits - logits.max()) * widths[ids]
            w /= w.sum()
        rows.extend([i] * len(ids))
        cols.extend(ids)
        values.extend(w)
    return sparse.csr_matrix((np.asarray(values, np.float32), (rows, cols)),
                             shape=(len(z), n))


def _marching_cubes(field, origin, spacing, backend="auto", device="cpu"):
    """All backends receive field[ix, iy, iz]; convert index coordinates once."""
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes"}:
        raise ValueError("backend must be auto, pymcubes, skimage, or torchmcubes.")
    if backend == "auto":
        try:
            import mcubes
        except ImportError:
            backend = "skimage"
        else:
            backend = "pymcubes"
    if backend == "pymcubes":
        import mcubes
        vertices, faces = mcubes.marching_cubes(np.ascontiguousarray(field), 0.0)
    elif backend == "skimage":
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(
            field, level=0.0, gradient_direction="descent", allow_degenerate=False
        )
    else:
        import torch
        from torchmcubes import marching_cubes
        # torchmcubes uses tensor[D,H,W] and returns coordinates in (W,H,D).
        volume = torch.as_tensor(np.ascontiguousarray(field.transpose(2, 1, 0)),
                                 dtype=torch.float32, device=device)
        vertices, faces = marching_cubes(volume, 0.0)
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float64) * spacing + origin
    faces = np.asarray(faces, dtype=np.int64)
    # Preserve the backend's consistent winding, including zero-area triangles
    # on exact grid zeros. Flip all faces together toward decreasing SDF.
    center = vertices[faces].mean(axis=1)
    ijk = ((center - origin) / spacing).T
    gradient = np.stack([
        ndi.map_coordinates(np.gradient(field, spacing[a], axis=a), ijk, order=1,
                            mode="nearest", prefilter=False) for a in range(3)
    ], axis=1)
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
    if np.einsum("ij,ij->", normal, gradient) > 0:
        faces = faces[:, [0, 2, 1]]
    nonzero = np.any(normal != 0, axis=1)
    if not nonzero.all():
        # Exact zero-valued grid nodes can produce duplicate vertices and
        # collapsed triangles in classic marching cubes.
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces[nonzero]]
        used, inverse = np.unique(faces, return_inverse=True)
        vertices, faces = vertices[used], inverse.reshape(-1, 3)
    return vertices, faces


def _stack_field(points, confidence, groups, levels, known_empty, axis,
                 origin, step, shape, d, sigma, radius, outlier, max_gap, method):
    """Shared physical-z interpolation; observations are occupancy or 2D SDF."""
    plane = [a for a in range(3) if a != axis]
    xy_shape = tuple(shape[plane])
    zz = origin[axis] + np.arange(shape[axis]) * step[axis]
    h = float(np.median(np.diff(levels)))
    band = max(4 * radius * d, 3 * float(sigma.max()), 4 * float(step.max()))
    sdf_mode = method == "sdf"
    output = np.full((len(zz), int(np.prod(xy_shape))), -band if sdf_mode else 0,
                     np.float32)
    xy = None
    if sdf_mode:
        xy = np.stack(np.meshgrid(*[
            origin[a] + np.arange(shape[a]) * step[a] for a in plane
        ], indexing="ij"), axis=-1).reshape(-1, 2)
    order = np.argsort(groups, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)
    indices = {int(groups[b[0]]): b for b in blocks}
    active = np.array(sorted(indices))
    empties = np.cumsum(known_empty)
    split = (np.diff(levels[active]) > max_gap) | (np.diff(empties[active]) > 0)
    segments = np.split(active, np.flatnonzero(split) + 1)
    bounds = []
    for s, segment in enumerate(segments):
        zobs = levels[segment]
        values = np.zeros((len(segment), output.shape[1]), np.float32)
        weight = np.ones_like(values) if sdf_mode else None
        for j, g in enumerate(segment):
            ids = indices[g]
            if sdf_mode:
                distance, nearest = cKDTree(points[ids][:, plane]).query(xy, workers=-1)
                mask = (distance <= radius * d).reshape(xy_shape)
                field = ndi.distance_transform_edt(mask, sampling=step[plane])
                field -= ndi.distance_transform_edt(~mask, sampling=step[plane])
                values[j] = np.clip(field.ravel(), -band, band)
                weight[j] = np.where(distance <= 2 * radius * d,
                                     confidence[ids[nearest]], 1.0)
            else:
                ij = np.rint((points[ids][:, plane] - origin[plane]) / step[plane]).astype(int)
                flat = np.ravel_multi_index(ij.T, xy_shape)
                np.maximum.at(values[j], flat, confidence[ids])
        if sdf_mode and outlier > 0:
            for j in range(1, len(segment) - 1):
                a = (zobs[j] - zobs[j - 1]) / (zobs[j + 1] - zobs[j - 1])
                prediction = (1 - a) * values[j - 1] + a * values[j + 1]
                scale = outlier * (d + 0.5 * abs(values[j + 1] - values[j - 1]))
                weight[j] /= 1 + ((values[j] - prediction) / scale) ** 2
        lo, hi = zobs[0] - 0.5 * h, zobs[-1] + 0.5 * h
        first, last = segment[0], segment[-1]
        if first > 0 and known_empty[first - 1]:
            lo = 0.5 * (levels[first - 1] + zobs[0])
        if last + 1 < len(levels) and known_empty[last + 1]:
            hi = 0.5 * (zobs[-1] + levels[last + 1])
        # Caps must remain disjoint even when max_gap is smaller than h.
        if s > 0:
            dz = zobs[0] - levels[segments[s - 1][-1]]
            if dz > max_gap:
                lo = max(lo, zobs[0] - 0.25 * dz)
        if s + 1 < len(segments):
            dz = levels[segments[s + 1][0]] - zobs[-1]
            if dz > max_gap:
                hi = min(hi, zobs[-1] + 0.25 * dz)
        pad = band if sdf_mode else 0.0
        iz = np.flatnonzero((zz >= lo - pad) & (zz <= hi + pad))
        kernel = _axial_weights(zz[iz], zobs, sigma[axis])
        if sdf_mode:
            interpolated = kernel @ (weight * values)
            interpolated /= np.maximum(kernel @ weight, 1e-8)
            cap = np.minimum(zz[iz] - lo, hi - zz[iz])
            interpolated = np.minimum(interpolated, cap[:, None])
        else:
            interpolated = kernel @ values
        output[iz] = np.maximum(output[iz], interpolated)
        bounds.append((lo, hi))
    output = output.reshape((len(zz),) + xy_shape)
    return np.transpose(output, np.argsort([axis] + plane)), bounds


def _volume_field(points, confidence, origin, step, shape, d, radius, method):
    """Non-planar data: weighted occupancy or the SDF of local point support."""
    grid = np.zeros(tuple(shape), np.float32)
    ijk = np.rint((points - origin) / step).astype(int)
    flat = np.ravel_multi_index(ijk.T, tuple(shape))
    np.maximum.at(grid.ravel(), flat, confidence)
    if method == "density":
        return grid
    if np.all(confidence == 1):
        mask = ndi.distance_transform_edt(grid == 0, sampling=step) <= radius * d
    else:
        distance, nearest = ndi.distance_transform_edt(
            grid == 0, sampling=step, return_indices=True
        )
        mask = distance <= radius * d * grid[tuple(nearest)] ** (1.0 / 3)
    field = ndi.distance_transform_edt(mask, sampling=step)
    field -= ndi.distance_transform_edt(~mask, sampling=step)
    return field.astype(np.float32)


def clay_slice_mesh(X, label=None, feature=None, *, method="sdf", confidence=None,
              label_mode="project", slice_axis=2, voxel_size=None,
              smooth=(1.0, 1.0, 1.5), outlier=3.0, outlier_k=16,
              max_gap=None, radius=2.0, density_level=0.30,
              min_component=0.0, smooth_iters=10, backend="auto",
              device="cpu", max_voxels=200_000_000):
    """Build one PolyData, or {label: PolyData}, from points inside tissue.

    X : (N,3) array/tensor, PyVista dataset, or readable point-cloud path
        Coordinates must use the same physical unit on all axes. This is a
        VOLUME-sampling model, not a reconstruction of boundary-only samples.
    label, feature, confidence : arrays or X.point_data names, optional
        label: categorical (N,). feature: numeric (N,) or (N,C), mapped to mesh
        vertices by confidence-weighted inverse squared distance, k<=8.
        confidence: reliability in [0,1]; zero removes a point. Density mode
        uses it as occupancy amplitude; SDF sections use it in regression.
    method : 'sdf' or 'density'
        sdf: point support -> SDF -> shape smoothing -> zero surface.
        density: smooth occupancy -> relative density threshold -> SDF.
        Both share denoising, physical scales, gaps, meshing and data mapping.
    label_mode : 'project' or 'separate'
        Default project returns one mesh with nearest-point labels. Separate
        returns a dict. A label absent from a globally observed section ends
        that label's interpolation segment. Labels with <8 retained points
        are skipped with a warning. Independently built label meshes may overlap.
    slice_axis : 0, 1, 2 or None
        Default 2. Exact planes along this axis enable section-aware processing.
        If fewer than two planes, or <8 points per plane on average, use the
        general 3D branch with a warning. None explicitly selects general 3D.
        Non-planar coordinates are not flattened; no local slice directions
        are inferred after nonlinear warping or combining differently oriented stacks.
    smooth : scalar or (sx,sy,sz), default (1,1,1.5)
        Dimensionless XYZ multipliers. Physical sigma = smooth * axis_spacing.
        axis_spacing=(d,d,h) for z sections, with h on the selected slice axis;
        general 3D uses (d,d,d). d uses the shared kNN estimate, h the median
        OBSERVED interval. Zero disables main smoothing on that axis; sections
        then use linear interpolation in z. A final half-voxel Gaussian removes
        grid artifacts on enabled axes. Refining the grid preserves main sigma.
    voxel_size : scalar or (dx,dy,dz), optional
        Physical voxel spacing. Auto uses 0.6*d, limited to h/4 on the section
        axis, and coarsens to max_voxels. An explicit grid exceeding the limit
        raises MemoryError instead of silently changing requested resolution.
    outlier : float, default 3
        Shared kNN median/MAD cutoff and cross-section support cutoff. Smaller
        positive values reject more; zero disables point filtering and SDF
        residual downweighting. Not numerically equivalent to old outlier_ratio.
    outlier_k : int, default 16
        Number of neighbors for sampling-scale estimation and local denoising.
    max_gap : float, optional
        Maximum PHYSICAL separation between observed sections to bridge;
        default 3*h. Larger gaps split interpolation. Ignored in general 3D.
        Gap/features smaller than grid resolution cannot be reliably resolved.
    radius : float, default 2
        SDF ONLY: initial point-support radius / d. Larger values close small
        sampling gaps, thicken tissue and narrow holes. Ignored by density.
    density_level : float in (0,1), default .30
        DENSITY ONLY: threshold / median smoothed density at retained points.
        Smaller values expand tissue; larger values contract it. Ignored by sdf.
    min_component : float in [0,1], default 0
        Minimum occupied-voxel component size / largest component size.
        0 retains all; 1 keeps only the largest occupied VOLUME, preserving its
        inner cavity surface. Absolute voxel counts are not used in this API.
    smooth_iters : int, default 10
        Final Taubin iterations, pass_band=.1. Main smoothing is in the field.
    backend : 'auto', 'pymcubes', 'skimage' or 'torchmcubes'
        Auto prefers PyMCubes. All backends use the same physical field, origin
        and spacing. No extra mcubes.smooth is applied.
    device : str, default 'cpu'
        CUDA uses torch Gaussian filtering, and torchmcubes when explicitly
        selected. Trees, EDT, axial regression, PyMCubes, skimage and PyVista
        remain on CPU; this is not an end-to-end GPU pipeline.
    max_voxels : int, default 20_000_000
        Limit for the output grid AND observed-section arrays. This is not a
        byte-level memory limit: several temporary arrays are also required.

    Mesh point_data stores 'label'/'feature'; field_data records method, actual
    grid spacing, axis sampling spacing, physical smoothing sigma and mode.
    No global hole filling is used. Small holes may still disappear under
    support expansion or smoothing. Dense coherent outliers remain ambiguous.
    """
    if isinstance(X, (str, Path)):
        X = pv.read(X)
    source = X if hasattr(X, "points") else None
    arrays = [label, feature, confidence]
    arrays = [np.asarray(source.point_data[a] if isinstance(a, str) else
                         a.detach().cpu().numpy() if hasattr(a, "detach") else a)
              if a is not None else None for a in arrays]
    label, feature, confidence = arrays
    raw = source.points if source is not None else X
    points = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw,
                        dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("X must contain at least eight 3D points.")
    if method not in {"sdf", "density"}:
        raise ValueError("method must be 'sdf' or 'density'.")
    if label_mode not in {"project", "separate"} or (label_mode == "separate" and label is None):
        raise ValueError("Use label_mode='project', or 'separate' with label supplied.")
    if slice_axis not in (None, 0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, 2, or None.")
    for a in arrays:
        if a is not None and (a.ndim == 0 or len(a) != len(points)):
            raise ValueError("label, feature and confidence must align with X.")
    if label is not None and label.ndim != 1:
        raise ValueError("label must be one-dimensional.")
    if feature is not None and (feature.ndim not in (1, 2) or
                                not np.issubdtype(feature.dtype, np.number)):
        raise ValueError("feature must be numeric, with shape (N,) or (N,C).")
    confidence = np.ones(len(points)) if confidence is None else np.asarray(confidence, float)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or np.any(
            (confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and in [0,1].")
    multipliers = np.broadcast_to(np.asarray(smooth, float), (3,)).copy()
    if not np.isfinite(multipliers).all() or np.any(multipliers < 0) or not (
            np.isfinite(outlier) and outlier >= 0 and outlier_k >= 2):
        raise ValueError("smooth/outlier must be nonnegative and outlier_k >=2.")
    if method == "sdf" and not (np.isfinite(radius) and radius > 0):
        raise ValueError("radius must be finite and positive for method='sdf'.")
    if method == "density" and not 0 < density_level < 1:
        raise ValueError("density_level must be in (0,1) for method='density'.")
    if not 0 <= min_component <= 1 or max_voxels < 1000 or smooth_iters < 0:
        raise ValueError("Require 0<=min_component<=1, max_voxels>=1000, smooth_iters>=0.")
    retained = np.flatnonzero(np.isfinite(points).all(axis=1) & (confidence > 0))
    retained = retained[np.argsort(-confidence[retained], kind="stable")]
    _, unique = np.unique(points[retained], axis=0, return_index=True)
    retained = retained[np.sort(unique)]
    points = points[retained]
    if len(points) < 8:
        raise ValueError("Fewer than eight distinct finite reliable points remain.")
    offset = np.median(points, axis=0)
    points = points - offset
    axis = slice_axis
    levels, groups = np.array([0.0]), np.zeros(len(points), dtype=int)
    if axis is not None:
        levels, groups = np.unique(points[:, axis], return_inverse=True)
        if not (2 <= len(levels) <= len(points) // 8):
            warnings.warn("No exact planar stack detected; using general 3D. "
                          "All smooth multipliers then use point spacing d.", stacklevel=2)
            axis = None
            levels, groups = np.array([0.0]), np.zeros(len(points), dtype=int)
    h = float(np.median(np.diff(levels))) if axis is not None else 0.0
    gap = 3 * h if max_gap is None else float(max_gap)
    if axis is not None and not (np.isfinite(gap) and gap > 0):
        raise ValueError("max_gap must be finite and positive.")
    original_groups = groups.copy()
    original_labels = label[retained] if label is not None else None
    keep, d = _denoise(points, groups, levels, axis, outlier, gap, outlier_k)
    points, groups, retained = points[keep], groups[keep], retained[keep]
    if len(points) < 8:
        raise ValueError("Fewer than eight points remain; increase outlier or set it to zero.")
    confidence = confidence[retained].astype(np.float32)
    label = label[retained] if label is not None else None
    feature = feature[retained] if feature is not None else None
    scale = np.full(3, d)
    if axis is not None:
        scale[axis] = h
    sigma = multipliers * scale
    automatic_grid = voxel_size is None
    requested_step = np.broadcast_to(np.asarray(0.6 * d if automatic_grid else
                                                voxel_size, float), (3,)).copy()
    if automatic_grid and axis is not None:
        requested_step[axis] = min(requested_step[axis], 0.25 * h)
    if not np.isfinite(requested_step).all() or np.any(requested_step <= 0):
        raise ValueError("voxel_size must be finite and positive.")
    support = radius if method == "sdf" else 2.0

    def build(ids, key=None):
        p, w = points[ids], confidence[ids]
        step = requested_step.copy()
        margin = max(2.0, support) * d + 3 * sigma + 3 * step
        if axis is not None:
            margin[axis] += h
        origin, upper = p.min(axis=0) - margin, p.max(axis=0) + margin
        if axis is not None:
            origin[axis], upper[axis] = levels[0] - margin[axis], levels[-1] + margin[axis]
        while True:
            shape = np.ceil((upper - origin) / step).astype(int) + 1
            storage = int(np.prod(shape))
            if axis is not None:
                storage = max(storage, int(np.prod(np.delete(shape, axis))) * len(levels))
            if storage <= max_voxels:
                break
            if not automatic_grid:
                raise MemoryError("Requested grid exceeds max_voxels; increase voxel_size or max_voxels.")
            step *= max(1.05, (storage / max_voxels) ** (1 / 3))
        if np.any(step > requested_step * 1.001):
            warnings.warn(f"Auto grid coarsened to voxel_size={tuple(step)}.", stacklevel=3)
        slabs = []
        if axis is None:
            field = _volume_field(p, w, origin, step, shape, d, support, method)
        else:
            known_empty = np.zeros(len(levels), dtype=bool)
            if label_mode == "separate":
                known_empty = np.bincount(original_groups[original_labels == key],
                                          minlength=len(levels)) == 0
            field, slabs = _stack_field(
                p, w, groups[ids], levels, known_empty, axis, origin, step, shape,
                d, sigma, support, outlier, gap, method
            )
        blur = sigma / step
        if axis is not None:
            blur[axis] = 0  # Physical-z smoothing was already done by the sparse kernel.
        field = _gaussian(field, blur, device)
        reference = None
        if method == "density":
            sample = ndi.map_coordinates(field, ((p - origin) / step).T,
                                         order=1, mode="nearest", prefilter=False)
            reference = float(np.median(sample))
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError("Empty density field; use a finer grid or larger smooth.")
            mask = field >= density_level * reference
            field = ndi.distance_transform_edt(mask, sampling=step)
            field -= ndi.distance_transform_edt(~mask, sampling=step)
            field = field.astype(np.float32)
        field = _gaussian(field, np.where(multipliers > 0, 0.5, 0.0), device)
        if slabs:
            zz = origin[axis] + np.arange(shape[axis]) * step[axis]
            cap = np.full(len(zz), -np.inf)
            for lo, hi in slabs:
                cap = np.maximum(cap, np.minimum(zz - lo, hi - zz))
            view = [1, 1, 1]
            view[axis] = len(zz)
            np.minimum(field, cap.reshape(view), out=field)
        # Filter occupied VOLUMES, so a cavity's inner surface is retained.
        removed_components = 0
        if min_component > 0:
            regions, number = ndi.label(field > 0)
            if number:
                counts = np.bincount(regions.ravel())
                largest = 1 + np.argmax(counts[1:])
                good = counts >= min_component * counts[largest]
                good[0] = False
                if min_component == 1:
                    good[:] = False
                    good[largest] = True
                removed_components = int(number - good.sum())
                remove = (regions > 0) & ~good[regions]
                field[remove] = -max(d, float(step.max()))
        for a in range(3):
            border = [slice(None)] * 3
            border[a] = [0, -1]
            field[tuple(border)] = -max(d, float(step.max()))
        if not (field.min() < 0 < field.max()):
            raise ValueError(f"No surface for label {key!r}; reduce smooth or adjust method support.")
        vertices, faces = _marching_cubes(field, origin, step, backend, device)
        mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces]).clean()
        if smooth_iters:
            mesh = mesh.smooth_taubin(n_iter=int(smooth_iters), pass_band=0.1,
                                      normalize_coordinates=True)
        mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
        if feature is not None or label is not None:
            k = min(8, len(p)) if feature is not None else 1
            distances, neighbors = cKDTree(p).query(
                mesh.points, k=list(range(1, k + 1)), workers=-1
            )
            if label is not None:
                projected = label[ids][neighbors[:, 0]]
                mesh.point_data["label"] = projected.astype(str) if projected.dtype.kind == "O" else projected
            if feature is not None:
                f = feature[ids]
                factors = w[neighbors] / np.maximum(distances, 1e-6 * d) ** 2
                factors /= factors.sum(axis=1, keepdims=True)
                values = np.empty((mesh.n_points,) + f.shape[1:], dtype=np.float64)
                chunk = max(1, 2_000_000 // (k * (1 if f.ndim == 1 else f.shape[1])))
                for start in range(0, mesh.n_points, chunk):
                    sl = slice(start, start + chunk)
                    values[sl] = np.einsum("nk,nk...->n...", factors[sl], f[neighbors[sl]])
                mesh.point_data["feature"] = values
        mesh.points = np.asarray(mesh.points, np.float64) + offset
        mesh.field_data["method"] = [method]
        mesh.field_data["voxel_size"] = step[None, :]
        mesh.field_data["sampling_spacing"] = scale[None, :]
        mesh.field_data["smooth_sigma"] = sigma[None, :]
        mesh.field_data["reconstruction_mode"] = ["sections" if axis is not None else "volume"]
        mesh.field_data["retained_points"] = [len(p)]
        mesh.field_data["removed_components"] = [removed_components]
        if reference is not None:
            mesh.field_data["density_reference"] = [reference]
            mesh.field_data["density_threshold"] = [density_level * reference]
        return mesh

    if label_mode == "project":
        return build(np.arange(len(points)))
    meshes = {}
    for key in dict.fromkeys(original_labels.tolist()):
        ids = np.flatnonzero(label == key)
        if len(ids) < 8:
            warnings.warn(f"Skip label {key!r}: fewer than eight retained points.", stacklevel=2)
            continue
        meshes[key] = build(ids, key)
    if not meshes:
        raise ValueError("No label has enough retained points.")
    return meshes


# pc = adatas['E7.5'].obsm['gw_2_map2']
# cs = adatas['E7.5'][ adatas['E7.5'].obs['Batch'] == 'E7.5_1' ]
# points = cs.obsm['gw_2'].copy()
# labels = points[:,2].copy()
# for i,ivle in enumerate(np.unique(points[:,2])):
#     labels[labels==ivle] = i
# labels = labels.astype(str)
# labels

# points_itp = isointerp_z(points, nz=100 )
# # cc.pl.pview(points_itp )


# mesh =clay_slice_mesh(
#     points_itp,      
            
#       smooth=[1.3,1.3,3.5],
#     backend = 'pymcubes',
#       outlier=1.5,
#       outlier_k=31,
#     method="sdf",   
#     radius=1.5,
#     # density_level=0.3,
#     # smooth_iters=10,
#     #   smooth_iters=100,
# )


# _ = cc.pl.surfaces3d(meshlist=[mesh], 
#                      pointslist= [pc, ],
#                      lighting=True,
#                      point_psize=5,)


"""Mesh occupied 3D points with XYZ smoothing and SDF/occupancy fields.

Required: numpy, scipy, pyvista, and PyMCubes (or scikit-image).
Optional: torch for CUDA Gaussian filtering. The rest of the pipeline is CPU.

Examples
--------
    from clay_mesh_points import clay_mesh
    common = dict(smooth=(2, 2, 4), voxel_scale=(0.5, 0.5, 1.0), backend="pymcubes")
    mesh = clay_mesh(points, method="sdf", **common)
    mesh.save("tissue.vtp")
    other = clay_mesh(points, method="density", density_level=0.5, **common)

    # Freeze physical smoothing/grid scales when replacing points by pc_grid.
    d = float(mesh.field_data["sampling_spacing"][0])
    mesh_grid = clay_mesh(pc_grid, reference_spacing=d, **common)

    # gap_size is a PHYSICAL closing diameter, in the units of points.
    mesh = clay_mesh(warped_points, gap_size=20.0, voxel_size=3.0)
    meshes = clay_mesh(points, label=labels, feature=F, label_mode="separate")

This reconstructs the boundary of VOLUME samples (e.g. tissue cell centers).
Surface-only samples instead produce a thick shell; no interior is inferred.
The algorithm is a practical geometric prior, not recovery of unknown tissue.
"""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree

# __all__ = ["clay_mesh"]


def _geometry(points, confidence, k, outlier, radius, anisotropy, chunk,
              reference_spacing=None):
    """Local-density rejection, bounded local scales and regularized PCA."""
    tree = cKDTree(points)
    k = min(k, len(points) - 1)
    rk = tree.query(points, k=[k + 1], workers=-1)[0][:, 0]
    keep = np.ones(len(points), bool)
    if outlier > 0:
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            ids = tree.query(points[sl], k=k + 1, workers=-1)[1][:, 1:]
            score = rk[sl] / np.maximum(np.median(rk[ids], axis=1), 1e-30)
            keep[sl] = score <= outlier
    retained = np.flatnonzero(keep)
    points, confidence = points[keep], confidence[keep]
    if len(points) < 4:
        raise ValueError("Fewer than four points remain after local outlier removal.")
    if not keep.all():
        tree = cKDTree(points)
        k = min(k, len(points) - 1)
        rk = tree.query(points, k=[k + 1], workers=-1)[0][:, 0]
    d = float(np.median(rk) / np.cbrt(k) if reference_spacing is None else reference_spacing)
    precision = np.empty((len(points), 3, 3), np.float32)
    axes = np.empty((len(points), 3), np.float64)
    eye = np.eye(3)
    # Fixed 20% shrinkage caps the possible axis ratio at sqrt(13).
    shrink = min(0.2, 1.0 / anisotropy ** 2)
    for start in range(0, len(points), chunk):
        sl = slice(start, start + chunk)
        dist, ids = tree.query(points[sl], k=k + 1, workers=-1)
        dist, ids = dist[:, 1:], ids[:, 1:]
        local = np.median(rk[ids], axis=1) / np.cbrt(k)
        r = radius * np.clip(local, 0.5 * d, 2.0 * d) * np.cbrt(confidence[sl])
        if anisotropy == 1:
            axes[sl] = r[:, None]
            precision[sl] = eye / r[:, None, None] ** 2
            continue
        delta = points[ids] - points[sl, None, :]
        w = np.exp(-0.5 * (dist / rk[sl, None]) ** 2) * confidence[ids]
        w /= w.sum(axis=1, keepdims=True)
        delta -= np.einsum("nk,nkj->nj", w, delta)[:, None, :]
        cov = np.einsum("nk,nki,nkj->nij", w, delta, delta)
        cov = ((1 - shrink) * cov + (shrink / 3) *
               np.trace(cov, axis1=1, axis2=2)[:, None, None] * eye)
        eigen, basis = np.linalg.eigh(cov)
        eigen = np.maximum(eigen, eigen[:, -1:] / anisotropy ** 2)
        relative = np.sqrt(eigen / np.exp(np.log(eigen).mean(axis=1))[:, None])
        lengths = r[:, None] * relative
        axes[sl] = lengths
        precision[sl] = (basis / lengths[:, None, :] ** 2) @ basis.transpose(0, 2, 1)
    return points, confidence, retained, precision, axes, d


def _blocks(points, reach):
    """Conservative spatial blocks; sparse cell graph avoids distant empty grids."""
    cells, inverse = np.unique(np.floor((points - points.min(0)) / reach).astype(np.int64),
                               axis=0, return_inverse=True)
    edges = cKDTree(cells).query_pairs(1.0, p=np.inf, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
                              shape=(len(cells), len(cells))).tocsr()
    _, groups = sparse.csgraph.connected_components(graph, directed=False)
    groups = groups[inverse]
    order = np.argsort(groups, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)


def _support_mask(points, precision, axes, origin, step, shape, chunk):
    """Exact ellipsoid union at grid nodes, with bounded batches of candidates.

    A fixed Euclidean kNN shortlist can miss a larger/elongated ellipsoid.
    Increase the shortlist until a containing ellipsoid is found or every
    center within the largest semiaxis has been checked.
    """
    tree = cKDTree(points)
    shortest, longest = axes.min(1), float(axes.max())
    mask = np.zeros(int(np.prod(shape)), bool)
    for start in range(0, len(mask), chunk):
        stop = min(start + chunk, len(mask))
        grid = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        q = origin + grid * step
        dist, ids = tree.query(q, workers=-1)
        inside = dist <= shortest[ids]
        pending = np.flatnonzero(~inside & (dist <= longest))
        count = min(8, len(points))
        while len(pending):
            remaining = []
            batch = max(1, min(chunk, 1_000_000 // count))
            for j in range(0, len(pending), batch):
                at = pending[j:j + batch]
                dist, ids = tree.query(q[at], k=list(range(1, count + 1)),
                                       distance_upper_bound=longest, workers=-1)
                valid = ids < len(points)
                ids = np.minimum(ids, len(points) - 1)
                delta = q[at, None, :] - points[ids]
                metric = np.einsum("nki,nkij,nkj->nk", delta, precision[ids], delta)
                found = np.any(valid & (metric <= 1.0), axis=1)
                inside[at[found]] = True
                if count < len(points):
                    remaining.append(at[~found & np.isfinite(dist[:, -1])])
            pending = np.concatenate(remaining) if remaining else np.empty(0, int)
            count = min(2 * count, len(points))
        mask[start:stop] = inside
    return mask.reshape(tuple(shape))


def _gaussian_1(field, sigma, device):
    """XYZ Gaussian widths in voxel units; zero skips that axis on CPU/CUDA."""
    if str(device) == "cpu":
        return ndi.gaussian_filter(field, sigma, mode="nearest", truncate=3.0)
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        value = torch.as_tensor(np.ascontiguousarray(field), device=device)[None, None]
        for axis, s in enumerate(sigma):
            r = int(3.0 * s + 0.5)
            if r == 0:
                continue
            t = torch.arange(-r, r + 1, dtype=value.dtype, device=device)
            kernel = torch.exp(-0.5 * (t / s).square())
            kernel /= kernel.sum()
            shape, pad = [1, 1, 1, 1, 1], [0] * 6
            shape[axis + 2] = 2 * r + 1
            pad[2 * (2 - axis):2 * (2 - axis) + 2] = [r, r]
            value = F.conv3d(F.pad(value, pad, mode="replicate"), kernel.reshape(shape))
        return value[0, 0].cpu().numpy()


def _marching_cubes_1(field, origin, step, backend):
    """field[ix,iy,iz] -> physical XYZ; orient all boundaries toward air."""
    if backend == "auto":
        try:
            import mcubes
        except ImportError:
            backend = "skimage"
        else:
            backend = "pymcubes"
    if backend == "pymcubes":
        import mcubes
        vertices, faces = mcubes.marching_cubes(np.ascontiguousarray(field), 0.0)
    else:
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(np.ascontiguousarray(field), 0.0,
                                              allow_degenerate=False)
    vertices = np.asarray(vertices, np.float64) * step + origin
    faces = np.asarray(faces, np.int64)
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
    area = np.linalg.norm(normal, axis=1)
    good = area > 0
    faces, normal, area = faces[good], normal[good], area[good]
    center = vertices[faces].mean(axis=1)
    shift = normal / area[:, None] * (0.25 * step.min())
    plus = ndi.map_coordinates(field, ((center + shift - origin) / step).T,
                                order=1, mode="nearest", prefilter=False)
    minus = ndi.map_coordinates(field, ((center - shift - origin) / step).T,
                                 order=1, mode="nearest", prefilter=False)
    if np.dot(area, plus - minus) > 0:
        faces = faces[:, [0, 2, 1]]
    return pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces])).clean()


def _reconstruct(points, precision, axes, d, smooth, gap_size, voxel_size,
                 min_component, smooth_iters, backend, device, max_voxels, chunk,
                 voxel_scale=0.5, method="sdf", density_level=0.5):
    step = np.broadcast_to(np.asarray(d * np.asarray(voxel_scale) if voxel_size is None else voxel_size,
                                      dtype=float), (3,)).copy()
    if not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("voxel_scale/voxel_size must be positive and scalar or length three.")
    sigma = d * np.broadcast_to(np.asarray(smooth, float), (3,))
    closing = gap_size / 2
    reach = 2 * axes.max() + gap_size + 6 * sigma.max() + 4 * step.max()
    blocks = _blocks(points, reach)
    parts, largest, n_components, peak_voxels = [], 0, 0, 0
    for ids in blocks:
        p, a = points[ids], axes[ids]
        margin = a.max() + gap_size + 3 * sigma + 3 * step.max()
        origin = p.min(0) - margin
        shape = np.ceil((p.max(0) + margin - origin) / step).astype(np.int64) + 1
        size = int(np.prod(shape.astype(object)))
        if size > max_voxels:
            raise MemoryError(f"One spatial block needs {size:,} voxels; increase "
                              "voxel_scale, voxel_size or max_voxels. Resolution was not changed.")
        peak_voxels = max(peak_voxels, size)
        mask = _support_mask(p, precision[ids], a, origin, step, shape, chunk)
        if not mask.any():
            continue
        if closing > 0:
            dilated = ndi.distance_transform_edt(~mask, sampling=step) <= closing
            mask |= ndi.distance_transform_edt(dilated, sampling=step) > closing
            del dilated
        if method == "sdf":
            field = ndi.distance_transform_edt(mask, sampling=step).astype(np.float32)
            field -= ndi.distance_transform_edt(~mask, sampling=step).astype(np.float32)
            band = max(a.max() + gap_size, 3 * sigma.max(), 2 * step.max())
            np.clip(field, -band, band, out=field)
        else:
            # A smoothed occupied-volume fraction, independent of point counts.
            field = mask.astype(np.float32)
            band = float(density_level)
        del mask
        if np.any(sigma > 0):
            field = _gaussian_1(field, sigma / step, device)
        if method == "density":
            field -= density_level
        for axis in range(3):
            border = [slice(None)] * 3
            border[axis] = [0, -1]
            field[tuple(border)] = -band
        regions, count = ndi.label(field > 0)
        n_components += count
        if count == 0:
            continue
        sizes = np.bincount(regions.ravel())
        largest = max(largest, int(sizes[1:].max()))
        boxes = ndi.find_objects(regions)
        for region, box in enumerate(boxes, 1):
            volume = int(sizes[region])
            if volume < min_component * largest:
                continue
            lo = np.array([max(0, s.start - 1) for s in box])
            hi = np.array([min(n, s.stop + 1) for s, n in zip(box, shape)])
            crop = tuple(slice(l, h) for l, h in zip(lo, hi))
            local = field[crop].copy()
            # Keep every air voxel, including cavities, of this VOLUME region.
            local[(regions[crop] != region) & (local > 0)] = (
                -step.min() if method == "sdf" else -density_level)
            mesh = _marching_cubes_1(local, origin + lo * step, step, backend)
            parts.append((volume, mesh))
        del field, regions
    parts = [(v, m) for v, m in parts if v >= min_component * largest]
    if min_component == 1 and parts:
        parts = [max(parts, key=lambda item: item[0])]
    if not parts:
        raise ValueError("No occupied region remains; decrease voxel_scale/voxel_size "
                         "or smooth, increase radius, or decrease density_level in density mode.")
    # Concatenate triangles without reorienting cavity surfaces or merging labels.
    vertices, faces, offset = [], [], 0
    for _, mesh in parts:
        vertices.append(mesh.points)
        faces.append(mesh.faces.reshape(-1, 4)[:, 1:] + offset)
        offset += mesh.n_points
    faces = np.concatenate(faces)
    mesh = pv.PolyData(np.concatenate(vertices), np.c_[np.full(len(faces), 3), faces])
    if smooth_iters:
        mesh = mesh.smooth_taubin(n_iter=smooth_iters, pass_band=0.1,
                                  normalize_coordinates=True)
    mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
    mesh.field_data["voxel_size"] = step[None, :]
    mesh.field_data["voxel_scale"] = (step / d)[None, :]
    mesh.field_data["sampling_spacing"] = [d]
    mesh.field_data["smooth_sigma"] = sigma[None, :]
    mesh.field_data["method"] = [method]
    if method == "density":
        mesh.field_data["density_level"] = [density_level]
    mesh.field_data["gap_size"] = [gap_size]
    mesh.field_data["volume_components"] = [len(parts)]
    mesh.field_data["removed_components"] = [n_components - len(parts)]
    mesh.field_data["spatial_blocks"] = [len(blocks)]
    mesh.field_data["peak_grid_voxels"] = [peak_voxels]
    return mesh


def clay_mesh(
    pc, label=None, feature=None, confidence=None, *, label_mode="project",
    method="sdf", density_level=0.5,
    radius=2.0, smooth=1.0, gap_size=0.0, anisotropy=2.0, voxel_size=None,
    voxel_scale=0.5, reference_spacing=None,
    outlier=2.5, outlier_k=24, min_component=0.01, smooth_iters=10,
    backend="auto", device="cpu", max_voxels=1_000_000_000, chunk_size=232_768,
):
    """Reconstruct a smooth occupied volume from arbitrarily arranged 3D points.

    Parameters
    ----------
    pc : (N,3) array, torch.Tensor, PyVista dataset or readable file path
        Current physical coordinates of volume samples. No stack detection.
    label, feature, confidence : arrays or names in pc.point_data, optional
        label: (N,) categories. feature: numeric (N,) or (N,C), projected with
        confidence-weighted inverse squared distance to at most eight points.
        confidence: (N,) in [0,1]; zero excludes a point. Support volume is
        proportional to confidence, via semiaxes multiplied by c**(1/3).
        Exact duplicate coordinates keep the highest-confidence observation;
        its label/feature is retained (the first input wins confidence ties).
    label_mode : 'project' or 'separate'
        Project returns one PolyData with nearest labels. Separate returns
        {label: PolyData}, reconstructing and estimating scales per label.
        Separate meshes can overlap; they are not a joint tissue partition.
    method : 'sdf' or 'density', default 'sdf'
        Both use the same denoising, ellipsoid union, closing, grid and sigma.
        sdf: smooth the clipped signed Euclidean distance, extract level zero.
        density: smooth the binary occupied volume, extract density_level.
        This density is a local OCCUPANCY FRACTION, not a point-count KDE.
        With gap_size=0, density skips the two full-grid distance transforms.
    density_level : float in (0,1), default 0.5
        Density only: iso-level of the smoothed occupancy in [0,1]. Lower values
        expand/connect regions; higher values contract them. Ignored by sdf.
    radius : float, default 2
        Support radius multiplier of bounded LOCAL effective spacing. Local
        spacing is clipped to [0.5*d,2*d]; automatic d=median(r_k)/k**(1/3). d is a scale
        estimate under a volume prior, not an inferred original slice spacing.
        Increasing radius fills sampling gaps but thickens tissue/narrows holes.
    smooth : nonnegative scalar or (sx,sy,sz), default 1
        Gaussian standard deviations in PHYSICAL XYZ are d*(sx,sy,sz).
        A scalar s means (s,s,s); zero disables field smoothing on that axis.
        These are the CURRENT coordinate axes, independent of local PCA axes.
        Final mesh smoothing can still move vertices along any direction.
    gap_size : nonnegative float, default 0
        Diameter of the Euclidean closing ball, in PHYSICAL coordinate units.
        Closing (dilation then erosion) fills small gaps without simply keeping
        the dilated boundary. 0 disables it. This is NOT the old max_gap between
        slices and not a guarantee that every gap below this diameter closes.
        Real narrow holes can also close; there is no global fill_holes step.
    anisotropy : float >=1, default 2
        Maximum local support semiaxis ratio. 1 gives balls and skips PCA.
        Covariance shrinkage=min(0.2,1/anisotropy**2); eigenvalues are floored to
        lambda_max/anisotropy**2, then axes are volume-normalized. Large ratios
        are no longer implicitly limited to sqrt(13) by a fixed 20% shrinkage.
        Small covariance eigenvalues are NEVER inverted to invent a gap axis.
    voxel_size : positive scalar or XYZ sequence, optional
        Absolute grid spacing. If supplied, overrides voxel_scale.
    voxel_scale : positive scalar or (vx,vy,vz), default 0.5
        Relative grid spacing: step=d*voxel_scale when voxel_size is None.
        Smaller values give finer grids. Does not scale coordinates or change
        physical support/smoothing widths. A coarse grid can lose thin parts.
    reference_spacing : positive scalar, optional
        Fix the reference d instead of estimating it from each input/label.
        Useful for matching PHYSICAL smoothing and voxel scales between raw
        points and a denser pc_grid. Neighbor geometry/local scales are still
        estimated from the actual input, so surfaces need not be identical.
    outlier : float, default 2.5
        Reject when r_k / median(neighbors' r_k) > outlier. 0 disables rejection;
        otherwise require >1. Local density is compared with local density.
        Dense coherent noise clusters need component filtering and can survive
        when attached to tissue or comparable in size to real structures.
    outlier_k : int >=3, default 24
        Shared neighbor count for denoising, local scales and covariance.
    min_component : float in [0,1], default .01
        Minimum occupied VOLUME relative to the largest, separately per label.
        0 keeps all; 1 keeps one largest volume and all its cavity boundaries.
        Small genuine disconnected structures can also be removed.
    smooth_iters : nonnegative int, default 10
        Light final Taubin smoothing; 0 disables. No projection back to points.
    backend : 'auto', 'pymcubes' or 'skimage'
        Auto prefers PyMCubes. Install PyMCubes; import name is mcubes.
    device : str, default 'cpu'
        'cuda' or 'cuda:0' uses optional torch for Gaussian filtering ONLY.
        Neighbor search, support evaluation, EDT, mesh extraction/output are CPU.
    max_voxels : int, default 1_000_000_000
        Maximum grid nodes per spatial block; raises on overflow, without
        silently coarsening. Not a byte limit: EDT and labels need temporaries.
    chunk_size : positive int, default 232768
        Query batch size; ellipsoid candidates additionally use bounded batches.

    Returns
    -------
    pyvista.PolyData or dict
        Point arrays: optional 'label', 'feature', and 'source_id' (nearest
        retained input row). Field arrays record scales, counts and grid use.
        Synthetic reconstructions require checking gap/feature scales against
        the tissue; no topology or true-boundary guarantee is made.
    """
    if isinstance(pc, (str, Path)):
        pc = pv.read(pc)
    source = pc if hasattr(pc, "points") else None
    raw = source.points if source is not None else pc
    points = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw,
                        dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("pc must contain at least four 3D points.")
    arrays = []
    for value in (label, feature, confidence):
        if isinstance(value, str):
            if source is None:
                raise ValueError("Named arrays require a PyVista input.")
            value = source.point_data[value]
        if value is not None:
            value = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
            if value.ndim == 0 or len(value) != len(points):
                raise ValueError("label, feature and confidence must align with pc.")
        arrays.append(value)
    label, feature, confidence = arrays
    if label is not None and label.ndim != 1:
        raise ValueError("label must have shape (N,).")
    if feature is not None and (feature.ndim not in (1, 2) or
            not np.issubdtype(feature.dtype, np.number) or not np.isfinite(feature).all()):
        raise ValueError("feature must be a finite numeric (N,) or (N,C) array.")
    confidence = np.ones(len(points)) if confidence is None else np.asarray(confidence, float)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or np.any(
            (confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must have shape (N,) and lie in [0,1].")
    if label_mode not in {"project", "separate"} or (label_mode == "separate" and label is None):
        raise ValueError("Use label_mode='project', or 'separate' with label supplied.")
    if backend not in {"auto", "pymcubes", "skimage"}:
        raise ValueError("backend must be auto, pymcubes or skimage.")
    smooth = np.broadcast_to(np.asarray(smooth, float), (3,)).copy()
    if not np.isfinite(smooth).all() or np.any(smooth < 0):
        raise ValueError("smooth must be nonnegative, scalar or length three.")
    if method not in {"sdf", "density"}:
        raise ValueError("method must be 'sdf' or 'density'.")
    if method == "density" and not 0 < density_level < 1:
        raise ValueError("density_level must lie in (0,1).")
    if reference_spacing is not None and not (np.ndim(reference_spacing) == 0 and
            np.isfinite(reference_spacing) and reference_spacing > 0):
        raise ValueError("reference_spacing must be a positive scalar.")
    if not np.isfinite([radius, gap_size, anisotropy, outlier]).all() or not (
            radius > 0 and gap_size >= 0 and anisotropy >= 1 and
            (outlier == 0 or outlier > 1) and 0 <= min_component <= 1):
        raise ValueError("Invalid support, smoothing, outlier or component parameters.")
    if any(int(v) != v for v in (outlier_k, smooth_iters, max_voxels, chunk_size)) or not (
            outlier_k >= 3 and smooth_iters >= 0 and max_voxels >= 1000 and chunk_size > 0):
        raise ValueError("Require integer k>=3, iterations>=0, max_voxels>=1000, chunk_size>0.")
    outlier_k, smooth_iters, max_voxels, chunk_size = map(
        int, (outlier_k, smooth_iters, max_voxels, chunk_size))

    def build(ids):
        ids = ids[np.isfinite(points[ids]).all(axis=1) & (confidence[ids] > 0)]
        ids = ids[np.argsort(-confidence[ids], kind="stable")]
        _, unique = np.unique(points[ids], axis=0, return_index=True)
        ids = ids[unique]
        if len(ids) < 4:
            raise ValueError("Each reconstruction needs four distinct finite reliable points.")
        offset = np.median(points[ids], axis=0)
        p, c, retained, precision, axes, d = _geometry(
            points[ids] - offset, confidence[ids], outlier_k, outlier, radius,
            anisotropy, min(chunk_size, max(1, 1_000_000 // outlier_k)), reference_spacing)
        ids = ids[retained]
        mesh = _reconstruct(p, precision, axes, d, smooth, gap_size, voxel_size,
                            min_component, smooth_iters, backend, device, max_voxels, chunk_size,
                            voxel_scale, method, density_level)
        tree = cKDTree(p)
        nearest = tree.query(mesh.points, workers=-1)[1]
        mesh.point_data["source_id"] = ids[nearest]
        if label is not None:
            values = label[ids[nearest]]
            mesh.point_data["label"] = values.astype(str) if values.dtype.kind == "O" else values
        if feature is not None:
            k = min(8, len(p))
            values = np.empty((mesh.n_points,) + feature.shape[1:], np.float64)
            width = 1 if feature.ndim == 1 else max(1, feature.shape[1])
            batch = min(chunk_size, max(1, 1_000_000 // (k * width)))
            for start in range(0, mesh.n_points, batch):
                sl = slice(start, start + batch)
                dist, near = tree.query(mesh.points[sl], k=list(range(1, k + 1)), workers=-1)
                w = c[near] / np.maximum(dist, 1e-6 * d) ** 2
                w /= w.sum(axis=1, keepdims=True)
                values[sl] = np.einsum("nk,nk...->n...", w, feature[ids[near]])
            mesh.point_data["feature"] = values
        mesh.points = np.asarray(mesh.points, np.float64) + offset
        mesh.field_data["retained_points"] = [len(p)]
        mesh.field_data["support_radius"] = [radius * d]
        mesh.field_data["support_max_semiaxis"] = [float(axes.max())]
        mesh.field_data["anisotropy"] = [anisotropy]
        return mesh

    if label_mode == "project":
        return build(np.arange(len(points)))
    keys, inverse = np.unique(label, return_inverse=True)
    result = {}
    for i, key in enumerate(keys):
        key = key.item() if isinstance(key, np.generic) else key
        try:
            result[key] = build(np.flatnonzero(inverse == i))
        except ValueError as error:
            raise ValueError(f"Cannot reconstruct label {key!r}: {error}") from error
    return result


"""Arbitrary 3-D volume samples -> binary NumPy image -> NRRD.

Dependencies: numpy, scipy >= 1.6. NRRD writing uses the Python standard library.
No mesh, normals, slice partitioning, labels or feature interpolation is used.

    result = points_to_image(points, spacing_scale=(0.5, 0.5, 0.75),
                             support_scale=(1.5, 1.5, 2.5),
                             smooth_scale=(1.0, 1.0, 2.0))
    image_to_nrrd(result, "tissue.nrrd")
    image, grid_points = result["image"], result["points"]

All image axes are XYZ. Returned points are the centers of foreground voxels,
NOT the original observations or mesh vertices. A surface-only cloud produces
supported surface neighborhoods; a filled solid interior cannot be inferred.
"""
from pathlib import Path
import math
import gzip
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _xyz(value, name, positive=False):
    a = np.broadcast_to(np.asarray(value, float), (3,)).copy()
    if not np.isfinite(a).all() or np.any(a <= 0 if positive else a < 0):
        raise ValueError(f"{name} must be finite, scalar or XYZ, and "
                         + ("positive." if positive else "nonnegative."))
    return a


def _geometry_2(x, support, adaptive, k, cap, outlier_k, threshold,
              iterations, min_points, chunk, workers):
    """Batched fixed-k LOF, then components of a bounded spatial kNN graph."""
    for _ in range(iterations if threshold is not None else 0):
        tree = cKDTree(x)
        nk = min(outlier_k, len(x) - 1)
        kd = tree.query(x, k=[nk + 1], workers=workers)[0][:, 0]
        lrd = np.empty(len(x))
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            dd, ids = tree.query(x[sl], k=nk + 1, workers=workers)
            reach = np.maximum(dd[:, 1:], kd[ids[:, 1:]])
            lrd[sl] = 1.0 / np.maximum(reach.mean(1), np.finfo(float).tiny)
        keep = np.empty(len(x), bool)
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            ids = tree.query(x[sl], k=nk + 1, workers=workers)[1][:, 1:]
            keep[sl] = lrd[ids].mean(1) / lrd[sl] <= threshold
        if keep.all():
            break
        x = x[keep]
        if len(x) < 2:
            raise ValueError("Fewer than two distinct points remain after outlier removal.")

    # Re-estimate scale after component filtering, never from discarded outliers.
    for stage in range(2):
        nk = min(k, len(x) - 1)
        rk = cKDTree(x).query(x, k=[nk + 1], workers=workers)[0][:, 0]
        median = float(np.median(rk))
        s0 = median / np.cbrt(nk)
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError("Point coordinates do not define a finite positive scale.")
        radii = np.clip(rk / median, 1.0 / cap, cap) if adaptive else np.ones(len(x))
        if stage == 1 or min_points == 0:
            return x, s0, radii
        metric = (x - x.min(0)) / (s0 * support)
        tree = cKDTree(metric)
        nk = min(max(k, outlier_k), len(x) - 1)
        rows, cols = [], []
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            dd, ids = tree.query(metric[start:stop], k=nk + 1, workers=workers)
            dd, ids = dd[:, 1:], ids[:, 1:]
            edge = dd <= radii[start:stop, None] + radii[ids]
            r, c = np.nonzero(edge)
            rows.append(r + start)
            cols.append(ids[r, c])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                           shape=(len(x), len(x)))
        _, ids = connected_components(graph, directed=False)
        sizes = np.bincount(ids)
        cutoff = min_points * sizes.max() if min_points < 1 else min_points
        keep = sizes[ids] >= cutoff
        if keep.all():
            return x, s0, radii
        x = x[keep]
        if len(x) < 2:
            raise ValueError("Fewer than two points remain; reduce min_points.")


"""Arbitrary 3-D volume samples -> mask, point/image scores -> NRRD.

Dependencies: numpy, scipy >= 1.6. NRRD writing uses the Python standard library.
No mesh, normals, slice partitioning, labels or feature interpolation is used.

    result = points_to_image(points, spacing_scale=(0.5, 0.5, 0.75),
                             support_scale=(1.5, 1.5, 2.5),
                             smooth_scale=(1.0, 1.0, 2.0))
    image_to_nrrd(result, "mask.nrrd", key="mask")
    image_to_nrrd(result, "score.nrrd", key="score_image")
    mask, score_image = result["mask"], result["score_image"]
    grid_points, point_score = result["points"], result["score"]

Resolution: decrease spacing_scale, or set spacing in coordinate units.
Smoothing: increase smooth_scale / smooth_sigma; smooth_iters multiplies
diffusion time, giving sigma_effective = sigma * sqrt(smooth_iters).
The continuous field is thresholded ONCE to avoid repeated voxel locking.
Changing resolution does not change the requested physical smoothing scale.

All image axes are XYZ. Returned points are the centers of foreground voxels,
NOT the original observations or mesh vertices. A surface-only cloud produces
supported surface neighborhoods; a filled solid interior cannot be inferred.
"""
from pathlib import Path
import math
import gzip
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components




def _geometry_3(x, support, adaptive, k, cap, outlier_k, threshold,
              iterations, min_points, chunk, workers, base_spacing=None):
    """Batched fixed-k LOF, then components of a bounded spatial kNN graph."""
    for _ in range(iterations if threshold is not None else 0):
        tree = cKDTree(x)
        nk = min(outlier_k, len(x) - 1)
        kd = tree.query(x, k=[nk + 1], workers=workers)[0][:, 0]
        lrd = np.empty(len(x))
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            dd, ids = tree.query(x[sl], k=nk + 1, workers=workers)
            reach = np.maximum(dd[:, 1:], kd[ids[:, 1:]])
            lrd[sl] = 1.0 / np.maximum(reach.mean(1), np.finfo(float).tiny)
        keep = np.empty(len(x), bool)
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            ids = tree.query(x[sl], k=nk + 1, workers=workers)[1][:, 1:]
            keep[sl] = lrd[ids].mean(1) / lrd[sl] <= threshold
        if keep.all():
            break
        x = x[keep]
        if len(x) < 2:
            raise ValueError("Fewer than two distinct points remain after outlier removal.")

    # Re-estimate scale after component filtering, never from discarded outliers.
    for stage in range(2):
        nk = min(k, len(x) - 1)
        rk = cKDTree(x).query(x, k=[nk + 1], workers=workers)[0][:, 0]
        median = float(np.median(rk))
        s0 = median / np.cbrt(nk) if base_spacing is None else float(base_spacing)
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError("Point coordinates do not define a finite positive scale.")
        radii = np.clip(rk / median, 1.0 / cap, cap) if adaptive else np.ones(len(x))
        if stage == 1 or min_points == 0:
            return x, s0, radii
        metric = (x - x.min(0)) / (s0 * support)
        tree = cKDTree(metric)
        nk = min(max(k, outlier_k), len(x) - 1)
        rows, cols = [], []
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            dd, ids = tree.query(metric[start:stop], k=nk + 1, workers=workers)
            dd, ids = dd[:, 1:], ids[:, 1:]
            edge = dd <= radii[start:stop, None] + radii[ids]
            r, c = np.nonzero(edge)
            rows.append(r + start)
            cols.append(ids[r, c])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                           shape=(len(x), len(x)))
        _, ids = connected_components(graph, directed=False)
        sizes = np.bincount(ids)
        cutoff = min_points * sizes.max() if min_points < 1 else min_points
        keep = sizes[ids] >= cutoff
        if keep.all():
            return x, s0, radii
        x = x[keep]
        if len(x) < 2:
            raise ValueError("Fewer than two points remain; reduce min_points.")


def _support_mask_3(x, radii, axes, origin, spacing, shape, chunk, workers):
    """Exact union of variable-radius ellipsoids at queried voxel centers.

    Common XYZ axis lengths are 'axes'; each point multiplies them by radii[i].
    Adaptive candidate expansion avoids missing a farther, larger ellipsoid.
    No full grid coordinate array or grid-by-point distance matrix is allocated.
    """
    anchor = x.min(0)
    tree = cKDTree((x - anchor) / axes)
    limit = np.nextafter(float(radii.max()), np.inf)
    mask = np.zeros(math.prod(shape), bool)
    for start in range(0, len(mask), chunk):
        stop = min(start + chunk, len(mask))
        ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        q = ((origin - anchor) + ijk * spacing) / axes
        dd, ids = tree.query(q, workers=workers)
        inside = dd <= radii[ids]
        pending = np.flatnonzero(~inside & (dd <= limit))
        count = min(8, len(x))
        while len(pending):
            remaining = []
            batch = max(1, min(chunk, 1_000_000 // count))
            for lo in range(0, len(pending), batch):
                at = pending[lo:lo + batch]
                dd, ids = tree.query(q[at], k=list(range(1, count + 1)),
                                     distance_upper_bound=limit, workers=workers)
                valid = ids < len(x)
                found = np.any(valid & (dd <= radii[np.minimum(ids, len(x) - 1)]), axis=1)
                inside[at[found]] = True
                if count < len(x):
                    remaining.append(at[~found & np.isfinite(dd[:, -1])])
            pending = np.concatenate(remaining) if remaining else np.empty(0, int)
            count = min(2 * count, len(x))
        mask[start:stop] = inside
    return mask.reshape(shape)



def _smooth_mask_3(mask, spacing, sigma, method, iterations, level, shift, preserve):
    """Diffuse a continuous field, then threshold once in a fixed SDF band."""
    if not mask.any() or not np.any(sigma) or iterations == 0 or shift == 0:
        return mask
    phi = None
    if method == "sdf" or shift is not None:
        phi = ndi.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
        phi -= ndi.distance_transform_edt(mask, sampling=spacing)
    band = np.ones(mask.shape, bool) if shift is None else np.abs(phi) <= shift
    target = int(np.count_nonzero(mask[band]))
    if not band.any():
        return mask
    # Gaussian variances add. One effective kernel avoids quantizing the shape
    # between passes and reduces EDT/filter work; no iterative binary masks.
    voxel_sigma = sigma * math.sqrt(iterations) / spacing
    if method == "sdf":
        field = ndi.gaussian_filter(phi, voxel_sigma, mode="nearest", truncate=3.0)
        cut = 0.0
    else:
        field = -ndi.gaussian_filter(mask.astype(np.float32), voxel_sigma,
                                     mode="constant", cval=0.0, truncate=3.0)
        cut = -level
    del phi
    inclusive = False
    if preserve:
        values = field[band]
        if target == 0:
            cut = float(values.min())
        elif target == len(values):
            cut, inclusive = float(values.max()), True
        else:
            values.partition((target - 1, target))
            cut = float(values[target - 1])
            inclusive = cut < values[target]
            if not inclusive:
                # Do not split a tied plateau by arbitrary voxel index. Choose
                # inclusion/exclusion with the smaller volume error.
                below = np.count_nonzero(values < cut)
                through = np.count_nonzero(values <= cut)
                inclusive = through - target <= target - below
        del values
    inside = field <= cut if inclusive else field < cut
    return np.where(band, inside, mask)


def points_to_image(
    points, *,
    spacing_scale=0.5,
    spacing=None,
    base_spacing=None,
    method="sdf",
    support_scale=1.5,
    adaptive=True,
    support_k=8,
    radius_cap=2.0,
    smooth_scale=1.0,
    smooth_sigma=None,
    smooth_iters=1,
    preserve_volume=True,
    max_shift_scale=2.0,
    density_level=0.5,
    closing_scale=0.0,
    opening_scale=0.0,
    hole_scale=0.0,
    offset_scale=0.0,
    outlier_k=24,
    outlier_thresh=1.8,
    outlier_iters=1,
    min_points=0.01,
    min_voxels=0,
    component_scale=0.0,
    keep_largest=False,
    score_k=8,
    score_scale=2.0,
    padding=2,
    bounds=None,
    max_voxels=2000_000_000,
    chunk_size=80_000,
    workers=-1,
):
    """Return mask, score_image, spacing, origin, points and score; no file I/O.

    Parameters
    ----------
    points : (N,3) array or object with .points
        XYZ coordinates in consistent units, with at least two distinct finite
        rows. Duplicates and nonfinite rows are removed. No slice axis is assumed.
    spacing_scale : positive scalar or XYZ, default 0.5
        Voxel spacing = s0 * spacing_scale. The automatic reference scale is
        s0 = median(distance to kth other point) / k**(1/3), k=support_k capped
        by N-1, after filtering. This is a VOLUME-sampling scale estimate, not an
        estimate of missing slice spacing. k=1 uses median nearest-neighbor distance.
        Decrease it to increase grid resolution; halving all three spacings
        requires about 8 times as many voxels for a fixed physical domain.
    spacing : positive scalar or XYZ, optional
        Actual voxel spacing in coordinate units. Overrides spacing_scale.
        Changing spacing alone does not change support or physical smoothing.
    base_spacing : positive float, optional
        Fixed reference scale s0, overriding the automatic estimate throughout
        component filtering and reconstruction. Useful to keep scale parameters
        comparable across clouds or different sampling densities. Does not disable
        local adaptive radii. Keep it fixed when comparing resolutions/settings.
    method : {'sdf', 'density', 'support', 'occupancy'}, default 'sdf'
        sdf: local ellipsoid support -> physical SDF -> Gaussian -> threshold.
        density: same support -> Gaussian-smoothed binary occupancy -> threshold;
                 it is not a raw-point KDE and does not sum point counts.
        support: local ellipsoid union, with no field smoothing.
        occupancy: mark retained input-point voxels, with no field smoothing.
        Optional morphology and cleanup below apply to ALL methods.
    support_scale : positive scalar or XYZ, default 1.5
        Ellipsoid semiaxes = s0 * support_scale * local_radius_factor.
        Larger z support bridges along WORLD z, independently of z voxel spacing.
        All support ellipsoids are axis-aligned; no local PCA or normals are used.
    adaptive, support_k, radius_cap : default True, 8, 2.0
        Factor = clip(local kth distance / its median, 1/radius_cap, radius_cap).
        adaptive=False uses a fixed factor of 1. radius_cap must be >=1. The cap
        limits excessive expansion in sparse regions. support_k also defines s0.
    smooth_scale : nonnegative scalar or XYZ, default 1.0
        Per-unit-time physical Gaussian sigma = s0 * smooth_scale. Zero disables
        filtering on an axis. SDF/density only. Sigma is a standard deviation,
        not a support radius or voxel count. Independent of voxel resolution.
    smooth_sigma : nonnegative scalar or XYZ, optional
        Physical Gaussian sigma in coordinate units; overrides smooth_scale.
        Effective filter sigma in voxels = smooth_sigma * sqrt(smooth_iters)
        / spacing. Without this override, use s0*smooth_scale in the numerator.
    smooth_iters : nonnegative int, default 1
        Gaussian diffusion-time multiplier: sigma_effective = sigma*sqrt(iters).
        The continuous SDF/occupancy field is filtered with ONE effective kernel,
        then thresholded once. 4 gives twice the linear smoothing scale of 1;
        0 disables smoothing. This replaces the old reinitialize/binarize-per-pass
        behavior for iters>1, which could stagnate on the voxel grid. Small
        structures can disappear; no topology or fine-detail guarantee is made.
    preserve_volume : bool, default True
        Adjust smoothing threshold to approximately retain the support volume
        AFTER opening/closing. For tied field values, include or exclude the
        complete plateau according to which is closer to the target volume.
        Exact volume is not always possible without splitting symmetric ties.
        Does not preserve unknown true volume, topology, or the later hole/offset
        changes. Applies only to smoothing, not to every reconstruction step.
    max_shift_scale : nonnegative scalar or None, default 2.0
        Smoothing can change signs only where the INITIAL voxel SDF has magnitude
        <= s0*this value. None removes the band limit; 0 disables smoothing.
        This is a voxel-SDF constraint, not an exact mesh Hausdorff bound.
        A small band can limit the effect even when smooth_scale is large.
    density_level : float in (0,1), default 0.5
        Threshold of the smoothed support occupancy for method='density', used
        only when preserve_volume=False. Lower values expand the foreground.
    closing_scale, opening_scale : nonnegative scalar or XYZ, default 0
        Ellipsoid DIAMETERS in multiples of s0. Closing bridges small gaps;
        opening removes narrow protrusions. Zero on an axis prevents movement
        along that axis. Both are disabled by default to avoid closing real gaps.
    hole_scale : nonnegative float, default 0
        After smoothing/offset, fill only ENCLOSED cavities whose volume is no
        larger than a sphere of diameter s0*hole_scale. 0 disables; np.inf fills
        all enclosed cavities. Through-holes are not enclosed cavities.
    offset_scale : finite float, default 0
        Final isotropic expansion (>0) or erosion (<0), by abs(offset_scale)*s0.
    outlier_k, outlier_thresh, outlier_iters : default 24, 1.8, 1
        Batched fixed-k LOF; reject scores above outlier_thresh (>1). None or
        outlier_iters=0 disables rejection. Compact distant noise groups may
        survive LOF; min_points provides an additional component filter.
    min_points : nonnegative float, default 0.01
        Filter POINT components before grid allocation. A distance-limited kNN
        graph in the support metric is used; k=max(support_k,outlier_k).
        (0,1): fraction of largest component; >=1: minimum point count; 0: off.
    min_voxels, keep_largest : default 0, False
        Final 26-connected MASK component filtering: absolute voxel count, and
        optionally retain only the largest mask component. min_voxels is NOT a
        physical size; use component_scale for consistent filtering across grids.
    component_scale : nonnegative float, default 0
        Minimum retained mask-component volume equals the volume of a sphere
        of diameter s0*component_scale. Unlike min_voxels, this physical threshold
        does not change when spacing changes. If both are positive, apply the
        stricter threshold. 0 disables this additional filter. This is a volume
        criterion, not a test of maximum width or component diameter.
    score_k : positive int, default 8
        Number of distinct retained observations used to score each generated
        point, capped at the retained point count. Larger k requires support
        from more observations. A score is not forced to 1 at a source point.
    score_scale : positive scalar or XYZ, default 2.0
        Scoring Gaussian bandwidth = s0 * score_scale. In this normalized metric,
        score(q) = mean(exp(-0.5 * distance(q, neighbor)**2)) over score_k nearest
        observations. Output lies in [0,1]: closer/more nearby observations give
        stronger support; unsupported gap-filling or expansion gives weak support.
        No division by the sum of Gaussian weights is applied. Fixed bandwidth
        avoids assigning high scores merely because adaptive support radii grew.
        Independent of the reconstruction support/smoothing settings except via
        their effect on point filtering and s0. Changing scoring parameters does
        not change the mask, grid or filtering. This is a geometric observation-
        support measure, not a calibrated probability or proof of correct anatomy.
        This changes the score definition, not shape smoothing; do not increase
        it merely to make reconstructed areas appear more reliable.
    padding : nonnegative integer or XYZ integers, default 2
        Extra empty voxel layers in addition to automatically computed support,
        morphology and smoothing margins. Automatically sized margins prevent
        normal boundary clipping; the volume includes background voxels.
    bounds : (2,3) array, optional
        Explicit lower/upper XYZ bounds BEFORE padding. Must enclose all retained
        points; outside points raise an error, never get silently cropped.
        Default is the filtered point bounding box.
    max_voxels, chunk_size, workers : default 20_000_000, 50_000, -1
        Padded grid size limit, KDTree query batch size, and query threads.
        Oversized grids raise instead of silently coarsening. Temporary EDT
        arrays need additional memory; mask + score_image require 5 bytes per grid
        voxel; float64 points + float32 score require 28 bytes per foreground
        voxel. Runtime is CPU; no CUDA dependency.

    Returns
    -------
    dict with exactly six keys:
        mask    : (nx,ny,nz) uint8, binary foreground 1 / background 0.
        score_image : (nx,ny,nz) float32, foreground scores in [0,1], background 0.
                  Exactly score_image[mask.astype(bool)] == score. A foreground
                  score can also be 0: use mask to distinguish background.
        spacing : (3,) float64, actual XYZ voxel spacing, in input units.
        origin  : (3,) float64, world coordinate of voxel center (0,0,0).
        points  : (M,3) float64, foreground voxel centers in np.argwhere order.
                  points = origin + np.argwhere(mask) * spacing.
        score   : (M,) float32 in [0,1], matching points row for row. Computed
                  against retained unique ORIGINAL observations, after all mask
                  operations; generated points never provide their own support.

    Grid axes align with the input Cartesian XYZ axes, so no direction matrix
    or transform is needed in the result. Image shapes/dtypes come from the arrays.
    The previous 'image' key is now 'mask'; no duplicate alias is returned.
    Coordinates alone cannot distinguish a missing observation from a real
    anatomical gap. Support, closing and smoothing can merge nearby structures.
    """
    scale = _xyz(spacing_scale, "spacing_scale", True)
    physical_spacing = None if spacing is None else _xyz(spacing, "spacing", True)
    support = _xyz(support_scale, "support_scale", True)
    smooth = _xyz(smooth_scale, "smooth_scale")
    physical_sigma = None if smooth_sigma is None else _xyz(smooth_sigma, "smooth_sigma")
    closing = _xyz(closing_scale, "closing_scale")
    opening = _xyz(opening_scale, "opening_scale")
    score_axes = _xyz(score_scale, "score_scale", True)
    pad = _xyz(padding, "padding")
    if np.any(pad != np.floor(pad)):
        raise ValueError("padding must contain integers.")
    if method not in {"sdf", "density", "support", "occupancy"}:
        raise ValueError("method must be sdf, density, support, or occupancy.")
    for name, value, lower in (("support_k", support_k, 1), ("outlier_k", outlier_k, 2),
                               ("outlier_iters", outlier_iters, 0), ("smooth_iters", smooth_iters, 0),
                               ("min_voxels", min_voxels, 0), ("score_k", score_k, 1),
                               ("chunk_size", chunk_size, 1),
                               ("max_voxels", max_voxels, 1)):
        if not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}.")
    support_k, outlier_k, outlier_iters, smooth_iters, chunk_size = map(
        int, (support_k, outlier_k, outlier_iters, smooth_iters, chunk_size))
    if (not np.isfinite([radius_cap, min_points, offset_scale, component_scale]).all()
            or radius_cap < 1 or min_points < 0 or component_scale < 0):
        raise ValueError("Require radius_cap >= 1, nonnegative min_points/component_scale, "
                         "and finite offset_scale.")
    if base_spacing is not None and (not np.isfinite(base_spacing) or base_spacing <= 0):
        raise ValueError("base_spacing must be a finite positive scalar, or None.")
    if np.isnan(hole_scale) or hole_scale < 0:
        raise ValueError("hole_scale must be nonnegative, or np.inf.")
    if not 0 < density_level < 1:
        raise ValueError("density_level must be in (0,1).")
    if outlier_thresh is not None and (not np.isfinite(outlier_thresh) or outlier_thresh <= 1):
        raise ValueError("outlier_thresh must exceed 1, or be None.")
    if max_shift_scale is not None and (not np.isfinite(max_shift_scale) or max_shift_scale < 0):
        raise ValueError("max_shift_scale must be nonnegative, or None.")
    x = np.asarray(getattr(points, "points", points), float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("points must have shape (N,3).")
    x = np.unique(x[np.isfinite(x).all(1)], axis=0)
    if len(x) < 2:
        raise ValueError("At least two distinct finite points are required.")
    x, s0, radii = _geometry_3(x, support, adaptive, support_k, radius_cap,
                            outlier_k, outlier_thresh, outlier_iters,
                            min_points, chunk_size, workers, base_spacing)
    spacing = s0 * scale if physical_spacing is None else physical_spacing
    sigma = s0 * smooth if physical_sigma is None else physical_sigma
    axes = s0 * support
    use_smooth = (method in {"sdf", "density"} and smooth_iters > 0
                  and np.any(sigma > 0) and max_shift_scale != 0)
    if not use_smooth:
        sigma = np.zeros(3)
    extent = np.zeros(3) if method == "occupancy" else axes * radii.max()
    # The entire dilation of closing must fit, including its later erosion.
    # One continuous Gaussian kernel, plus the permitted sign-change band.
    smooth_margin = 3 * sigma * math.sqrt(smooth_iters)
    if use_smooth and max_shift_scale is not None:
        smooth_margin += s0 * max_shift_scale
    margin = (extent + s0 * (closing + opening) + abs(offset_scale) * s0
              + smooth_margin + 2 * spacing)
    lower, upper = x.min(0), x.max(0)
    if bounds is not None:
        box = np.asarray(bounds, float)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
            raise ValueError("bounds must be finite (2,3), lower <= upper.")
        if np.any(lower < box[0]) or np.any(upper > box[1]):
            raise ValueError("bounds must enclose all retained points.")
        lower, upper = box
    layers = np.ceil(margin / spacing) + pad
    size = np.ceil((upper - lower) / spacing) + 1 + 2 * layers
    if not np.isfinite(size).all() or np.any(size > max_voxels):
        raise ValueError("Grid exceeds max_voxels; increase spacing/spacing_scale or reduce the domain.")
    shape = tuple(int(v) for v in size)
    if math.prod(shape) > max_voxels:
        raise ValueError(f"Grid {shape} has {math.prod(shape):,} voxels, exceeding max_voxels; "
                         "increase spacing/spacing_scale or reduce the domain.")
    origin = lower - layers * spacing
    mask = (np.zeros(shape, bool) if method == "occupancy" else
            _support_mask_3(x, radii, axes, origin, spacing, shape, chunk_size, workers))
    voxel = np.floor((x - origin) / spacing + 0.5).astype(np.int64)
    mask[tuple(voxel.T)] = True  # Retain sampled voxels even on a very coarse grid.
    del radii, voxel
    if opening.any():
        radius = 0.5 * s0 * opening
        mask = _morph(_morph(mask, radius, spacing, "erode"), radius, spacing, "dilate")
    if closing.any():
        radius = 0.5 * s0 * closing
        mask = _morph(_morph(mask, radius, spacing, "dilate"), radius, spacing, "erode")
    if use_smooth:
        shift = None if max_shift_scale is None else s0 * max_shift_scale
        mask = _smooth_mask_3(mask, spacing, sigma, method, smooth_iters,
                            density_level, shift, preserve_volume)
    if offset_scale:
        radius = np.full(3, abs(offset_scale) * s0)
        mask = _morph(mask, radius, spacing, "dilate" if offset_scale > 0 else "erode")
    if hole_scale > 0 and mask.any():
        holes = ndi.binary_fill_holes(mask) & ~mask
        if np.isfinite(hole_scale):
            ids, _ = ndi.label(holes)  # 6-connected background / 26-connected foreground.
            sizes = np.bincount(ids.ravel())
            limit = math.pi / 6 * float(s0 * hole_scale) ** 3 / float(np.prod(spacing))
            valid = sizes <= limit
            valid[0] = False
            holes = valid[ids]
            del ids, sizes, valid
        mask |= holes
        del holes
    if (min_voxels or component_scale or keep_largest) and mask.any():
        ids, _ = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(ids.ravel())
        sizes[0] = 0
        volume_cutoff = math.pi / 6 * float(s0 * component_scale) ** 3 / float(np.prod(spacing))
        keep = sizes >= max(1, min_voxels, volume_cutoff)
        if keep_largest:
            keep &= np.arange(len(sizes)) == sizes.argmax()
        mask = keep[ids]
        del ids, sizes, keep
    image = mask.astype(np.uint8)
    indices = np.flatnonzero(mask)
    del mask
    grid_points = np.empty((len(indices), 3), np.float64)
    score = np.empty(len(indices), np.float32)
    score_image = np.zeros(shape, np.float32)
    flat_score_image = score_image.ravel()
    score_axes = _xyz(s0 * score_axes, "physical score bandwidth", True)
    anchor = x.min(0)
    tree = cKDTree((x - anchor) / score_axes)
    nk = min(int(score_k), len(x))
    batch = max(1, min(chunk_size, 1_000_000 // nk))
    for start in range(0, len(indices), batch):
        sl = slice(start, start + batch)
        ijk = np.column_stack(np.unravel_index(indices[sl], shape))
        grid_points[sl] = origin + ijk * spacing
        q = ((origin - anchor) + ijk * spacing) / score_axes
        distance = tree.query(q, k=list(range(1, nk + 1)), workers=workers)[0]
        np.square(distance, out=distance)
        distance *= -0.5
        np.exp(distance, out=distance)
        score[sl] = distance.mean(axis=1)
        flat_score_image[indices[sl]] = score[sl]
    return {"mask": image, "score_image": score_image,
            "spacing": spacing, "origin": origin,
            "points": grid_points, "score": score}


def image_to_nrrd(result, filename, *, key="mask", space="3D-right-handed", units=None,
                  compression_level=1):
    """Write one selected image and its spatial metadata; return the output path.

    key='mask' writes a uint8 binary mask; key='score_image' writes float32 scores,
    preserving fractional values and writing an explicit little-endian payload.
    Call twice to save both images, which share exactly the same spatial grid.
    No reconstruction is performed and result is not modified. Axis 0 (X) is fastest.
    Read with pynrrd: nrrd.read(path, index_order='F'). The writer itself has
    no pynrrd dependency: it writes a standard NRRD0005 header and gzip payload.

    space describes the INPUT coordinate convention: '3D-right-handed' (generic
    default), '3D-left-handed', 'RAS' or 'LPS' (or their long NRRD names).
    units is an optional string, e.g. 'um' or 'mm', or a three-string XYZ sequence.
    Metadata does not rescale/reorient coordinates. For mm coordinates, convert
    the input points yourself before reconstruction if conversion is needed.
    compression_level is a gzip level from 0 to 9. Filename must end in .nrrd.
    """
    if key not in {"mask", "score_image"}:
        raise ValueError("key must be 'mask' or 'score_image'.")
    image = np.asarray(result[key])
    if image.ndim != 3 or 0 in image.shape:
        raise ValueError(f"result[{key!r}] must have a nonempty 3-D shape.")
    if key == "mask":
        if not np.isin(image, (0, 1)).all():
            raise ValueError("mask must contain only 0 and 1.")
        dtype, nrrd_type = np.dtype("uint8"), "uint8"
    else:
        if image.shape != np.shape(result["mask"]):
            raise ValueError("score_image and mask must have the same shape.")
        if np.iscomplexobj(image) or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError("score_image must contain finite real values in [0,1].")
        dtype, nrrd_type = np.dtype("<f4"), "float"
    spacing = _xyz(result["spacing"], "spacing", True)
    origin = np.asarray(result["origin"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be finite XYZ.")
    space = {"RAS": "right-anterior-superior", "LPS": "left-posterior-superior"}.get(space, space)
    if space not in {"3D-right-handed", "3D-left-handed", "right-anterior-superior", "left-posterior-superior"}:
        raise ValueError("Unsupported coordinate convention.")
    if not isinstance(compression_level, (int, np.integer)) or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer from 0 to 9.")
    path = Path(filename).expanduser()
    if path.suffix.lower() != ".nrrd":
        raise ValueError("filename must end in .nrrd.")
    vector = lambda v: "(" + ",".join(format(float(t), ".17g") for t in v) + ")"
    header = ["NRRD0005", "type: " + nrrd_type, "dimension: 3",
              "sizes: " + " ".join(map(str, image.shape)), "space: " + space,
              "space directions: " + " ".join(vector(v) for v in np.diag(spacing)),
              "space origin: " + vector(origin), "kinds: domain domain domain",
              "centers: cell cell cell", "encoding: gzip"]
    if dtype.itemsize > 1:
        header.append("endian: little")
    if units is not None:
        unit_list = [units] * 3 if isinstance(units, str) else list(units)
        if (len(unit_list) != 3 or any(not isinstance(u, str) or not u or
                any(ord(c) < 32 or ord(c) > 126 or c in '\\"' for c in u)
                for u in unit_list)):
            raise ValueError("units must be one or three printable ASCII strings without quotes/backslashes.")
        header.append('space units: ' + ' '.join('"' + u + '"' for u in unit_list))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n\n").encode("ascii"))
        with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=int(compression_level),
                           mtime=0, filename="") as compressed:
            # Plane batches avoid copying the complete XYZ array to Fortran order.
            for z in range(image.shape[2]):
                compressed.write(image[:, :, z].astype(dtype, copy=False).tobytes(order="F"))
    return str(path)


# pc = adatas['E7.75'].obsm['gw_2_map2'].copy()
# result = points_to_image(
#     pc,

#     method="sdf",

#     # 提高分辨率：减小间距
#     spacing_scale=(0.25, 0.25, 0.375),

#     # 稀疏区域的支持范围
#     support_scale=(1.5, 1.5, 2.5),
#     adaptive=True,
#     radius_cap=2.0,

#     # 增强平滑，z 方向更强
#     smooth_scale=(1.5, 1.5, 3.0),
#     smooth_iters=1,
#     max_shift_scale=4.0,
#     preserve_volume=True,

#     # 去噪；补缺、去突起按需要单独开启
#     outlier_thresh=1.8,
#     outlier_iters=1,
#     min_points=0.01,
#     closing_scale=0,
#     opening_scale=0,
#     hole_scale=0,

#     score_k=8,
#     score_scale=2.0,
# )

# # image_to_nrrd(result, "mask.nrrd", key="mask")
# # image_to_nrrd(result, "score.nrrd", key="score_image")

# grid_points = result["points"]  # (M, 3)
# score = result["score"]         # (M,), float32
# points_with_score = pd.DataFrame(np.column_stack([grid_points, score]), columns=["x", "y", "z", "score"])

# # image_to_nrrd(result, "mask.nrrd")


"""Denoise occupied-region points, reconstruct a volume, and sample a grid/mesh.

Required: numpy, scipy >= 1.6. CPU only.
Mesh only: PyMCubes or scikit-image. No PyVista dependency.

All arrays use (x, y, z) axis order. direction columns are orthonormal world
vectors of the grid axes. No labels/features, slice grouping, or normal input.
Surface-only samples reconstruct a thick shell, not an inferred solid.

Examples
--------
    grid, info = grid_point_cloud(points, return_info=True)
    grid, info = grid_point_cloud(
        points, spacing=(2, 2, 3), axis_scale=(1, 1, 3),
        smooth=(4, 4, 10), smooth_scales=(2, 1), return_info=True,
        return_field=True)
    vertices, faces = grid_to_mesh(info, mc_method="pymcubes")

    # Reuse the resolved physical smoothing scale at higher grid resolution.
    dense = grid_point_cloud(points, spacing=info["spacing"] / 2,
                             axis_scale=(1, 1, 3), smooth=info["smooth"],
                             smooth_scales=(2, 1))

    # PyVista display/save is optional and external to reconstruction.
    # import pyvista as pv
    # cells = np.column_stack([np.full(len(faces), 3), faces])
    # mesh = pv.PolyData(vertices, cells.ravel())
    # mesh.save("tissue.vtp")

    # NRRD: retain xyz array order, including on reading.
    # import nrrd
    # nrrd.write("tissue.nrrd", info["mask"].astype(np.uint8), header={
    #     "space dimension": 3,
    #     "space origin": info["origin"],
    #     "space directions": np.diag(info["spacing"]) @ info["direction"].T,
    # }, index_order="F")
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _vec3(value, name, positive=False):
    value = np.broadcast_to(np.asarray(value, dtype=float), (3,)).copy()
    if not np.isfinite(value).all() or np.any(value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    return value


def _clean_points(x, k, support_k, method, threshold, iters, outlier_radius,
                  min_component, keep_largest, component_scale, radius_cap, workers):
    """Filter in the same anisotropic metric used for support construction."""
    # Cache queries when a stage retains every point.
    tree = dist = ids = None
    for stage in ["outlier"] * iters + ["components", "final"]:
        if len(x) < max(2, support_k + 1):
            raise ValueError("Too few unique points remain; reduce filtering or support_k.")
        if tree is None:
            tree = cKDTree(x)
            kn = min(max(k, support_k), len(x) - 1)
            dist, ids = tree.query(x, k=kn + 1, workers=workers)
            dist, ids = dist[:, 1:], ids[:, 1:]
        if stage == "final":
            return x, tree, dist
        if stage == "outlier":
            if method is None or threshold is None:
                continue
            kk = min(k, dist.shape[1])
            dd, nn = dist[:, :kk], ids[:, :kk]
            if method == "lof":
                reach = np.maximum(dd, dd[:, -1][nn])
                lrd = 1.0 / reach.mean(axis=1)
                keep = (lrd[nn] / lrd[:, None]).mean(axis=1) <= threshold
            elif method == "sor":
                score = dd.mean(axis=1)
                med = np.median(score)
                mad = 1.4826 * np.median(np.abs(score - med))
                keep = score <= med + threshold * max(mad, np.finfo(float).eps * med)
            else:  # radius: require k neighbours within the chosen metric radius
                radius = threshold * np.median(dd[:, -1]) if outlier_radius is None else outlier_radius
                keep = dd[:, -1] <= radius
        else:
            if not (min_component or keep_largest):
                continue
            local = np.median(dist[:, :min(6, dist.shape[1])], axis=1)
            local = np.minimum(local, radius_cap * np.median(local))
            edge = dist <= component_scale * np.minimum(local[:, None], local[ids])
            rows, cols = np.nonzero(edge)
            graph = csr_matrix((np.ones(len(rows), dtype=np.uint8),
                                (rows, ids[rows, cols])), shape=(len(x), len(x)))
            _, labels = connected_components(graph, directed=False)
            sizes = np.bincount(labels)
            cutoff = min_component * sizes.max() if 0 < min_component < 1 else min_component
            valid = sizes >= max(1, cutoff)
            if keep_largest:
                valid &= np.arange(len(sizes)) == sizes.argmax()
            keep = valid[labels]
        if not keep.all():
            x = x[keep]
            tree = None


def _support_mask_2(x, tree, radii, origin, spacing, shape, axis_scale, method,
                  k, bandwidth, density_level, chunk_size, workers):
    """kNN/nearest in batches; KDE via trilinear deposition and convolution."""
    if method == "kde":
        density = np.zeros(shape, dtype=np.float32)
        # x and tree coordinates are metric coordinates; grid is in physical units.
        for start in range(0, len(x), chunk_size):
            q = (x[start:start + chunk_size] * axis_scale - origin) / spacing
            ijk = np.floor(q).astype(np.int64)
            frac = q - ijk
            for offset in np.ndindex(2, 2, 2):
                offset = np.asarray(offset)
                weight = np.prod(np.where(offset, frac, 1 - frac), axis=1)
                np.add.at(density, tuple((ijk + offset).T), weight)
        ndi.gaussian_filter(density, bandwidth / spacing, output=density,
                            mode="constant", truncate=3.0)
        sampled = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), chunk_size):
            q = (x[start:start + chunk_size] * axis_scale - origin) / spacing
            sampled[start:start + chunk_size] = ndi.map_coordinates(
                density, q.T, order=1, mode="constant", prefilter=False)
        level = density_level * float(np.median(sampled))
        return density > level

    total = int(np.prod(shape))
    mask = np.empty(total, dtype=bool)
    neighbors = list(range(1, k + 1)) if method == "knn" else [1]
    radius = float(np.median(radii))
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        q = (origin + ijk * spacing) / axis_scale
        dist, ids = tree.query(q, k=neighbors, workers=workers)
        if method == "nearest":
            mask[start:stop] = dist[:, 0] <= radius
        else:
            dist /= radii[ids]
            np.square(dist, out=dist)
            mask[start:stop] = dist.mean(axis=1) <= 1.0
    return mask.reshape(shape)


def _sdf(mask, spacing):
    phi = ndi.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
    phi -= ndi.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
    return phi


def _smooth_field(mask, spacing, smooth, scales, method, max_shift, preserve_volume):
    """Each scale re-embeds the previous mask; all stages use one original band."""
    phi0 = _sdf(mask, spacing)
    if method is None or not np.any(smooth) or max_shift == 0:
        return phi0, mask
    # Freeze BOTH endpoints explicitly; do not count immutable boundary ties.
    band = np.abs(phi0) < max_shift
    if not band.any():
        return phi0, mask
    target = int(mask[band].sum())
    frozen_inside = (phi0 < 0) & ~band
    frozen_outside = (phi0 > 0) & ~band
    current = mask
    field = phi0
    for scale in scales:
        if method == "sdf":
            base = phi0 if current is mask else _sdf(current, spacing)
            field = ndi.gaussian_filter(base, smooth * scale / spacing,
                                        mode="nearest", truncate=3.0)
        else:  # occupancy: a dimensionless implicit field, negative inside
            field = ndi.gaussian_filter(current.astype(np.float32),
                                        smooth * scale / spacing,
                                        mode="constant", truncate=3.0)
            field *= -1
            field += 0.5
        if preserve_volume:
            values = field[band]
            if target == 0:
                level = values.min()  # field < 0: exclude even the smallest value
            elif target == len(values):
                level = np.nextafter(values.max(), np.float32(np.inf))
            else:
                values.partition(target)
                level = values[target]  # correct count when threshold values are distinct
                below = np.count_nonzero(values < level)
                through = np.count_nonzero(values <= level)
                if target - below > through - target:
                    level = np.nextafter(level, np.float32(np.inf))
            field -= level
        if method == "sdf":
            np.maximum(field, phi0 - max_shift, out=field)
            np.minimum(field, phi0 + max_shift, out=field)
            field[~band] = phi0[~band]
        else:
            field[frozen_inside] = -0.5
            field[frozen_outside] = 0.5
        current = field < 0
    return field, current


def grid_to_mesh(info, mc_method="pymcubes"):
    """Extract a zero level set; return (world_vertices, triangle_indices).

    mc_method : {"pymcubes", "lewiner", "lorensen"}
        PyMCubes or scikit-image backend. No implicit backend fallback.
    info : dict from grid_point_cloud
        Set return_field=True during reconstruction to reuse its smooth field.
        Without field, this function reconstructs a voxel SDF from info['mask'],
        which may produce a rougher surface. Vertices are NOT lattice nodes.
    """
    field = info.get("field")
    if field is None:
        field = _sdf(info["mask"], info["spacing"])
    if not np.any(field < 0) or not np.any(field > 0):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    if mc_method == "pymcubes":
        import mcubes
        vertices, faces = mcubes.marching_cubes(field, 0.0)
        # Exact zero nodes can create coincident vertices/zero-area triangles.
        vertices, remap = np.unique(vertices, axis=0, return_inverse=True)
        faces = remap[faces]
        normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                          vertices[faces[:, 2]] - vertices[faces[:, 0]])
        faces = faces[np.any(normal != 0, axis=1)]
        used, faces = np.unique(faces, return_inverse=True)
        vertices, faces = vertices[used], faces.reshape(-1, 3)
    elif mc_method in ("lewiner", "lorensen"):
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(
            field, 0.0, method=mc_method, gradient_direction="ascent",
            allow_degenerate=False)
    else:
        raise ValueError("mc_method must be 'pymcubes', 'lewiner', or 'lorensen'.")
    vertices = info["origin"] + (vertices * info["spacing"]) @ info["direction"].T
    # Both backends above use inward conventional cross-product winding for
    # our negative-inside field. Reverse once to obtain outward face normals.
    faces = np.asarray(faces, dtype=np.int64)[:, ::-1]
    # A reflected orthonormal grid reverses triangle orientation.
    if np.linalg.det(info["direction"]) < 0:
        faces = faces[:, ::-1]
    return vertices, faces


def grid_point_cloud(
    points, spacing=None, *,
    spacing_scale=1.0, axis_scale=1.0, direction=None,
    method="knn", support_scale=1.25, support_k=8,
    support_radius=None, radius_cap=2.0,
    kernel_scale=2.0, density_level=0.2,
    smooth=None, smooth_scale=1.0, smooth_scales=(1.0,),
    smooth_method="sdf", max_shift=None, preserve_volume=True,
    outlier_method="lof", outlier_k=24, outlier_thresh=1.8,
    outlier_iters=1, outlier_radius=None,
    min_component=0.01, keep_largest=False, component_scale=2.0,
    closing=0.0, opening=0.0, fill_holes=False,
    output="volume", mc_method="pymcubes",
    max_voxels=20_000_000, chunk_size=100_000, workers=-1,
    return_info=False, return_field=False,
):
    """Reconstruct an occupied region from arbitrary 3-D volume samples.

    Parameters
    ----------
    points : (N, 3) array or object with .points
        World coordinates in consistent physical units. Remove nonfinite rows
        and exact duplicates. No original slice coordinates are required.
    spacing : positive scalar or (sx, sy, sz), optional
        Physical grid steps along direction columns. Default: 0.5*h*axis_scale,
        where h is median nearest-neighbour distance in the filtered METRIC
        coordinates q = (points @ direction) / axis_scale.
    spacing_scale : positive scalar or triple, default 1
        Multiply explicit/automatic spacing. Smaller values give a denser grid;
        they do not change physical reconstruction or smoothing parameters.
    axis_scale : positive scalar or triple, default 1
        Distance metric axes: d(x,y)=norm(((x-y) @ direction)/axis_scale).
        (1,1,3) compresses the third axis in all kNN filtering/support queries,
        giving support ellipsoids with a longer third physical semi-axis.
        Relative axis ratios are normally useful; this does not move points.
    direction : orthonormal (3,3) array, optional
        Columns are grid basis vectors in world coordinates. Default identity.
        Supports rotated axes; cannot follow a spatially varying nonlinear warp.
    method : {"knn", "nearest", "kde"}, default "knn"
        knn: mean of k squared distances normalised by neighbour support radii
        <= 1; adaptive to local sampling density, as in the original algorithm.
        nearest: union of fixed-radius ellipsoids, using a nearest-point query.
        kde: trilinear point deposition then Gaussian density convolution;
        faster on dense grids, but sensitive to sampling-density differences.
    support_scale : positive float, default 1.25
        Multiplier of adaptive RMS kNN radii (knn) or their capped median
        (nearest). Larger values bridge gaps AND expand boundaries. Ignored by
        KDE. When support_radius is supplied, it is also multiplied by this.
    support_k : positive int, default 8
        Number of neighbours for local RMS scales and knn support evaluation.
        Must be smaller than the retained unique point count.
    support_radius : positive float, optional
        Fixed radius in metric coordinates, replacing adaptive RMS radii.
        Physical ellipsoid semi-axes are support_radius*support_scale*axis_scale.
        Ignored by KDE. Set support_scale=1 for a direct radius specification.
    radius_cap : float >= 1, default 2
        Cap adaptive radii and graph local scales at this multiple of their
        respective medians. Does not cap an explicitly supplied support_radius.
    kernel_scale : positive scalar or triple, default 2
        KDE bandwidth = h * axis_scale * kernel_scale, in physical grid-axis
        units. Controls reconstruction only; independent of final smoothing.
    density_level : float in (0,1), default 0.2
        KDE threshold relative to median density evaluated at retained points.
        Smaller values expand support; values are not occupancy probabilities.
    smooth : nonnegative scalar or triple, optional
        Physical Gaussian standard deviation along grid axes. Default:
        h*axis_scale. Explicit values override that automatic base.
    smooth_scale : nonnegative scalar or triple, default 1
        Multiply explicit/automatic smooth. Resolved base sigma is returned
        in info['smooth']; it is independent of output spacing.
    smooth_scales : positive float sequence, default (1,)
        Multipliers applied successively to resolved sigma, usually decreasing,
        e.g. (2,1). Each stage thresholds and rebuilds SDF/occupancy before the
        next stage. This is not repeated Gaussian filtering of a fixed field.
        Use (1,) for one stage; more stages do not guarantee a better result.
    smooth_method : {"sdf", "occupancy", None}, default "sdf"
        sdf: Gaussian smoothing of physical-distance embedding (recommended).
        occupancy: Gaussian smoothing of binary occupancy; useful for strong
        suppression of small protrusions, but may remove thin regions faster.
        None disables smoothing. Returned field is negative inside in both
        methods; occupancy field is dimensionless, SDF field is only implicit
        after filtering/clipping, not an exact signed distance.
    max_shift : nonnegative float, optional
        Physical displacement BAND around the initial post-morphology voxel
        SDF, shared by every scale. Default: 2*max(smooth)*max(smooth_scales).
        Nodes with abs(initial SDF) >= max_shift keep their original sign.
        For sdf, field changes are also clipped to +/-max_shift. A bound smaller
        than every spacing freezes the voxel mask. Zero disables smoothing.
        This is not a strict geometric/topological guarantee for the mesh.
    preserve_volume : bool, default True
        Fit the threshold inside the eligible band to approximately retain
        the initial post-morphology voxel count at every stage. Threshold ties
        can prevent an exact count. Does not preserve the unknown true volume,
        topology, or volumes before filtering/closing/opening/hole filling.
    outlier_method : {"lof", "sor", "radius", None}, default "lof"
        lof: local density-ratio rejection, suited to nonuniform sampling.
        sor: reject mean-kNN distances > median + outlier_thresh*1.4826*MAD.
        radius: require outlier_k neighbours within outlier_radius; if absent,
        radius=outlier_thresh*median(kth-neighbour distance). These all use the
        axis_scale metric. None disables point outlier filtering only.
    outlier_k : int >= 2, default 24
        Outlier neighbours; also controls the component graph's candidate edges.
        Automatically capped at N-1 for outliers/graph.
    outlier_thresh : positive float or None, default 1.8
        Method-specific cutoff; LOF normally > 1. Smaller values reject more.
        None disables outlier filtering. Ignored for radius if radius supplied.
    outlier_iters : nonnegative int, default 1
        Recompute neighbours after each removal. 1-2 is usually sufficient;
        many iterations may peel away valid boundaries.
    outlier_radius : positive float, optional
        Radius-filter distance in metric coordinates; ellipsoidal physically.
    min_component : nonnegative number, default 0.01
        Drop POINT graph components smaller than this fraction of the largest
        (0<value<1), or this absolute point count (value>=1). 0 disables cutoff.
        Real disconnected/sparse slices can also be removed; use 0 if needed.
    keep_largest : bool, default False
        Keep only the largest point graph component, also subject to cutoff.
    component_scale : positive float, default 2
        Edge length <= component_scale*min(local_i,local_j). Local scales are
        capped medians of up to six neighbour distances. Larger values permit
        longer edges, provided they are among the kNN candidates. Disabling
        outlier filtering does not disable this independent component stage.
    closing, opening : nonnegative floats, default 0
        Morphological radii in the axis_scale metric. Closing then opening,
        using EDT with sampling=spacing/axis_scale. Closing joins small gaps;
        opening removes thin protrusions. Physical axis radii = value*axis_scale.
        Larger real cavities/thin parts cannot be distinguished from defects.
    fill_holes : bool, default False
        Explicitly fill every enclosed 3-D cavity (6-connected exterior).
    output : {"volume", "boundary", "mask", "mesh"}, default "volume"
        volume/boundary: (M,3) regular world grid points. Boundary uses six
        neighbours and includes cavity walls; it retains grid stair steps.
        mask: bool (nx,ny,nz), avoids materialising M coordinates.
        mesh: (vertices, faces), both (*,3); vertices lie on an interpolated
        isosurface, not on grid nodes. No PyVista object is required.
    mc_method : {"pymcubes", "lewiner", "lorensen"}, default "pymcubes"
        Used only for output='mesh'. PyMCubes or scikit-image respectively.
    max_voxels : positive int, default 20_000_000
        Limit padded grid nodes; raise instead of silently changing spacing.
        Not a total RAM limit: EDT and filtering need multiple dense arrays.
    chunk_size : positive int, default 100_000
        Grid-query and KDE-deposition batch size.
    workers : int, default -1
        cKDTree query threads; -1 means all available CPUs.
    return_info : bool, default False
        Return (result, info). info contains mask, origin, spacing, direction,
        and resolved smooth only; shape is already available as mask.shape.
    return_field : bool, default False
        With return_info=True, also retain the final implicit field to reuse
        with grid_to_mesh. No extra mesh reconstruction query is then needed.

    Notes
    -----
    Arbitrary orientation/nonlinear-transformed coordinates are accepted, but
    one global axis_scale/direction cannot recover spatially varying former
    slice directions. Geometric priors cannot identify whether a large missing
    region is a genuine cavity. KDE is an approximate density estimator, knn
    support can jump when the adaptive kth-neighbour set changes, and all
    methods remain resolution/scale dependent. GPU is not used.

    Memory: O(N*max(outlier_k,support_k) + V + chunk_size*support_k + M).
    Grid and smoothing are dense; only the point-component graph is sparse.
    """
    raw = np.asarray(getattr(points, "points", points), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("points must have shape (N,3).")
    if method not in ("knn", "nearest", "kde") or output not in ("volume", "boundary", "mask", "mesh"):
        raise ValueError("Unknown reconstruction method or output.")
    if outlier_method not in ("lof", "sor", "radius", None) or smooth_method not in ("sdf", "occupancy", None):
        raise ValueError("Unknown outlier_method or smooth_method.")
    for name, value, minimum in [("support_k", support_k, 1), ("outlier_k", outlier_k, 2),
                                  ("outlier_iters", outlier_iters, 0),
                                  ("chunk_size", chunk_size, 1), ("max_voxels", max_voxels, 1)]:
        if not np.isfinite(value) or value != int(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    support_k, outlier_k, outlier_iters, chunk_size = map(int, (support_k, outlier_k, outlier_iters, chunk_size))
    positive = [support_scale, component_scale, radius_cap]
    positive += [v for v in (support_radius, outlier_radius, outlier_thresh) if v is not None]
    if (not np.isfinite(positive).all() or min(positive) <= 0 or radius_cap < 1
            or not np.isfinite([min_component, closing, opening]).all()
            or min(min_component, closing, opening) < 0 or not 0 < density_level < 1):
        raise ValueError("Invalid support, outlier, component, morphology, or density parameter.")
    direction = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    if direction.shape != (3, 3) or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError("direction columns must be orthonormal.")
    axis_scale = _vec3(axis_scale, "axis_scale", positive=True)
    scales = np.atleast_1d(np.asarray(smooth_scales, dtype=float))
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("smooth_scales must be a nonempty sequence of positive values.")

    x = np.unique(raw[np.isfinite(raw).all(axis=1)], axis=0)
    x = (x @ direction) / axis_scale
    x, tree, dist = _clean_points(
        x, outlier_k, support_k, outlier_method, outlier_thresh, outlier_iters,
        outlier_radius, min_component, keep_largest, component_scale, radius_cap, workers)
    step = float(np.median(dist[:, 0]))
    spacing = _vec3(0.5 * step * axis_scale if spacing is None else spacing, "spacing", True)
    spacing *= _vec3(spacing_scale, "spacing_scale", True)
    smooth = _vec3(step * axis_scale if smooth is None else smooth, "smooth")
    smooth *= _vec3(smooth_scale, "smooth_scale")
    bandwidth = step * axis_scale * _vec3(kernel_scale, "kernel_scale", True)
    max_shift = float(2 * smooth.max() * scales.max() if max_shift is None else max_shift)
    if not np.isfinite(max_shift) or max_shift < 0:
        raise ValueError("max_shift must be finite and nonnegative.")
    local = np.sqrt(np.mean(dist[:, :support_k] ** 2, axis=1))
    radii = np.minimum(local, radius_cap * np.median(local))
    if support_radius is not None:
        radii.fill(support_radius)
    radii *= support_scale
    del dist

    support_pad = 3 * bandwidth if method == "kde" else radii.max() * axis_scale
    active_smooth = smooth if smooth_method is not None and max_shift > 0 else np.zeros(3)
    padding = support_pad + 2 * closing * axis_scale + 3 * active_smooth * scales.max()
    padding += (max_shift if np.any(active_smooth) else 0) + 2 * spacing
    physical = x * axis_scale
    origin = np.floor((physical.min(axis=0) - padding) / spacing) * spacing
    dims = np.ceil((physical.max(axis=0) + padding - origin) / spacing) + 1
    size = float(np.prod(dims))
    if not np.isfinite(size) or size > max_voxels:
        raise MemoryError(f"Grid needs {size:,.0f} nodes; max_voxels={max_voxels:,}. Increase spacing or crop/filter points.")
    shape = tuple(dims.astype(np.int64))
    mask = _support_mask_2(x, tree, radii, origin, spacing, shape, axis_scale, method,
                         support_k, bandwidth, density_level, chunk_size, workers)
    del physical, x, tree, radii
    if not mask.any():
        raise ValueError("Empty support mask; reduce spacing or increase support.")

    # Exact EDT-based binary morphology in the chosen ellipsoidal metric.
    sampling = spacing / axis_scale
    if closing:
        mask = ndi.distance_transform_edt(~mask, sampling=sampling) <= closing
        mask = ndi.distance_transform_edt(mask, sampling=sampling) > closing
    if opening:
        mask = ndi.distance_transform_edt(mask, sampling=sampling) > opening
        if mask.any():
            mask = ndi.distance_transform_edt(~mask, sampling=sampling) <= opening
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if not mask.any():
        raise ValueError("Empty support mask; reduce spacing/opening or increase support.")
    field, mask = _smooth_field(mask, spacing, smooth, scales, smooth_method, max_shift, preserve_volume)
    info = dict(mask=mask, origin=origin @ direction.T, spacing=spacing,
                direction=direction.copy(), smooth=smooth)
    if output == "mesh":
        result = grid_to_mesh(dict(info, field=field), mc_method=mc_method)
    elif output == "mask":
        result = mask
    else:
        selected = mask if output == "volume" else mask & ~ndi.binary_erosion(mask)
        ijk = np.column_stack(np.nonzero(selected))
        result = info["origin"] + (ijk * spacing) @ direction.T
    if return_info:
        if return_field:
            info["field"] = field
        return result, info
    return result
