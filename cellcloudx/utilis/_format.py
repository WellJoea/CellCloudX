import numpy as np
import numpy as np
import pandas as pd

def pv2trim(mesh):
    import trimesh
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    else:
        import pyvista as pv
        if isinstance(mesh, pv.PolyData):
            tmesh = mesh.extract_surface().triangulate()
            faces_as_array = tmesh.faces.reshape((tmesh.n_cells, 4))[:, 1:]
            tmesh = trimesh.Trimesh(tmesh.points, faces_as_array) 
            return tmesh

def trim2pv(mesh):
    import pyvista as pv
    return pv.wrap(mesh)

def mesh_voxelize(mesh, density= None, check_surface=True):
    import pyvista as pv
    if mesh.__class__.__name__ == "Trimesh":
        mesh = pv2trim(mesh)
    if density is None:
        density = mesh.length/100
    return pv.voxelize(mesh, density=density, check_surface=check_surface)

def ras_lps_inverse(inx):
    ivmat = np.array([[-1,-1,1]])
    return inx * ivmat


def mesh_voxel_labels(
    meshes,
    voxel_size,
    chunk=200_000,
    method="cell_locator",
    repair=True,
    repair_max=0.5,
    check_surface=False,
):
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import pyvista as pv

    # -------------------------------------------------------------
    # Input
    # -------------------------------------------------------------
    if isinstance(meshes, (list, tuple)):
        meshes = dict(enumerate(meshes))

    if not isinstance(meshes, dict) or not meshes:
        raise ValueError("meshes must be a non-empty dict/list/tuple")

    spacing = np.asarray(voxel_size, dtype=float)
    spacing = np.repeat(spacing, 3) if spacing.ndim == 0 else spacing.ravel()

    if (
        spacing.size != 3
        or np.any(~np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise ValueError(
            "voxel_size must be a positive scalar or length-3 array"
        )

    if not isinstance(chunk, (int, np.integer)) or chunk <= 0:
        raise ValueError("chunk must be a positive integer")

    if method not in ("cell_locator", "signed_distance"):
        raise ValueError(
            "method must be 'cell_locator' or 'signed_distance'"
        )

    if repair and (
        not np.isfinite(repair_max)
        or repair_max <= 0
    ):
        raise ValueError("repair_max must be > 0")

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def prepare_mesh(mesh):
        if isinstance(mesh, (str, os.PathLike)):
            mesh = pv.read(mesh)

        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()

        mesh = (
            mesh
            .extract_surface(algorithm=None)
            .triangulate()
            .clean()
        )

        return mesh

    def obb_volume(mesh):
        """
        PCA-oriented bounding-box volume.

        This metric is independent of whether a mesh is open or closed,
        and is therefore used only for mesh priority ordering.
        """
        pts = np.asarray(mesh.points, dtype=np.float64)

        if len(pts) < 3:
            return 0.0

        # Covariance without allocating centered Nx3 matrix.
        mean = pts.mean(axis=0)
        cov = pts.T @ pts / len(pts) - np.outer(mean, mean)

        _, axes = np.linalg.eigh(cov)

        pmin = np.full(3, np.inf)
        pmax = np.full(3, -np.inf)

        # Avoid allocating a very large projected coordinate matrix.
        block = max(chunk, 100_000)

        for start in range(0, len(pts), block):
            p = pts[start:start + block] @ axes

            pmin = np.minimum(
                pmin,
                p.min(axis=0),
            )
            pmax = np.maximum(
                pmax,
                p.max(axis=0),
            )

        extent = pmax - pmin

        # Structures thinner than voxel resolution are treated as
        # at least one voxel thick.
        extent = np.maximum(
            extent,
            spacing.min(),
        )

        return float(np.prod(extent))

    def repair_mesh(mesh):
        """
        Progressively close an open surface.

        Returns
        -------
        mesh
        n_open_before
        n_open_after
        """
        n_before = int(mesh.n_open_edges)

        if n_before == 0 or not repair:
            return mesh, n_before, n_before

        b = np.asarray(mesh.bounds, dtype=float)
        extent = b[[1, 3, 5]] - b[[0, 2, 4]]

        scale = float(np.linalg.norm(extent))

        if not np.isfinite(scale) or scale <= 0:
            return mesh, n_before, n_before

        # Progressive hole closing:
        #
        # small holes -> medium holes -> large holes
        #
        # e.g. repair_max=0.5:
        # 0.01 -> ... -> 0.5 × mesh diagonal
        start_ratio = min(0.01, repair_max)

        if repair_max <= start_ratio:
            ratios = np.asarray([repair_max])
        else:
            ratios = np.geomspace(
                start_ratio,
                repair_max,
                6,
            )

        current = mesh
        n_open = n_before

        for ratio in ratios:
            try:
                candidate = (
                    current
                    .fill_holes(scale * float(ratio))
                    .triangulate()
                    .clean()
                )

                n_candidate = int(
                    candidate.n_open_edges
                )

            except Exception:
                continue

            # Only accept a repair if it actually improves closure.
            if n_candidate < n_open:
                current = candidate
                n_open = n_candidate

            if n_open == 0:
                break

        return current, n_before, n_open

    def empty_result():
        return pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "mesh_id",
                "mesh_name",
            ]
        )

    # -------------------------------------------------------------
    # Prepare / repair all meshes
    # -------------------------------------------------------------
    prepared = {}
    open_meshes = []

    for name, mesh in meshes.items():

        mesh = prepare_mesh(mesh)

        if mesh.n_points == 0 or mesh.n_cells == 0:
            warnings.warn(
                f"Skip empty mesh: {name!r}"
            )
            continue

        if not np.all(np.isfinite(mesh.points)):
            warnings.warn(
                f"Skip mesh containing NaN/Inf: {name!r}"
            )
            continue

        # Priority is calculated BEFORE artificial surface repair.
        priority = obb_volume(mesh)

        mesh, n_before, n_after = repair_mesh(mesh)

        if n_after > 0:
            open_meshes.append(
                (name, n_before, n_after)
            )

            if check_surface:
                raise ValueError(
                    f"mesh {name!r} is not closed after repair: "
                    f"{n_before} -> {n_after} open edges"
                )

        prepared[name] = {
            "mesh": mesh,
            "priority": priority,
        }

    if not prepared:
        return empty_result()

    # One summary warning instead of one warning per mesh.
    if open_meshes and not check_surface:
        msg = ", ".join(
            f"{name}({before}->{after})"
            for name, before, after in open_meshes
        )

        warnings.warn(
            "Some meshes remain open after repair and will be "
            "processed with check_surface=False: "
            + msg
        )

    # -------------------------------------------------------------
    # Small structure first
    #
    # All meshes use the SAME metric:
    # PCA-oriented bounding-box volume.
    # -------------------------------------------------------------
    ordered = sorted(
        prepared.items(),
        key=lambda x: x[1]["priority"],
    )

    names = [
        name
        for name, _ in ordered
    ]

    surfaces = [
        (name, info["mesh"])
        for name, info in ordered
    ]

    # -------------------------------------------------------------
    # Global voxel grid
    # -------------------------------------------------------------
    bounds = np.asarray(
        [
            mesh.bounds
            for _, mesh in surfaces
        ],
        dtype=float,
    )

    xyz_min = bounds[:, [0, 2, 4]].min(axis=0)
    xyz_max = bounds[:, [1, 3, 5]].max(axis=0)

    axes = [
        np.arange(
            xyz_min[d] + spacing[d] / 2,
            xyz_max[d],
            spacing[d],
            dtype=float,
        )
        for d in range(3)
    ]

    if any(len(x) == 0 for x in axes):
        return empty_result()

    nx, ny, nz = map(len, axes)

    n_voxels = (
        int(nx)
        * int(ny)
        * int(nz)
    )

    # Only store global labels.
    # Do not create the entire Nx3 coordinate matrix.
    labels = np.full(
        n_voxels,
        -1,
        dtype=np.int32,
    )

    # -------------------------------------------------------------
    # Point-in-mesh
    # -------------------------------------------------------------
    for mesh_id, (name, mesh) in enumerate(surfaces):

        b = mesh.bounds

        # Restrict computation to the mesh bounding box.
        ix0 = np.searchsorted(
            axes[0],
            b[0],
            side="left",
        )
        ix1 = np.searchsorted(
            axes[0],
            b[1],
            side="right",
        )

        iy0 = np.searchsorted(
            axes[1],
            b[2],
            side="left",
        )
        iy1 = np.searchsorted(
            axes[1],
            b[3],
            side="right",
        )

        iz0 = np.searchsorted(
            axes[2],
            b[4],
            side="left",
        )
        iz1 = np.searchsorted(
            axes[2],
            b[5],
            side="right",
        )

        sx = ix1 - ix0
        sy = iy1 - iy0
        sz = iz1 - iz0

        if sx <= 0 or sy <= 0 or sz <= 0:
            continue

        n_local = (
            int(sx)
            * int(sy)
            * int(sz)
        )

        # ---------------------------------------------------------
        # Chunked voxel generation
        # ---------------------------------------------------------
        for start in range(
            0,
            n_local,
            chunk,
        ):
            stop = min(
                start + chunk,
                n_local,
            )

            q = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            # Convert local flattened index -> ix, iy, iz.
            iz = q % sz
            q //= sz

            iy = q % sy
            ix = q // sy

            ix += ix0
            iy += iy0
            iz += iz0

            # Global flattened index.
            flat = (
                (ix * ny + iy)
                * nz
                + iz
            )

            # Smaller meshes were processed first.
            # Existing labels must never be overwritten.
            free = labels[flat] < 0

            if not np.any(free):
                continue

            flat = flat[free]
            ix = ix[free]
            iy = iy[free]
            iz = iz[free]

            points = np.column_stack(
                (
                    axes[0][ix],
                    axes[1][iy],
                    axes[2][iz],
                )
            )

            selected = (
                pv.PolyData(points)
                .select_interior_points(
                    mesh,
                    method=method,
                    check_surface=False,
                )
            )

            inside = np.asarray(
                selected["selected_points"]
            ).astype(
                bool,
                copy=False,
            )

            if np.any(inside):
                labels[
                    flat[inside]
                ] = mesh_id

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------
    valid_idx = np.flatnonzero(
        labels >= 0
    )

    if valid_idx.size == 0:
        return empty_result()

    codes = labels[valid_idx]

    q = valid_idx.astype(
        np.int64,
        copy=True,
    )

    iz = q % nz
    q //= nz

    iy = q % ny
    ix = q // ny

    df = pd.DataFrame({
        "x": axes[0][ix],
        "y": axes[1][iy],
        "z": axes[2][iz],
        "mesh_id": codes,
    })

    df["mesh_name"] = pd.Categorical.from_codes(
        codes,
        categories=names,
    )

    return df


def mesh_voxel_labels0(
    meshes,
    voxel_size,
    chunk=200_000,
    method="cell_locator",
):
    import numpy as np
    import pandas as pd
    import pyvista as pv

    if isinstance(meshes, (list, tuple)):
        meshes = dict(enumerate(meshes))
    if not isinstance(meshes, dict) or not meshes:
        raise ValueError("meshes must be a non-empty dict/list/tuple")

    spacing = np.asarray(voxel_size, dtype=float)
    spacing = np.repeat(spacing, 3) if spacing.ndim == 0 else spacing
    if spacing.size != 3 or np.any(spacing <= 0):
        raise ValueError("voxel_size must be positive scalar or length 3")


    ms = {
        name: (
            pv.read(mesh) if isinstance(mesh, str) else mesh
        ).extract_surface(
            algorithm=None
        ).triangulate().clean()
        for name, mesh in meshes.items()
    }


    ordered = sorted(
        ms.items(),
        key=lambda x: abs(x[1].volume),
    )
    names = [name for name, _ in ordered]


    bounds = np.asarray([mesh.bounds for _, mesh in ordered])
    xyz_min = bounds[:, [0, 2, 4]].min(axis=0)
    xyz_max = bounds[:, [1, 3, 5]].max(axis=0)

    axes = [
        np.arange(
            xyz_min[i] + spacing[i] / 2,
            xyz_max[i],
            spacing[i],
        )
        for i in range(3)
    ]

    if any(len(axis) == 0 for axis in axes):
        return pd.DataFrame(
            columns=["x", "y", "z", "mesh_id", "mesh_name"]
        )

    points = np.stack(
        np.meshgrid(*axes, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)

    labels = np.full(len(points), -1, dtype=np.int32)

    for mesh_id, (_, mesh) in enumerate(ordered):
        b = mesh.bounds


        idx = np.flatnonzero(
            (labels < 0)
            & (points[:, 0] >= b[0])
            & (points[:, 0] <= b[1])
            & (points[:, 1] >= b[2])
            & (points[:, 1] <= b[3])
            & (points[:, 2] >= b[4])
            & (points[:, 2] <= b[5])
        )

        for start in range(0, len(idx), chunk):
            ii = idx[start:start + chunk]

            selected = pv.PolyData(
                points[ii]
            ).select_interior_points(
                mesh,
                method=method,
                check_surface=False,
            )

            inside = np.asarray(
                selected["selected_points"],
                dtype=bool,
            )

            labels[ii[inside]] = mesh_id

    valid = labels >= 0
    codes = labels[valid]

    df = pd.DataFrame(
        points[valid],
        columns=["x", "y", "z"],
    )
    df["mesh_id"] = codes
    df["mesh_name"] = pd.Categorical.from_codes(
        codes,
        categories=names,
    )

    return df