from scipy.interpolate import interpn, RBFInterpolator
import skimage as ski
from typing import Optional, Union
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator, interp1d
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    uniform_filter,
    distance_transform_edt,binary_dilation, generate_binary_structure
)
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, shift, uniform_filter
from skimage.color import rgb2hed, rgb2lab
from skimage.feature import SIFT, local_binary_pattern
from skimage.filters import gabor, gaussian, scharr_h, scharr_v
from skimage.util import img_as_float32

from ..utilis._arrays import list_iter


def image_to_points(
    image,
    mask=None,
    step=4,
    features=("appearance", "gradient", "lbp"),
    color_space="lab",
    sigma=1.0,
    patch_size=7,
    lbp_points=8,
    lbp_radius=1,
    lbp_window=7,
    gabor_frequencies=(0.1, 0.2),
    gabor_angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    normal=True,
    ngf_eta=None,
    mind_radius=2.0,
    mind_angles=8,
    mind_patch_size=3,
    mind_rotation_invariant=True,
    rihog_bins=8,
    rihog_window=9,
    hessian_sigmas=(1.0, 2.0, 4.0),
    sift_kwargs={},
):
    """
    Convert an image into sampled ``[x, y]`` points and local descriptors.

    Except in SIFT mode, points are sampled on a uniform pixel grid and the
    requested descriptor blocks are concatenated in ``features`` order.

    Parameters
    ----------
    image : (H, W) or (H, W, C) array
        Grayscale or multichannel input image. Integer images are converted
        with ``img_as_float32``; floating-point values are preserved. For
        ``color_space='lab'`` or ``'hed'``, the first three channels are
        interpreted as RGB and floating RGB values must lie in ``[0, 1]``.
        With ``color_space='raw'``, all channels and their original floating
        value range are retained. All values must be finite.
    mask : (H, W) bool array, optional
        Foreground mask in image coordinates. Only grid points or SIFT
        keypoints whose mask value is true are retained. Non-Boolean inputs
        are converted to Boolean, so threshold probability masks before use.
        Use a tighter mask to suppress irrelevant background and a wider mask
        to retain uncertain tissue boundaries. If ``None``, use the full image.
    step : int, default=4
        Positive grid spacing in pixels. The number of sampled points is
        approximately the foreground area divided by ``step**2``. Decrease it
        for denser, more precise sampling; increase it for faster processing
        and coarser representation. A useful estimate for a target of ``N``
        points is ``step ~= sqrt(foreground_area / N)``. Ignored in SIFT mode.
    features : str or sequence of str,
        default=('appearance', 'gradient', 'lbp')
        Descriptor blocks to return, concatenated in the supplied order:

        - ``'appearance'``: smoothed grayscale, Lab, HED, or raw channel values;
        - ``'structure'``: one scalar structural-intensity value for every image;
        - ``'gradient'``: one Scharr gradient-magnitude value;
        - ``'ngf'``: a 3-D sign-insensitive normalized-gradient tensor;
        - ``'patch'``: a zero-mean, L2-normalized local gray patch;
        - ``'lbp'``: a local uniform-LBP histogram;
        - ``'gabor'``: orientation-pooled responses, one per frequency;
        - ``'mind'``: modality-independent local self-similarity values;
        - ``'rihog'``: rotation-insensitive Fourier gradient histogram;
        - ``'hessian'``: absolute Hessian eigenvalues at multiple scales;
        - ``'sift'``: sparse 128-dimensional SIFT descriptors.

        ``'sift'`` must be used alone. ``'appearance'`` retains all selected
        channels and therefore is not directly comparable between RGB/Lab and
        grayscale inputs. Use ``'structure'`` for a common one-column intensity
        block, or omit intensity and use ``gradient/ngf/mind/rihog/hessian`` for
        cross-platform images. Use identical feature parameters for all images.
    color_space : {'lab', 'raw', 'hed'}, default='lab'
        Representation used for a multichannel image. ``'lab'`` converts RGB
        to perceptual Lab and uses normalized L as the scalar image;
        ``'hed'`` performs H&E color deconvolution and uses the hematoxylin
        channel as the scalar image; ``'raw'`` retains every channel and uses
        their mean as the scalar image. The scalar image supplies gradient,
        patch, LBP, Gabor, and SIFT features. Use ``'lab'`` for ordinary RGB,
        ``'hed'`` for H&E sections, and ``'raw'`` for independent fluorescence
        or measurement channels. It does not alter a 2-D grayscale image.
    sigma : float, default=1.0
        Standard deviation, in pixels, of Gaussian smoothing applied before
        dense feature extraction. Use ``0`` to disable smoothing. Increase it
        to suppress pixel noise and stabilize gradients; decrease it to retain
        fine boundaries. Values near ``0.25--0.5 * step`` are a useful start.
        This parameter controls only the image feature scale.
        Ignored in SIFT mode.
    patch_size : int, default=7
        Positive odd side length, in pixels, of each ``'patch'`` descriptor.
        Its output dimension is ``patch_size**2``. Larger values capture wider
        structures and tolerate small translations but reduce localization and
        increase memory. A common choice is approximately ``2 * step + 1``.
        Used only when ``'patch'`` is requested.
    lbp_points : int, default=8
        Number of circular neighbors in the uniform-LBP code. The LBP output
        dimension is ``lbp_points + 2``. More points describe angular texture
        more finely but cost more and can be more noise-sensitive. A common
        relation is ``lbp_points ~= 8 * lbp_radius``.
    lbp_radius : float, default=1
        Positive sampling radius, in pixels, of the LBP neighborhood. Increase
        it to describe coarser texture and decrease it for finer texture.
    lbp_window : int, default=7
        Positive odd side length, in pixels, of the window used to pool local
        LBP-code proportions. Larger windows give smoother, more
        translation-tolerant histograms but weaker spatial localization. It
        should normally be at least ``2 * lbp_radius + 1``.
    gabor_frequencies : sequence of float, default=(0.1, 0.2)
        Spatial frequencies in cycles per pixel, each in ``(0, 0.5]``. One
        output column is produced per frequency after orientation max-pooling.
        A frequency ``f`` targets a texture wavelength of approximately
        ``1 / f`` pixels. Add lower frequencies for coarser texture and higher
        frequencies for finer texture.
    gabor_angles : sequence of float
        Gabor orientations in radians. The default uses 0, 45, 90, and 135
        degrees. Responses are max-pooled over angles, so adding angles improves
        approximate rotation tolerance and increases computation time without
        increasing descriptor dimension. Used only when ``'gabor'`` is requested.
    ngf_eta : float or None, default=None
        Positive edge-noise scale for ``'ngf'``. If ``None``, use 10% of the
        90th percentile of foreground gradient magnitudes. The adaptive value
        makes NGF insensitive to global intensity scaling. NGF is also
        insensitive to contrast reversal, but its tensor axes still follow the
        image x/y axes.
    mind_radius : float, default=2.0
        Positive radius, in pixels, of the circular MIND sampling ring.
    mind_angles : int, default=8
        Number of evenly spaced offsets on the MIND ring and therefore the
        ``'mind'`` output dimension.
    mind_patch_size : int, default=3
        Positive odd window size used to pool squared patch differences for
        MIND. Larger windows suppress platform noise but blur fine structures.
    mind_rotation_invariant : bool, default=True
        If true, sort the ring responses at every point. This removes their
        angular order and makes MIND approximately invariant to in-plane
        rotation and reflection while retaining local self-similarity values.
    rihog_bins : int, default=8
        Number of unsigned orientation bins before the Fourier magnitude is
        computed. The ``'rihog'`` dimension is ``rihog_bins // 2 + 1``.
    rihog_window : int, default=9
        Positive odd window size used to pool the local gradient histogram.
    hessian_sigmas : sequence of float, default=(1.0, 2.0, 4.0)
        Positive Gaussian scales in pixels. ``'hessian'`` returns the two
        absolute, magnitude-sorted Hessian eigenvalues per scale, for a total
        dimension of ``2 * len(hessian_sigmas)``. These values are invariant to
        in-plane axis rotation and to intensity-sign reversal.
    normal : bool, default=True
        If true, normalize each dense descriptor block before concatenation.
        Appearance, structure, gradient, Gabor, and Hessian columns receive a
        per-image robust median-MAD scaling and division by the square root of
        their block dimension. Patch, MIND, and RIHOG rows are L2-normalized;
        LBP receives a Hellinger transform and row normalization; NGF keeps its
        bounded tensor geometry and is divided by the square root of dimension.
        Every block is finally multiplied by ``1 / sqrt(number_of_blocks)``.
        Set false when shared scaling will be applied later. Intrinsic patch,
        NGF, MIND, RIHOG, and SIFT descriptor construction is always retained.

    Returns
    -------
    points : (N, 2) float32 array
        Sample or keypoint coordinates in ``[x, y]`` pixel order.
    descriptors : (N, K) float32 array
        Descriptor rows aligned with ``points``. For dense mode, ``K`` is the
        sum of the requested block dimensions; for SIFT, ``K=128``.

    Notes
    -----
    Use the same extraction parameters and the same physical image scale for
    all images being compared. After extracting ``FX`` and ``FY`` with
    ``normal=True``, use
    :func:`normalize_feature_blocks` with ``pre_normalized=True`` to impose one
    shared FX/FY scale and optional block weights. If joint
    scaling is always available, extracting with ``normal=False`` and calling
    that function with ``pre_normalized=False`` avoids the preliminary
    per-image scaling.

    The function does not infer pixel size. Images from different scanners or
    pyramid levels should first be resampled to a common physical pixel size;
    otherwise pixel-valued radii, windows, frequencies, and Hessian scales do
    not describe the same tissue scale even though descriptor dimensions match.
    """
    image = img_as_float32(np.asarray(image))
    features = (features,) if isinstance(features, str) else tuple(features)

    if not features:
        raise ValueError("features cannot be empty")
    if len(set(features)) != len(features):
        raise ValueError("features cannot contain duplicates")
    if color_space not in {"lab", "raw", "hed"}:
        raise ValueError("color_space must be 'lab', 'raw', or 'hed'")
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values")
    if not isinstance(normal, (bool, np.bool_)):
        raise TypeError("normal must be a boolean")

    if image.ndim == 2:
        raw_gray = image
        appearance = image[..., None]

    elif image.ndim == 3 and image.shape[-1] > 0:
        if color_space == "raw":
            appearance = image
            raw_gray = image.mean(axis=-1)

        elif color_space in {"lab", "hed"}:
            if image.shape[-1] < 3:
                raise ValueError(
                    f"color_space={color_space!r} requires at least 3 channels"
                )

            rgb = image[..., :3]
            if rgb.min() < 0 or rgb.max() > 1:
                raise ValueError(
                    "floating RGB values must be in [0, 1] for Lab/HED conversion"
                )

            if color_space == "lab":
                appearance = rgb2lab(rgb).astype(np.float32)
                raw_gray = appearance[..., 0] / 100.0
            else:
                appearance = rgb2hed(rgb).astype(np.float32)
                raw_gray = appearance[..., 0]

        else:
            raise ValueError("color_space must be 'lab', 'raw', or 'hed'")

    else:
        raise ValueError("image must have shape (H, W) or (H, W, C)")

    h, w = raw_gray.shape
    if h == 0 or w == 0:
        raise ValueError("image spatial dimensions cannot be empty")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError("mask shape must match image shape")

    # Sparse SIFT point cloud.
    if features == ("sift",):
        if mask is not None and not np.any(mask):
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32),
            )

        support = raw_gray[mask] if mask is not None else raw_gray.ravel()
        lo, hi = np.percentile(support, (1, 99))
        sift_image = np.clip(
            (raw_gray - lo) / (hi - lo + 1e-8),
            0,
            1,
        ).astype(np.float32)

        sift = SIFT(**sift_kwargs)
        try:
            sift.detect_and_extract(sift_image)
        except RuntimeError as exc:
            if "found no features" not in str(exc).lower():
                raise
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32),
            )

        yx = getattr(sift, "positions", sift.keypoints)
        ij = np.rint(yx).astype(np.intp)
        keep = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < h)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < w)
        )

        if mask is not None:
            valid = np.flatnonzero(keep)
            keep[valid] = mask[ij[valid, 0], ij[valid, 1]]

        return (
            yx[keep, ::-1].astype(np.float32),
            sift.descriptors[keep].astype(np.float32),
        )

    if "sift" in features:
        raise ValueError("'sift' must be used alone")

    valid_features = {
        "appearance",
        "structure",
        "gradient",
        "ngf",
        "patch",
        "lbp",
        "gabor",
        "mind",
        "rihog",
        "hessian",
    }
    unknown = set(features) - valid_features
    if unknown:
        raise ValueError(f"unknown feature: {sorted(unknown)!r}")

    if "patch" in features and (
        not isinstance(patch_size, (int, np.integer))
        or patch_size < 1
        or patch_size % 2 != 1
    ):
        raise ValueError("patch_size must be a positive odd integer")

    if "lbp" in features:
        if not isinstance(lbp_points, (int, np.integer)) or lbp_points < 1:
            raise ValueError("lbp_points must be a positive integer")
        if not np.isfinite(lbp_radius) or lbp_radius <= 0:
            raise ValueError("lbp_radius must be finite and positive")
        if (
            not isinstance(lbp_window, (int, np.integer))
            or lbp_window < 1
            or lbp_window % 2 != 1
        ):
            raise ValueError("lbp_window must be a positive odd integer")

    if "gabor" in features:
        gabor_frequencies = tuple(gabor_frequencies)
        gabor_angles = tuple(gabor_angles)
        if len(gabor_frequencies) == 0 or len(gabor_angles) == 0:
            raise ValueError("gabor_frequencies and gabor_angles cannot be empty")
        if any(not np.isfinite(f) or f <= 0 or f > 0.5 for f in gabor_frequencies):
            raise ValueError("Gabor frequencies must be in (0, 0.5]")
        if any(not np.isfinite(theta) for theta in gabor_angles):
            raise ValueError("Gabor angles must be finite")

    if "ngf" in features and (
        ngf_eta is not None
        and (not np.isfinite(ngf_eta) or ngf_eta <= 0)
    ):
        raise ValueError("ngf_eta must be None or finite and positive")

    if "mind" in features:
        if not np.isfinite(mind_radius) or mind_radius <= 0:
            raise ValueError("mind_radius must be finite and positive")
        if not isinstance(mind_angles, (int, np.integer)) or mind_angles < 4:
            raise ValueError("mind_angles must be an integer >= 4")
        if (
            not isinstance(mind_patch_size, (int, np.integer))
            or mind_patch_size < 1
            or mind_patch_size % 2 != 1
        ):
            raise ValueError("mind_patch_size must be a positive odd integer")
        if not isinstance(mind_rotation_invariant, (bool, np.bool_)):
            raise TypeError("mind_rotation_invariant must be a boolean")

    if "rihog" in features:
        if not isinstance(rihog_bins, (int, np.integer)) or rihog_bins < 4:
            raise ValueError("rihog_bins must be an integer >= 4")
        if (
            not isinstance(rihog_window, (int, np.integer))
            or rihog_window < 1
            or rihog_window % 2 != 1
        ):
            raise ValueError("rihog_window must be a positive odd integer")

    if "hessian" in features:
        hessian_sigmas = tuple(hessian_sigmas)
        if len(hessian_sigmas) == 0 or any(
            not np.isfinite(value) or value <= 0
            for value in hessian_sigmas
        ):
            raise ValueError("hessian_sigmas must contain positive finite values")

    # Uniform grid sampling.
    yy, xx = np.meshgrid(
        np.arange(step // 2, h, step),
        np.arange(step // 2, w, step),
        indexing="ij",
    )
    yy, xx = yy.ravel(), xx.ravel()

    if mask is not None:
        keep = mask[yy, xx]
        yy, xx = yy[keep], xx[keep]

    if len(xx) == 0:
        dims = {
            "appearance": appearance.shape[-1],
            "structure": 1,
            "gradient": 1,
            "ngf": 3,
            "patch": patch_size**2,
            "lbp": lbp_points + 2,
            "gabor": len(gabor_frequencies),
            "mind": mind_angles,
            "rihog": rihog_bins // 2 + 1,
            "hessian": 2 * len(hessian_sigmas),
        }
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, sum(dims[name] for name in features)), dtype=np.float32),
        )

    gray = gaussian(
        raw_gray,
        sigma=sigma,
        preserve_range=True,
    ).astype(np.float32)

    if "appearance" in features:
        appearance = gaussian(
            appearance,
            sigma=sigma,
            channel_axis=-1,
            preserve_range=True,
        ).astype(np.float32)

    if set(features) & {"gradient", "ngf", "rihog"}:
        gx = scharr_v(gray).astype(np.float32)
        gy = scharr_h(gray).astype(np.float32)
        gradient = np.hypot(gx, gy).astype(np.float32)

    blocks = []

    for name in features:
        if name == "appearance":
            blocks.append(appearance[yy, xx])

        elif name == "structure":
            blocks.append(gray[yy, xx, None])

        elif name == "gradient":
            blocks.append(gradient[yy, xx, None])

        elif name == "ngf":
            if ngf_eta is None:
                support = gradient[mask] if mask is not None else gradient.ravel()
                eta = max(0.1 * np.percentile(support, 90), 1e-6)
            else:
                eta = float(ngf_eta)

            gxs = gx[yy, xx]
            gys = gy[yy, xx]
            denominator = gxs * gxs + gys * gys + eta * eta
            blocks.append(np.column_stack([
                gxs * gxs / denominator,
                np.sqrt(2.0) * gxs * gys / denominator,
                gys * gys / denominator,
            ]))

        elif name == "patch":
            radius = patch_size // 2
            padded = np.pad(gray, radius, mode="reflect")

            patches = np.lib.stride_tricks.sliding_window_view(
                padded,
                (patch_size, patch_size),
            )[yy, xx].reshape(len(xx), patch_size * patch_size)

            patches -= patches.mean(axis=1, keepdims=True)
            patches /= np.linalg.norm(
                patches,
                axis=1,
                keepdims=True,
            ) + 1e-8
            blocks.append(patches)

        elif name == "lbp":
            support = gray[mask] if mask is not None else gray.ravel()
            lo, hi = np.percentile(support, (1, 99))
            gray8 = np.clip(
                (gray - lo) / (hi - lo + 1e-8),
                0,
                1,
            )

            code = local_binary_pattern(
                (255 * gray8).astype(np.uint8),
                lbp_points,
                lbp_radius,
                method="uniform",
            )

            hist = np.column_stack([
                uniform_filter(
                    (code == value).astype(np.float32),
                    size=lbp_window,
                    mode="reflect",
                )[yy, xx]
                for value in range(lbp_points + 2)
            ])
            blocks.append(hist)

        elif name == "gabor":
            responses = []

            for frequency in gabor_frequencies:
                pooled = np.zeros_like(gray)

                for theta in gabor_angles:
                    real, imag = gabor(
                        gray,
                        frequency,
                        theta=theta,
                    )
                    magnitude = np.hypot(real, imag).astype(np.float32)
                    np.maximum(pooled, magnitude, out=pooled)

                responses.append(pooled[yy, xx])

            blocks.append(np.column_stack(responses))

        elif name == "mind":
            mind = np.empty((len(xx), mind_angles), dtype=np.float32)

            for iangle in range(mind_angles):
                theta = 2 * np.pi * iangle / mind_angles
                shifted = shift(
                    gray,
                    shift=(
                        mind_radius * np.sin(theta),
                        mind_radius * np.cos(theta),
                    ),
                    order=1,
                    mode="reflect",
                    prefilter=False,
                )
                distance = uniform_filter(
                    np.square(gray - shifted),
                    size=mind_patch_size,
                    mode="reflect",
                )
                mind[:, iangle] = distance[yy, xx]

            variance = mind.mean(axis=1, keepdims=True)
            floor = max(float(np.median(variance)) * 1e-6, 1e-12)
            mind = np.exp(-mind / np.maximum(variance, floor))
            mind /= np.maximum(mind.max(axis=1, keepdims=True), 1e-6)

            if mind_rotation_invariant:
                mind.sort(axis=1)
                mind = mind[:, ::-1]

            blocks.append(mind)

        elif name == "rihog":
            angle = np.mod(np.arctan2(gy, gx), np.pi)
            bin_position = angle * (rihog_bins / np.pi)
            low = np.floor(bin_position).astype(np.intp) % rihog_bins
            fraction = bin_position - np.floor(bin_position)
            high = (low + 1) % rihog_bins
            histogram = np.empty((len(xx), rihog_bins), dtype=np.float32)

            for ibin in range(rihog_bins):
                votes = gradient * (
                    (low == ibin) * (1.0 - fraction)
                    + (high == ibin) * fraction
                )
                pooled = uniform_filter(
                    votes.astype(np.float32),
                    size=rihog_window,
                    mode="reflect",
                )
                histogram[:, ibin] = pooled[yy, xx]

            rihog = np.abs(np.fft.rfft(histogram, axis=1)).astype(np.float32)
            rihog /= np.maximum(
                np.linalg.norm(rihog, axis=1, keepdims=True),
                1e-6,
            )
            blocks.append(rihog)

        elif name == "hessian":
            response = []

            for hessian_sigma in hessian_sigmas:
                scale2 = hessian_sigma**2
                hxx = scale2 * gaussian_filter(
                    raw_gray,
                    sigma=hessian_sigma,
                    order=(0, 2),
                    mode="reflect",
                )[yy, xx]
                hyy = scale2 * gaussian_filter(
                    raw_gray,
                    sigma=hessian_sigma,
                    order=(2, 0),
                    mode="reflect",
                )[yy, xx]
                hxy = scale2 * gaussian_filter(
                    raw_gray,
                    sigma=hessian_sigma,
                    order=(1, 1),
                    mode="reflect",
                )[yy, xx]

                trace = hxx + hyy
                delta = np.sqrt(
                    np.maximum((hxx - hyy) ** 2 + 4 * hxy**2, 0)
                )
                eigen1 = np.abs(0.5 * (trace - delta))
                eigen2 = np.abs(0.5 * (trace + delta))
                response.extend([
                    np.minimum(eigen1, eigen2),
                    np.maximum(eigen1, eigen2),
                ])

            blocks.append(np.column_stack(response))

    if normal and len(xx):
        block_weight = 1.0 / np.sqrt(len(blocks))

        for ib, (name, block) in enumerate(zip(features, blocks)):
            if name in {
                "appearance",
                "structure",
                "gradient",
                "gabor",
                "hessian",
            }:
                center = np.median(block, axis=0)
                scale = 1.4826 * np.median(np.abs(block - center), axis=0)
                scale = np.where(
                    np.isfinite(scale) & (scale > 1e-6),
                    scale,
                    1.0,
                )
                block = (block - center) / (scale * np.sqrt(block.shape[1]))

            elif name in {"patch", "mind", "rihog"}:
                block = block / np.maximum(
                    np.linalg.norm(block, axis=1, keepdims=True),
                    1e-6,
                )

            elif name == "lbp":
                block = np.sqrt(
                    block
                    / np.maximum(block.sum(axis=1, keepdims=True), 1e-6)
                )
                block /= np.maximum(
                    np.linalg.norm(block, axis=1, keepdims=True),
                    1e-6,
                )

            else:  # Bounded NGF tensor.
                block = block / np.sqrt(block.shape[1])

            blocks[ib] = block_weight * block

    points = np.column_stack([xx, yy]).astype(np.float32)
    descriptors = np.concatenate(blocks, axis=1).astype(np.float32)

    return points, descriptors


def normalize_feature_blocks(
    FX,
    FY,
    features=("appearance", "gradient"),
    weights=None,
    patch_size=7,
    lbp_points=8,
    gabor_frequencies=(0.1, 0.2),
    pre_normalized=True,
    max_samples=100_000,
    eps=1e-6,
    random_state=0,
    mind_angles=8,
    rihog_bins=8,
    hessian_sigmas=(1.0, 2.0, 4.0),
):
    """
    Jointly normalize and weight heterogeneous FX/FY feature blocks.

    The column layout is reconstructed from ``features`` and the descriptor
    parameters used by :func:`image_to_points`. The appearance dimension is
    inferred from the remaining columns, so grayscale, Lab/HED, and arbitrary
    raw-channel inputs are supported without a separate channel argument.

    Parameters
    ----------
    FX : (NX, K) array
        Descriptor matrix for the fixed point set. It must be nonempty, finite,
        and have the same columns and block order as ``FY``.
    FY : (NY, K) array
        Descriptor matrix for the moving point set.
    features : str or sequence of str, default=('appearance', 'gradient')
        Dense feature blocks and their column order. Use exactly the same value
        that was passed to :func:`image_to_points`. SIFT is not supported here.
    weights : mapping, sequence, or None, default=None
        Non-negative weights assigned to whole blocks in squared Euclidean
        distance. A mapping is indexed by feature name; a sequence follows
        ``features`` order. Values are normalized to sum to one. If ``None``,
        every block receives equal weight. The descriptors are multiplied by
        ``sqrt(weight)``, so the squared feature distance uses the requested
        weight rather than its square.
    patch_size : int, default=7
        Patch side length used during extraction. It defines a block of
        ``patch_size**2`` columns and must match both FX and FY.
    lbp_points : int, default=8
        Number of LBP neighbors used during extraction. It defines a block of
        ``lbp_points + 2`` columns and must match both FX and FY.
    gabor_frequencies : sequence of float, default=(0.1, 0.2)
        Frequencies used during extraction. Their count defines the Gabor block
        dimension and must match both FX and FY.
    mind_angles : int, default=8
        Number of MIND ring directions used during extraction. It defines the
        MIND block dimension and must match both FX and FY.
    rihog_bins : int, default=8
        Number of orientation bins used during extraction. The RIHOG block has
        ``rihog_bins // 2 + 1`` columns and must match both inputs.
    hessian_sigmas : sequence of float, default=(1.0, 2.0, 4.0)
        Hessian scales used during extraction. Their count defines a block of
        ``2 * len(hessian_sigmas)`` columns and must match both inputs.
    pre_normalized : bool, default=True
        Set true when FX and FY came from ``image_to_points(normal=True)``.
        Intrinsic patch, NGF, MIND, RIHOG, and Hellinger-LBP geometry is then
        preserved while final block weights are refreshed. Set false for raw
        descriptors from ``normal=False``; LBP then receives its Hellinger
        transform here. Continuous blocks receive a shared robust FX/FY scaling
        in either case.
    max_samples : int, default=100000
        Maximum number of rows drawn from each matrix to estimate shared median
        and MAD values. Balanced sampling prevents the larger point set from
        dominating the scale estimate.
    eps : float, default=1e-6
        Positive threshold for constant robust scales, histogram sums, and row
        norms. Degenerate continuous scales are replaced by one.
    random_state : int, array-like, None, or NumPy SeedSequence, default=0
        Seed passed to ``numpy.random.default_rng`` for reproducible balanced
        row sampling.

    Returns
    -------
    FX_normalized : (NX, K) float32 array
        Jointly normalized and block-weighted FX descriptors.
    FY_normalized : (NY, K) float32 array
        FY descriptors transformed with the same continuous-block statistics.
    center : dict
        Shared per-column medians for appearance, structure, gradient, Gabor,
        and Hessian blocks. Intrinsically normalized block entries are ``None``.
    scale : dict
        Shared MAD scales for continuous blocks. Intrinsically normalized block
        entries are ``None``.

    Notes
    -----
    For a continuous block of dimension ``d_b``, the returned representation is
    ``sqrt(w_b / d_b) * (F_b - center_b) / scale_b``. Consequently, its
    squared-distance contribution is the weighted mean per-column distance,
    rather than a sum that automatically favors high-dimensional blocks.
    """
    FX = np.asarray(FX, dtype=np.float32)
    FY = np.asarray(FY, dtype=np.float32)
    features = (features,) if isinstance(features, str) else tuple(features)

    if FX.ndim != 2 or FY.ndim != 2:
        raise ValueError("FX and FY must be two-dimensional arrays")
    if FX.shape[1] != FY.shape[1]:
        raise ValueError("FX and FY must have the same feature dimension")
    if len(FX) == 0 or len(FY) == 0:
        raise ValueError("FX and FY cannot be empty")
    if not np.all(np.isfinite(FX)) or not np.all(np.isfinite(FY)):
        raise ValueError("FX and FY must contain only finite values")
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be nonempty and contain no duplicates")
    if "sift" in features:
        raise ValueError("SIFT must be normalized separately")

    valid_features = {
        "appearance",
        "structure",
        "gradient",
        "ngf",
        "patch",
        "lbp",
        "gabor",
        "mind",
        "rihog",
        "hessian",
    }
    unknown = set(features) - valid_features
    if unknown:
        raise ValueError(f"unknown feature: {sorted(unknown)!r}")
    if (
        not isinstance(patch_size, (int, np.integer))
        or patch_size < 1
        or patch_size % 2 != 1
    ):
        raise ValueError("patch_size must be a positive odd integer")
    if not isinstance(lbp_points, (int, np.integer)) or lbp_points < 1:
        raise ValueError("lbp_points must be a positive integer")
    if not isinstance(pre_normalized, (bool, np.bool_)):
        raise TypeError("pre_normalized must be a boolean")
    if not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    gabor_frequencies = tuple(gabor_frequencies)
    if "gabor" in features and len(gabor_frequencies) == 0:
        raise ValueError("gabor_frequencies cannot be empty")
    if not isinstance(mind_angles, (int, np.integer)) or mind_angles < 4:
        raise ValueError("mind_angles must be an integer >= 4")
    if not isinstance(rihog_bins, (int, np.integer)) or rihog_bins < 4:
        raise ValueError("rihog_bins must be an integer >= 4")

    hessian_sigmas = tuple(hessian_sigmas)
    if "hessian" in features and (
        len(hessian_sigmas) == 0
        or any(
            not np.isfinite(value) or value <= 0
            for value in hessian_sigmas
        )
    ):
        raise ValueError("hessian_sigmas must contain positive finite values")

    dims = {
        "structure": 1,
        "gradient": 1,
        "ngf": 3,
        "patch": patch_size**2,
        "lbp": lbp_points + 2,
        "gabor": len(gabor_frequencies),
        "mind": mind_angles,
        "rihog": rihog_bins // 2 + 1,
        "hessian": 2 * len(hessian_sigmas),
    }
    known_dim = sum(dims[name] for name in features if name != "appearance")

    if "appearance" in features:
        dims["appearance"] = FX.shape[1] - known_dim
        if dims["appearance"] < 1:
            raise ValueError("feature dimension is inconsistent with parameters")
    elif known_dim != FX.shape[1]:
        raise ValueError("feature dimension is inconsistent with parameters")

    if weights is None:
        block_weights = np.ones(len(features), dtype=float)
    elif isinstance(weights, dict):
        missing = set(features) - set(weights)
        if missing:
            raise ValueError(f"missing feature weights: {sorted(missing)!r}")
        block_weights = np.asarray([weights[name] for name in features], dtype=float)
    else:
        block_weights = np.asarray(weights, dtype=float)
        if block_weights.shape != (len(features),):
            raise ValueError("weights must have one value per feature block")

    if not np.all(np.isfinite(block_weights)) or np.any(block_weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(block_weights > 0):
        raise ValueError("at least one feature weight must be positive")
    block_weights /= block_weights.sum()

    n = min(len(FX), len(FY), max_samples)
    rng = np.random.default_rng(random_state)
    ix = rng.choice(len(FX), n, replace=False)
    iy = rng.choice(len(FY), n, replace=False)

    Xout, Yout = [], []
    center, scale = {}, {}
    start = 0

    for name, weight in zip(features, block_weights):
        stop = start + dims[name]
        Xb = FX[:, start:stop].copy()
        Yb = FY[:, start:stop].copy()

        if name in {
            "appearance",
            "structure",
            "gradient",
            "gabor",
            "hessian",
        }:
            Z = np.concatenate([Xb[ix], Yb[iy]], axis=0)
            cb = np.median(Z, axis=0)
            sb = 1.4826 * np.median(np.abs(Z - cb), axis=0)
            sb = np.where(np.isfinite(sb) & (sb > eps), sb, 1.0)
            Xb = (Xb - cb) / (sb * np.sqrt(dims[name]))
            Yb = (Yb - cb) / (sb * np.sqrt(dims[name]))
            center[name], scale[name] = cb, sb

        elif name in {"patch", "mind", "rihog"}:
            Xb /= np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), eps)
            Yb /= np.maximum(np.linalg.norm(Yb, axis=1, keepdims=True), eps)
            center[name] = scale[name] = None

        elif name == "lbp":
            if not pre_normalized:
                if np.any(Xb < 0) or np.any(Yb < 0):
                    raise ValueError("raw LBP histogram values must be non-negative")
                Xb = np.sqrt(Xb / np.maximum(Xb.sum(axis=1, keepdims=True), eps))
                Yb = np.sqrt(Yb / np.maximum(Yb.sum(axis=1, keepdims=True), eps))
            Xb /= np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), eps)
            Yb /= np.maximum(np.linalg.norm(Yb, axis=1, keepdims=True), eps)
            center[name] = scale[name] = None

        else:  # Bounded NGF tensor.
            if pre_normalized:
                Xb *= np.sqrt(len(features))
                Yb *= np.sqrt(len(features))
            else:
                Xb /= np.sqrt(dims[name])
                Yb /= np.sqrt(dims[name])
            center[name] = scale[name] = None

        block_scale = np.sqrt(weight)
        Xout.append(block_scale * Xb)
        Yout.append(block_scale * Yb)
        start = stop

    return (
        np.concatenate(Xout, axis=1).astype(np.float32),
        np.concatenate(Yout, axis=1).astype(np.float32),
        center,
        scale,
    )

def shared_robust_scale(
    FX,
    FY,
    max_samples=100_000,
    eps=1e-6,
    random_state=0,
):
    """
    Apply a shared robust per-feature scaling to two descriptor matrices.

    The function estimates one center and scale from equally sized random
    samples of ``FX`` and ``FY`` and applies them to both matrices. Sharing the
    transformation keeps corresponding feature dimensions comparable without
    allowing the larger point set to dominate the estimate.

    Parameters
    ----------
    FX : (NX, K) array
        First descriptor matrix. Rows are points and columns are feature
        dimensions. It must be nonempty and contain only finite values.
    FY : (NY, K) array
        Second descriptor matrix with the same ``K`` columns as ``FX``. It must
        be nonempty and contain only finite values.
    max_samples : int, default=100000
        Maximum number of rows sampled from each matrix. The actual number per
        matrix is ``min(NX, NY, max_samples)``. Increase it for a more stable
        estimate on heterogeneous million-point data; decrease it to reduce
        scaling time and temporary memory. Values from 50,000 to 200,000 are
        usually sufficient for large dense point sets.
    eps : float, default=1e-6
        Positive threshold used to identify an effectively constant feature.
        If its MAD-based scale is not finite or is no greater than ``eps``, the
        scale is set to 1 to avoid division by zero. This is a numerical
        threshold, not a regularization strength; the default is normally kept.
    random_state : int, array-like, None, or NumPy SeedSequence, default=0
        Seed passed to ``numpy.random.default_rng`` for row subsampling. Keep a
        fixed integer for reproducible scaling, change it to draw another
        sample, or use ``None`` for nondeterministic sampling.

    Returns
    -------
    FX_scaled : (NX, K) array
        ``FX`` transformed as ``(FX - center) / scale``.
    FY_scaled : (NY, K) array
        ``FY`` transformed with the same ``center`` and ``scale``.
    center : (K,) array
        Per-feature median of the combined, balanced sample.
    scale : (K,) array
        Robust scale ``1.4826 * median(abs(Z - center))``. Degenerate columns
        have scale 1.

    Notes
    -----
    This function is intended for continuous feature blocks whose columns need
    a common scale, especially appearance and gradient descriptors extracted
    from two images. Avoid applying it blindly to L2-normalized patch or SIFT
    descriptors because per-column rescaling changes their original metric.
    """
    FX = np.asarray(FX, dtype=np.float32)
    FY = np.asarray(FY, dtype=np.float32)

    if FX.ndim != 2 or FY.ndim != 2:
        raise ValueError("FX and FY must be two-dimensional arrays")
    if FX.shape[1] != FY.shape[1]:
        raise ValueError("FX and FY must have the same feature dimension")
    if len(FX) == 0 or len(FY) == 0:
        raise ValueError("FX and FY cannot be empty")
    if not np.all(np.isfinite(FX)) or not np.all(np.isfinite(FY)):
        raise ValueError("FX and FY must contain only finite values")
    if not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    n = min(len(FX), len(FY), max_samples)
    rng = np.random.default_rng(random_state)

    Z = np.concatenate([
        FX[rng.choice(len(FX), n, replace=False)],
        FY[rng.choice(len(FY), n, replace=False)],
    ])

    center = np.median(Z, axis=0)
    scale = 1.4826 * np.median(np.abs(Z - center), axis=0)
    scale = np.where(
        np.isfinite(scale) & (scale > eps),
        scale,
        1.0,
    )

    return (
        (FX - center) / scale,
        (FY - center) / scale,
        center,
        scale,
    )


def rawgrid2pts(images, scale=1, interp_weight=True, points=None, weight=None, method = 'linear', **kargs ):
    shp = images.shape 
    ndim = images.ndim 

    if isinstance(scale, (float, int)):
        scale = [scale] * ndim
    else:
        scale = list(scale)[:ndim]  

    coords = []
    for d in range(ndim):
        step = scale[d]
        # axis_coords = np.arange(0, shp[d], step).clip(0, shp[d] - 1).astype(np.float32)
        axis_coords = np.linspace(0, shp[d]-1, int(shp[d]/step)).astype(np.float32)
        coords.append(axis_coords)

    mesh = np.meshgrid(*coords, indexing='ij')
    pointr = np.stack([m.ravel() for m in mesh], axis=-1)
    
    if interp_weight:
        import scipy as sci
        values = sci.interpolate.griddata(points, weight, pointr,  rescale=False,  method=method, **kargs)
    else:
        indices = np.round(pointr).astype(np.int64)
        values = images[tuple(indices.T)]
    
    return pointr, values

def scalergid2pts(images, scale=1, method = 'linear', interp_weight=False,  **kargs):
    ndim = images.ndim 

    if isinstance(scale, (float, int)):
        scale = [scale] * ndim
    else:
        scale = list(scale)[:ndim]  
    
    if interp_weight and (scale != [1] * ndim):
        pointr, valuesr =  rawgrid2pts(images, scale=1, interp_weight=False)
        pointr, valuesr =  rawgrid2pts(images, scale=scale, interp_weight=True, points=pointr, weight=valuesr, method = method, **kargs)
    else:
        pointr, valuesr =  rawgrid2pts(images, scale=scale, interp_weight=False)

    return pointr, valuesr

def grayimg2pts(images, scale=1, inverse_weight=False, sample=None, vmin=0., seed=0,
                  method = 'linear', interp_weight=False,  **kargs):
    points, value = scalergid2pts(images, scale=scale,  method = method, interp_weight=interp_weight,  **kargs)
    if inverse_weight:
        weigth = value.max()-value
    else:
        weigth = value

    if not vmin is None:
        kidx = weigth>vmin
        points = points[kidx]
        weigth = weigth[kidx]
    if not sample is None:
        if sample <=1:
            size = int(sample * points.shape[0])
        else:
            if sample <10:
                print('Warning: sample size is smaller than 10')
            size = int(sample)
        size = min(size, points.shape[0])
        np.random.seed(seed)
        kidx = np.random.choice(points.shape[0], size=size,  replace=False, p=weigth/weigth.sum())
    else:
        kidx = np.ones_like(weigth, dtype=bool)
    return points, weigth, kidx


def _bspline_matrix(z, n_basis, degree):
    """Construct an open-uniform B-spline design matrix."""
    n_inner = n_basis - degree - 1

    inner = (
        np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        if n_inner > 0
        else np.empty(0)
    )

    knots = np.r_[
        np.zeros(degree + 1),
        inner,
        np.ones(degree + 1),
    ]

    return BSpline(
        knots,
        np.eye(n_basis),
        degree,
        extrapolate=False,
    )(np.asarray(z, dtype=float))

def interpolate_slices(
    points,
    z_obs,
    z_new,
    method="pspline",
    grid_scale=0.75,
    radius_scale=1.0,
    margin=4,
    closing_iterations=1,
    fill_holes=True,
    smooth=0.1,
    delta=1e-8,
    n_basis=None,
    degree=3,
    z_bandwidth=None,
    reliability=None,
    chunk_size=50_000,
    n_points="density",
    jitter=True,
    sampling=True,
    random_state=0,
    return_info=False,
):
    """
    Reconstruct intermediate point-cloud slices through SDF interpolation.

    Shape methods
    -------------
    linear   : local piecewise-linear SDF interpolation.
    pchip    : local shape-preserving cubic Hermite interpolation.
    cubic    : natural cubic spline; smooth but may overshoot.
    gaussian : Gaussian kernel regression along z.
    pspline  : penalized B-spline with optional reliability weights.

    Sampling
    --------
    sampling=True
        Sample the number of points specified by n_points.

    sampling=False
        Return all foreground grid points. In this case, n_points is ignored.

    jitter=True
        Randomly perturb each selected point inside its grid cell.

    jitter=False
        Keep points at the grid-cell centers.
    Returns
    -------
    sampled : list of (Ni, 2) arrays
        Interpolated xy point clouds corresponding to z_new.
    info : dict, optional
        Returned as ``(sampled, info)`` only when return_info=True. Contains
        the shared x/y axes, observed/predicted masks and SDFs, sorted z_obs,
        z_new, and the original input-slice sorting order. ``cell_indices``
        stores one (Ni, 2) array of [iy, ix] per target slice;
        ``jitter_offsets`` stores the matching [dx, dy] physical offsets.
        ``slice_index`` and ``slice_offsets`` locate sampled points in a
        concatenation of sampled. Geometry metadata does not encode material
        correspondence or feature values. The default return is unchanged.
    """
    methods = {"linear", "pchip", "cubic", "gaussian", "pspline"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}.")

    z_obs = np.asarray(z_obs, dtype=float).reshape(-1)
    z_new = np.atleast_1d(np.asarray(z_new, dtype=float)).reshape(-1)

    if len(z_obs) < 2 or len(points) != len(z_obs):
        raise ValueError("At least two point slices matching z_obs are required.")

    if len(z_new) == 0 and not return_info:
        return []
    if not np.all(np.isfinite(z_obs)) or not np.all(np.isfinite(z_new)):
        raise ValueError("z_obs and z_new must contain finite positions.")

    order = np.argsort(z_obs)
    z_obs = z_obs[order]
    points = [np.asarray(points[i], dtype=float) for i in order]
    if any(p.ndim != 2 or p.shape[1] < 2 or len(p) == 0 for p in points):
        raise ValueError("Each observed slice must be a nonempty (N, 2) array.")
    points = [p[:, :2] for p in points]
    if any(not np.all(np.isfinite(p)) for p in points):
        raise ValueError("Observed xy coordinates must be finite.")

    if np.any(np.diff(z_obs) <= 0):
        raise ValueError("z_obs must contain unique positions.")

    if len(z_new) and (z_new.min() < z_obs[0] or z_new.max() > z_obs[-1]):
        raise ValueError("z_new must lie inside the observed z range.")

    # Estimate the in-plane sampling scale from nearest-neighbor distances.
    nn = [cKDTree(p).query(p, k=2)[0][:, 1] for p in points if len(p) > 1]
    d = np.concatenate(nn) if nn else np.empty(0)
    d = d[np.isfinite(d) & (d > 0)]
    if not len(d):
        raise ValueError("No positive within-slice XY spacing can be estimated.")
    dxy = float(np.median(d))
    
    if np.isscalar(grid_scale):
        dx = dy = float(grid_scale) * dxy
    else:
        dx, dy = np.array(grid_scale) * dxy
    mask_radius = dxy * radius_scale
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("grid_scale must give positive finite grid spacing.")
    if not np.isfinite(mask_radius) or mask_radius < 0:
        raise ValueError("radius_scale must be finite and nonnegative.")

    # One common physical xy grid is required before SDF interpolation.
    xy = np.vstack(points)
    xmin, ymin = xy.min(0)
    xmax, ymax = xy.max(0)
    x = np.arange(
        np.floor(xmin / dx) * dx - margin * dx,
        np.ceil(xmax / dx) * dx + (margin + 0.5) * dx,
        dx,
    )
    y = np.arange(
        np.floor(ymin / dy) * dy - margin * dy,
        np.ceil(ymax / dy) * dy + (margin + 0.5) * dy,
        dy,
    )

    # B_r = {(du, dv): du^2 + dv^2 <= r^2}.
    rx = max(1, int(np.ceil(mask_radius / dx)))
    ry = max(1, int(np.ceil(mask_radius / dy)))
    xx = np.arange(-rx, rx + 1) * dx
    yy = np.arange(-ry, ry + 1) * dy
    structure = yy[:, None] ** 2 + xx[None, :] ** 2 <= mask_radius**2

    masks_obs = []
    for p in points:
        ix = np.rint((p[:, 0] - x[0]) / dx).astype(int)
        iy = np.rint((p[:, 1] - y[0]) / dy).astype(int)
        valid = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        mask = np.zeros((len(y), len(x)), dtype=bool)
        mask[iy[valid], ix[valid]] = True

        # Eq. (1): M1 = M0 (+) B_r;
        #           M  = close(M1, B_r) = (M1 (+) B_r) (-) B_r.
        mask = binary_dilation(mask, structure=structure)
        if closing_iterations > 0:
            mask = binary_closing(
                mask,
                structure=structure,
                iterations=int(closing_iterations),
            )
        if fill_holes:
            mask = binary_fill_holes(mask)
        masks_obs.append(mask)
    masks_obs = np.stack(masks_obs)

    # Eq. (2): phi = d(x, outside) - d(x, inside); phi > 0 inside.
    sdf_obs = np.stack(
        [ distance_transform_edt(mask, sampling=(dy, dx)) - distance_transform_edt(~mask, sampling=(dy, dx))
            for mask in masks_obs
        ]
    )
    shape = sdf_obs.shape
    Y = sdf_obs.reshape(len(z_obs), -1)

    if method == "linear":
        # Eq. (3): phi(z) = (1-t)phi_i + t*phi_(i+1).
        sdf_new = interp1d(
            z_obs,
            Y,
            axis=0,
            kind="linear",
            assume_sorted=True,
        )(z_new)

    elif method == "pchip":
        # Piecewise cubic Hermite interpolation limits local overshoot.
        sdf_new = PchipInterpolator(z_obs, Y, axis=0)(z_new)

    elif method == "cubic":
        # Natural cubic spline: phi''(z_min) = phi''(z_max) = 0.
        sdf_new = CubicSpline(
            z_obs,
            Y,
            axis=0,
            bc_type="natural",
        )(z_new)

    elif method == "gaussian":
        # Eq. (4): phi(z) = sum_i K_h(z-z_i)phi_i / sum_i K_h(z-z_i).
        if z_bandwidth is None:
            z_bandwidth = float(np.median(np.diff(z_obs)))
        W = np.exp(
            -0.5 * ((z_new[:, None] - z_obs[None, :]) / z_bandwidth) ** 2
        )
        W /= W.sum(axis=1, keepdims=True)
        sdf_new = W @ Y

    else:
        # Eq. (5): min_C ||W^(1/2)(Y-BC)||_F^2
        #                   + lambda||D2*C||_F^2 + ridge||C||_F^2.
        degree = min(int(degree), len(z_obs) - 1)
        if n_basis is None:
            n_basis = min(12, max(degree + 1, len(z_obs)))
        n_basis = max(degree + 1, int(n_basis))
        z0, z1 = z_obs[[0, -1]]
        B = _bspline_matrix((z_obs - z0) / (z1 - z0), n_basis, degree)
        Bq = _bspline_matrix((z_new - z0) / (z1 - z0), n_basis, degree)
        D2 = np.diff(np.eye(n_basis), n=2, axis=0)
        penalty = smooth * (D2.T @ D2) + delta * np.eye(n_basis)

        if reliability is None:
            w = np.ones(len(z_obs))
            coef = np.linalg.solve(
                B.T @ (w[:, None] * B) + penalty,
                B.T @ (w[:, None] * Y),
            )
        else:
            R = np.asarray(reliability, dtype=float)[order]
            if R.ndim == 1:
                coef = np.linalg.solve(
                    B.T @ (R[:, None] * B) + penalty,
                    B.T @ (R[:, None] * Y),
                )
            else:
                R = np.broadcast_to(R, shape).reshape(len(z_obs), -1)
                coef = np.empty((n_basis, Y.shape[1]))
                for start in range(0, Y.shape[1], chunk_size):
                    stop = min(start + chunk_size, Y.shape[1])
                    Rc = R[:, start:stop]
                    Yc = Y[:, start:stop]
                    A = np.einsum("si,sv,sj->vij", B, Rc, B)
                    A += penalty[None, :, :]
                    rhs = np.einsum("si,sv,sv->vi", B, Rc, Yc)
                    coef[:, start:stop] = np.linalg.solve(
                        A, rhs[..., None]
                    )[..., 0].T
        sdf_new = Bq @ coef

    sdf_new = sdf_new.reshape(len(z_new), len(y), len(x))
    masks_new = sdf_new > 0.0

    # Preserve the median observed point density unless explicit counts are set.
    area_obs = masks_obs.sum((1, 2)) * dx * dy
    area_new = masks_new.sum((1, 2)) * dx * dy
    if isinstance(n_points, str):
        density = np.median(
            np.array([len(p) for p in points])
            / np.maximum(area_obs, 1e-12)
        )
        counts = np.where(
            area_new > 0,
            np.maximum(1, np.rint(density * area_new).astype(int)),
            0,
        )
    elif n_points is None:
        counts = np.array([mask.sum() for mask in masks_new], dtype=int)
    elif np.isscalar(n_points):
        counts = np.full(len(z_new), int(n_points))
    else:
        counts = np.asarray(n_points, dtype=int)
        if len(counts) != len(z_new):
            raise ValueError("n_points must contain one count per z_new.")

    rng = np.random.default_rng(random_state)
    sampled, cell_indices, jitter_offsets = [], [], []
    for i, (zi, mask, count) in enumerate(zip(z_new, masks_new, counts)):
        iy, ix = np.where(mask)

        if len(ix) == 0:
            if count:
                raise ValueError(
                    f"The predicted mask at z={zi:g} is empty."
                )

            sampled.append(np.empty((0, 2)))
            if return_info:
                cell_indices.append(np.empty((0, 2), dtype=int))
                jitter_offsets.append(np.empty((0, 2), dtype=float))
            continue

        if sampling:
            choose = rng.choice( len(ix), size=int(count), replace=count > len(ix), )
            ix = ix[choose]
            iy = iy[choose]
        else:
            count = len(ix)

        px = x[ix].copy()
        py = y[iy].copy()

        if jitter:
            px += rng.uniform( -0.5 * dx, 0.5 * dx, int(count),)
            py += rng.uniform( -0.5 * dy, 0.5 * dy, int(count),)

        sampled.append(np.c_[px, py])
        if return_info:
            cell_indices.append(np.c_[iy, ix])
            jitter_offsets.append(np.c_[px - x[ix], py - y[iy]])

    if return_info:
        lengths = np.array([len(p) for p in sampled], dtype=int)
        info = dict(
            x=x, y=y, dx=dx, dy=dy, dxy=dxy,
            z_obs=z_obs, z_new=z_new, order=order,
            masks_obs=masks_obs, sdf_obs=sdf_obs,
            masks_new=masks_new, sdf_new=sdf_new,
            cell_indices=cell_indices, jitter_offsets=jitter_offsets,
            slice_index=np.repeat(np.arange(len(z_new)), lengths),
            slice_offsets=np.r_[0, np.cumsum(lengths)],
            method=method,
            config=dict(
                grid_scale=grid_scale, radius_scale=radius_scale, margin=margin,
                closing_iterations=closing_iterations, fill_holes=fill_holes,
                smooth=smooth, delta=delta, n_basis=n_basis, degree=degree,
                z_bandwidth=z_bandwidth, reliability=reliability,
                n_points=n_points, sampling=sampling, jitter=jitter,
                random_state=random_state,
            ),
        )
        result = sampled, info
    else:
        result = sampled
    return result


def pcd_grid(position, spacing = np.array([30., 30., 30.])):
    import numpy as np
    from scipy import spatial
    
    ndim = position.shape[1]
    spacing = np.array(spacing)[:ndim]

    xyzs = []
    for i in range(ndim):
        imin = position[:,i].min()
        imax = position[:,i].max()
        ispc = spacing[i]
        ix=np.arange(imin, imax+ispc,ispc).clip(imin, imax)
        xyzs.append(ix)
    xyzs2=np.meshgrid(*xyzs,indexing='ij')
    all_grid=np.array([ix.flatten() for ix in xyzs2 ]).T

    point_tree = spatial.cKDTree(all_grid)
    dist, index = point_tree.query(position)
    return all_grid, dist, index

def interp_pcd_grid(Points, bins = None, space = None, include_edge=True, return_bins=True):
    ndim = Points.shape[1]
    space = list_iter(space)
    bins = list_iter(bins)
    grids = []
    grids_center = []
    for i in range(ndim):
        irange = Points[:, i]
        ispace = space[i]
        ibin = bins[i]
        istart = np.min(irange)
        iend = np.max(irange)
        if not ibin is None:
            if include_edge:
                i_cent = np.linspace(istart, iend, ibin, endpoint=True)
            else:
                i_cent = np.linspace(istart, iend, ibin+1, endpoint=True)
                i_cent = (i_cent[1:]+i_cent[:-1])/2, 

        elif not ispace is None:
            if include_edge:
                i_cent = np.arange(istart, iend, ispace).clip(istart, iend)
            else:
                i_cent = np.arange(istart+ispace/2, iend+ispace/2, ispace).clip(istart, iend)
            i_cent = np.unique(i_cent)

        elif ispace is None:
            i_cent = np.unique(irange)
            ispace =  np.diff(i_cent).mean()

        i_bins = np.r_[ i_cent[0]-ispace,
                        (i_cent[1:]+i_cent[:-1])/2, 
                        i_cent[-1] + ispace ]
        assert len(i_bins) - len(i_cent) == 1
        grids.append(i_bins)
        grids_center.append(i_cent)

    cent_cpos = localize_points(Points, grids, grids_center=grids_center)
    # cent_idxstr = np.array(list(map(lambda x:'-'.join(x),  cent_index.astype(str))))

    # cent_cpos = pd.DataFrame(cent_cpos, columns=list('xyz'), index=cent_idxstr)
    # cent_cpos[['x_pos', 'y_pos', 'z_pos']] = Points
    # cent_cpos['xyz_inds'] = cent_idxstr
    return cent_cpos

def interp_shape_grid(ranges, bins = None, space = 150, include_edge=True, return_grid=True):
    ndim = len(ranges)
    space = list_iter(space)
    bins = list_iter(bins)
    grids = []
    for i in range(ndim):
        irange = ranges[i]
        ispace = space[i]
        ibin = bins[i]
        istart = irange[0]
        iend = irange[1]
        if not ibin is None:
            if include_edge:
                i_cent = np.linspace(istart, iend, ibin, endpoint=True)
            else:
                i_cent = np.linspace(istart, iend, ibin+1, endpoint=True)
                i_cent = (i_cent[1:]+i_cent[:-1])/2, 

        elif not ispace is None:
            if include_edge:
                i_cent = np.arange(istart, iend, ispace).clip(istart, iend)
            else:
                i_cent = np.arange(istart+ispace/2, iend+ispace/2, ispace).clip(istart, iend)
            i_cent = np.unique(i_cent)

        grids.append(i_cent)

    mgrid = np.meshgrid(*grids)
    if return_grid:
        return mgrid
    else:
        Points = np.vstack([ igd.ravel() for igd in mgrid ]).T
        return Points

def localize_points(Points, grids, grids_center=None, extend_outside=False):
    ndim = Points.shape[1]
    assert len(grids) == ndim
    if not grids_center is None:
        assert len(grids_center) == ndim

    cent_index = []
    cent_cpos  = [] 
    for i in range(ndim):
        irange = Points[:, i]
        i_bins = grids[i]

        if extend_outside:
            if np.min(irange) < np.min(i_bins):
                i_bins =  np.r_[np.min(irange), i_bins]
            if  np.max(irange) > np.max(i_bins):
                i_bins =  np.r_[i_bins,  np.max(irange)]
        else:
            if (np.max(irange) > np.max(i_bins)) or  (np.min(irange) < np.min(i_bins)):
                raise('points is range out of grids. Please set extend_outside=True or reset grids range.')

        if not grids_center is None:
            i_cent = grids_center[i]
        else:
            i_cent = (i_bins[1:]+i_bins[:-1])/2

        i_inds = np.digitize(irange, i_bins, right=False)
        i_cpos = i_cent[i_inds-1]
        cent_index.append(i_inds)
        cent_cpos.append(i_cpos)
    cent_index = np.c_[cent_index].T
    cent_cpos = np.c_[cent_cpos].T
    return np.c_[cent_cpos, cent_index]

def interp_pcd_zslice(point3d, axis=2, n_interp=3, 
                      interp_frac = None,
                      S=0.5, R =1, thred=0,
                      interp_self = False,
                      down_sample =True, sample_frac=None, sample_scale=1, sample_method = 'open3d',
                      use_bw=True, seed=200504, **kargs):
    zspace = np.unique(point3d[:,axis])
    axes = [0,1,2]
    axes.pop(axis)

    zspace_int = []
    interp_pos = []
    #for i in range(len(zspace)-1):
    n_epochs = len(zspace)-1
    for i in tqdm(range(n_epochs), colour='red'):
        itick = np.linspace(zspace[i], zspace[i+1], n_interp+2).astype(np.float64)
        ispace = np.linspace(0,1, n_interp+2).astype(np.float64) if interp_frac is None else  np.array([0, *interp_frac, 1])
        assert len(itick) == len(ispace), "tick and space must have the same length"

        if not interp_self:
            itick = itick[1:-1]
            ispace = ispace[1:-1]

        pointa = point3d[point3d[:,axis]==zspace[i]][:,axes].copy()
        pointb = point3d[point3d[:,axis]==zspace[i+1]][:,axes].copy()

        imageo, pointxy = interp_point(pointa.copy(), pointb.copy(), 
                                precision=ispace, S=S,R=R,
                                use_bw=use_bw, thred=thred, **kargs)
        # for isp in ispace:
        #     if isp> 0.5:
        #         imgbp, ipointxy = interp_point(pointb.copy(), pointa.copy(),
        #                                    precision=(1-isp), S=S,R=R,
        #                                         use_bw=use_bw, thred=thred, **kargs)
        #     else:
        #         imgbp, ipointxy = interp_point(pointa.copy(), pointb.copy(), 
        #                                    precision=isp, S=S,R=R,
        #                                         use_bw=use_bw, thred=thred, **kargs)
        #     pointxy.extend(ipointxy)

        pointxyz = []
        size_a, size_b = pointa.shape[0], pointb.shape[0]
        for i, (isp, itc) in enumerate(zip(ispace, itick)):
            ipointxyz = np.c_[pointxy[i],  np.full(pointxy[i].shape[0], itc)]
            size_c = sample_frac or (size_a* (1-isp)  + size_b*isp) * sample_scale
            if (down_sample and size_c<ipointxyz.shape[0]):
                ipointxyz = down_sample_point(ipointxyz, samples=size_c,
                                              k_points=None, method =sample_method,
                                              seed=seed)
            
            ipointxyz = np.c_[ipointxyz, np.full(ipointxyz.shape[0], isp)]
            pointxyz.append(ipointxyz)
        pointxyz = np.vstack(pointxyz)

        interp_pos.append(pointxyz)
        zspace_int.extend(itick)

    interp_pos = np.vstack(interp_pos)
    # if not interp_self:
    #     interp_pos = np.concatenate([point3d, interp_pos], axis=0)
    return interp_pos

def interp_point(pointA, pointB, precision=0, S=1, R =1,  HW=None, use_bw=True, thred=0, order=2, 
               csize = 5, dsize=5, method='circle',**kargs):
    S,R  = float(S), float(R)
    if HW is None:
        HW = np.c_[pointA.max(0), pointB.max(0)].max(1)[:2]
    else:
        HW = np.array(HW)
    HW = np.int64( np.ceil(HW *S) + 10 )

    # Ap = np.int64(np.round(pointA*S))
    # Bp = np.int64(np.round(pointB*S))

    # imageA = np.zeros(HW, dtype=np.int64)
    # imageB = np.zeros(HW, dtype=np.int64)

    # imageA[Ap[:,0], Ap[:,1]] = 1
    # imageB[Bp[:,0], Bp[:,1]] = 1

    imageA = points2bmask(pointA*S, shape = HW, rgbcol=1, rgbbg=0, csize = csize, dsize=dsize, method=method)
    imageB = points2bmask(pointB*S, shape = HW, rgbcol=1, rgbbg=0, csize = csize, dsize=dsize, method=method)

    imageo = interp_shape(imageA, imageB, precision, use_bw=use_bw, **kargs)
    imageo = [ ski.transform.rescale(img, R/S, order=order) for img in imageo]
    points = [ np.c_[np.where(img>thred)]/R for img in imageo]

    # imageo = [ ski.transform.rescale(img>thred, R/S, order=0, anti_aliasing=False) for img in imageo]
    # points = [ np.c_[np.where(img>0)]/R for img in imageo]
    return imageo, points

def interp_shape(top, bottom, precision=0, zscale = 2, use_bw=True, interp_method='linear',
                 kernel = 'regular', n_job =2, use_interp=True,
                 iterations=1, structure=[2,2]):
    '''
    Interpolate between two contours

    Input: top
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if isinstance(precision, (float, int)):
        precision = [precision]
    precision = np.array(precision)
    assert np.all(precision>=0) and np.all(precision<=1), "Precision must be between 0 and 1 (float)"
    precision = np.array(precision)*zscale

    if use_bw:
        top = signed_bwdist(top, iterations=iterations, structure=structure)
        bottom = signed_bwdist(bottom, iterations=iterations, structure=structure)
    r, c = top.shape
    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # interpolation
    if use_interp:
        points = (np.r_[0, zscale], np.arange(r), np.arange(c))
        xs = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))

        def interp_N_grid(points, values, r, c, xs, zpos,  method='linear'):
            xi = np.c_[np.full((r*c), zpos), xs].copy()
            out = interpn(points, values, xi, method=method)
            #yflat = RBFInterpolator(xobs, yobs)(xflat)
            out = out.reshape((r, c))
            return out
        outs = Parallel( n_jobs=n_job, verbose=0)( delayed( interp_N_grid)
                                (points, top_and_bottom, r,c, xs, zpos, method=interp_method) 
                            for zpos in precision)
    else:
        outs = []
        for zpos in precision:
            out = top*(1-zpos) + bottom*zpos
            outs.append(out)
    return outs

def interp_point3d(points, interp_frac=0.5, S=1, R =1,  HW=None, use_bw=True, 
                    down_sample =True, sample_frac=None, sample_method = 'open3d', seed=200504, 
                   thred=0, order=2, verbose=1, **kargs):
    '''
    Input: points:
    [X, Y, Z]
    '''
    if isinstance(interp_frac, (float, int)):
        interp_frac = np.array([interp_frac])

    S,R  = float(S), float(R)
    if HW is None:
        HW = points.max(0)[:2]
    else:
        HW = np.array(HW)
    HW = np.int64( np.ceil(HW *S) + 10 )

    zaxes = np.unique(points[:,2])
    zaxes_space= zaxes[1:] - zaxes[:-1]
    D = len(zaxes)

    images = np.zeros((D,*HW), dtype=np.int64)
    points_size = []
    for idx, iz in enumerate(zaxes):
        ipxy = points[points[:,2]==iz,:2]
        ipxy = np.int64(np.round(ipxy*S))
        images[idx, ipxy[:,0], ipxy[:,1]] = 1
        points_size.append(ipxy.shape[0])

    imageo = interp_shape3d(images, precision=interp_frac, use_bw=use_bw, out_size=False, **kargs)

    pointps = []
    for idx, iprc in enumerate(interp_frac):
        izspace = zaxes[:-1] + zaxes_space * iprc
        imgintp = imageo[idx]
        ipointps = []
        for islc, (izs, imgb) in enumerate(zip(izspace, imgintp)):
            imgbp = ski.transform.rescale(imgb, R/S, order=order)
            ipointp = np.c_[np.where(imgbp>thred)]/R
            # imgbp = ski.transform.rescale(imgb>thred, R/S, order=0, anti_aliasing=False)
            # ipointp = np.c_[np.where(imgbp>0)]/R

            p_size  = ipointp.shape[0]
            ipointp = np.c_[ ipointp, np.full((p_size),izs) ]
            verbose and print(f'inter_frac:{iprc}->slice {islc}-> size {p_size}')

            if down_sample:
                size_a, size_b = points_size[islc], points_size[islc+1]
                size_c = sample_frac or (size_a* (1-iprc)  + size_b*iprc)
                ipointp = down_sample_point(ipointp, samples=size_c,
                                              k_points=None, method =sample_method,
                                              seed=seed)
                verbose and print(f'size {p_size}->down sample {ipointp.shape[0]}')
            ipointps.append(ipointp)
        ipointps = np.vstack(ipointps)
        pointps.append(ipointps)
    pointps = np.vstack(pointps)
    return pointps, imageo

def interp_shape3d(imgbs, precision=0.5, use_bw=True, out_size=False, interp_method='linear',
                   iterations=1, structure=[2, 2]):
    '''
    Interpolate between two contours

    Input: imgbs
            [Z, X,Y] - Image of top contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [Z,X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if isinstance(precision, (float, int)):
        precision = np.array([precision])

    precision = np.array(precision)
    assert np.all(precision >= 0) and np.all(precision <= 1), "Precision must be between 0 and 1 (float)"

    d, r, c = imgbs.shape
    if use_bw:
        imgbm = []
        for img in imgbs:
            img = signed_bwdist(img, iterations=iterations, structure=structure)
            imgbm.append(img)
        imgbs = np.array(imgbm)
        del imgbm

    # create ndgrids
    points = (np.arange(d), np.arange(r), np.arange(c))
    D = d if out_size else d-1
    xs = np.rollaxis(np.mgrid[:D, :r, :c], 0, 4).reshape((D * r * c, -1))
    didx = np.arange(D).astype(np.float64)

    outs = []
    didc = []
    for iperc in precision: # large data
        xi = xs.copy()
        xi[:, 0] = xi[:, 0] + iperc
        # Interpolate for new plane
        out = interpn(points, imgbs, xi, method=interp_method)
        out = out.reshape((D, r, c))

        # Threshold distmap to values above 0
        # out = out > 0
        outs.append(out)
        didc.append(didx+iperc)
    return outs

def down_sample_point(POINTS, samples=None, k_points=None, method ='open3d', seed=0):
    POINTS = np.array(POINTS)
    n_point = POINTS.shape[0]
    if not samples is None:
        samples = int(samples)

    if method=='random':
        np.random.seed(seed)
        obs_idex = np.random.choice(np.arange(n_point), 
                         size=samples, 
                         replace=False, p=None)
        return POINTS[obs_idex]
    elif  method=='open3d':
        import open3d as o3d
        k_points = k_points or int(n_point/samples)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(POINTS)

        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        uni_down_pcd = pcd.uniform_down_sample(every_k_points=k_points)
        down_points = np.asarray(uni_down_pcd.points)
        return down_points
    elif method == 'pyvista':
        import pyvista as pv
        tolerance =  n_point/samples
        pld = pv.PolyData(POINTS)
        dpld = pld.clean(
                point_merging=True,
                merge_tol=tolerance,
                lines_to_points=False,
                polys_to_lines=False,
                strips_to_polys=False,
                inplace=False,
                absolute=False,
                progress_bar=False)
        down_points = np.asarray(dpld.points)
        return down_points

def bwperim(bw, n=4, iterations=2, structure=[2,2]):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    # if n not in (4,8):
    #     raise ValueError('mahotas.bwperim: n must be 4 or 8')
    # print(bw.shape)
    # rows,cols = bw.shape
    #
    # # Translate image by one pixel in all direct
    # # ions
    # north = np.zeros((rows,cols))
    # south = np.zeros((rows,cols))
    # west = np.zeros((rows,cols))
    # east = np.zeros((rows,cols))
    #
    # north[:-1,:] = bw[1:,:]
    # south[1:,:]  = bw[:-1,:]
    # west[:,:-1]  = bw[:,1:]
    # east[:,1:]   = bw[:,:-1]
    # idx = (north == bw) & \
    #       (south == bw) & \
    #       (west  == bw) & \
    #       (east  == bw)
    # plt.imshow(idx.astype(int))
    # plt.title('move')
    # plt.show()
    #
    # if n == 8:
    #     north_east = np.zeros((rows, cols))
    #     north_west = np.zeros((rows, cols))
    #     south_east = np.zeros((rows, cols))
    #     south_west = np.zeros((rows, cols))
    #     north_east[:-1, 1:]   = bw[1:, :-1]
    #     north_west[:-1, :-1]  = bw[1:, 1:]
    #     south_east[1:, 1:]    = bw[:-1, :-1]
    #     south_west[1:, :-1]   = bw[:-1, 1:]
    #     idx &= (north_east == bw) & \
    #            (south_east == bw) & \
    #            (south_west == bw) & \
    #            (north_west == bw)
    # return ~idx * bw
    # struct1 = generate_binary_structure(2, 1)
    struct = generate_binary_structure(*structure)
    return binary_dilation(bw, iterations=iterations, structure=struct, border_value=0) - bw

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im
# https://stackoverflow.com/questions/48818373/interpolate-between-two-images

def signed_bwdist(im, iterations=2, structure=[2,2]):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    imbp = bwperim(im.copy(), iterations=iterations, structure=structure)
    imbd = bwdist(imbp)
    ima = -imbd * np.logical_not(im) + imbd*im
    #@cc.pl.qview(im, imbp, imbd, ima, titles=('raw', 'diff_neib', 'dist', 'masked'), show=True)
    return ima

def points2bmask(posn, shape=None, rgbcol=(255, 255, 255), rgbbg=(0,0,0), 
               csize = 5, dsize=5,# thresh=127, maxval=255,
               method='circle', margin = [0, 0]):

    ipos = np.int64(np.round(posn))
    if shape is None:
        shape = ipos.max(0).astype(np.int64) + 1
    h, w = shape
    h += margin[0]
    w += margin[1]

    if isinstance( rgbcol, (int, float) ):
        assert  isinstance( rgbbg, type(rgbcol))
        img = np.full((h, w), rgbbg)
        val = [rgbcol] * posn.shape[0]

    elif isinstance( rgbcol, (list, tuple) ) and (3<= len(rgbcol)<=4):
        assert  isinstance( rgbbg, type(rgbcol))
        img = np.full((h, w, len(rgbcol)), rgbbg)
        val = [rgbcol] * posn.shape[0]

        if isinstance(dsize, int):
            dsize = [dsize, dsize, 1, 1][:len(rgbcol)]

    elif len(rgbcol) == posn.shape[0]:
        assert  isinstance( rgbbg, (int, float))
        img = np.full((h, w), rgbbg)
        val = rgbcol
    else:
        raise('error type of rgbcol')

    if method =='dilation':
        val = np.array(val)
        from scipy.ndimage import grey_dilation, grey_erosion
        img[ipos[:,0], ipos[:,1]] = val
        # img[ipos[:,0]+1, ipos[:,1]+1] = rgbcol
        # img[ipos[:,0], ipos[:,1]+1] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]-1] = rgbcol
        # img[ipos[:,0], ipos[:,1]-1] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]] = rgbcol
        img = grey_dilation(img, size=dsize)
        # img = grey_erosion(img, size=(5,5,1));

    else:
        import cv2
        ipos = ipos[:,[1,0]]
        ndix = range(ipos.shape[0])
        img = img.astype(np.int32).copy()

        for i in ndix:
            cv2.circle(img, (int(ipos[i][0]), int(ipos[i][1])), csize, val[i], -1)
    return img
    # binary_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary_mask = cv2.threshold(binary_mask, thresh, maxval, cv2.THRESH_BINARY)
    # binary_mask = ski.filters.gaussian(mask1, 30)

def pointsinmask(mask, points):
    pos = np.int64(np.round(points))
    inc = (pos[:,0] < mask.shape[0]) & (pos[:,1]< mask.shape[1] )
    ipos = pos[inc]
    iinc = mask[ipos[:,0], ipos[:,1]].astype(np.bool_)
    inc[inc] = iinc
    return inc
