from typing import Optional, Tuple, Union, List

import numpy as np
import pyvista as pv

from pyvista import DataSet, PolyData, UnstructuredGrid
from scipy.spatial.distance import cdist
import matplotlib as mpl

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

def surface_mesh(
    pc: PolyData,
    key_added: str = "groups",
    label: str = "surface",
    color: Optional[str] = "gainsboro",
    alpha: Union[float, int] = 1.0,
    uniform_pc: bool = False,
    uniform_pc_alpha: Union[float, int] = 0,
    cs_method: Literal["pyvista", "alpha_shape", "ball_pivoting", "poisson", "marching_cube"] = "marching_cube",
    cs_args: Optional[dict] = None,
    nsub: Optional[int] = 3,
    nclus: int = 20000,
    smooth: Optional[int] = 3000,
    scale_distance: Union[float, int, list, tuple] = None,
    scale_factor: Union[float, int, list, tuple] = None,
) -> Tuple[Union[PolyData, UnstructuredGrid, None], PolyData, Optional[str]]:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pc: A point cloud model.
        key_added: The key under which to add the labels.
        label: The label of reconstructed surface mesh model.
        color: Color to use for plotting mesh. The default ``color`` is ``'gainsboro'``.
        alpha: The opacity of the color to use for plotting mesh. The default ``alpha`` is ``0.8``.
        uniform_pc: Generates a uniform point cloud with a larger number of points.
        uniform_pc_alpha: Specify alpha (or distance) value to control output of this filter.
        cs_method: The methods of generating a surface mesh. Available ``cs_method`` are:

                * ``'pyvista'``: Generate a 3D tetrahedral mesh based on pyvista.
                * ``'alpha_shape'``: Computes a triangle mesh on the alpha shape algorithm.
                * ``'ball_pivoting'``: Computes a triangle mesh based on the Ball Pivoting algorithm.
                * ``'poisson'``: Computes a triangle mesh based on thee Screened Poisson Reconstruction.
                * ``'marching_cube'``: Computes a triangle mesh based on the marching cube algorithm.
        cs_args: Parameters for various surface reconstruction methods. Available ``cs_args`` are:
                * ``'pyvista'``: {'alpha': 0}
                * ``'alpha_shape'``: {'alpha': 2.0}
                * ``'ball_pivoting'``: {'radii': [1]}
                * ``'poisson'``: {'depth': 8, 'width'=0, 'scale'=1.1, 'linear_fit': False, 'density_threshold': 0.01}
                * ``'marching_cube'``: {'levelset': 0, 'mc_scale_factor': 1}
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.
        smooth: Number of iterations for Laplacian smoothing.
        scale_distance: The distance by which the model is scaled. If ``scale_distance`` is float, the model is scaled same
                        distance along the xyz axis; when the ``scale factor`` is list, the model is scaled along the xyz
                        axis at different distance. If ``scale_distance`` is None, there will be no scaling based on distance.
        scale_factor: The scale by which the model is scaled. If ``scale factor`` is float, the model is scaled along the
                      xyz axis at the same scale; when the ``scale factor`` is list, the model is scaled along the xyz
                      axis at different scales. If ``scale_factor`` is None, there will be no scaling based on scale factor.

    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            ``uniform_surf.cell_data[key_added]``, the ``label`` array;
            ``uniform_surf.cell_data[f'{key_added}_rgba']``, the rgba colors of the ``label`` array.
        inside_pc: A point cloud, which contains the following properties:
            ``inside_pc.point_data['obs_index']``, the obs_index of each coordinate in the original adata.
            ``inside_pc.point_data[key_added]``, the ``groupby`` information.
            ``inside_pc.point_data[f'{key_added}_rgba']``, the rgba colors of the ``groupby`` information.
        plot_cmap: Recommended colormap parameter values for plotting.
    """

    # Generates a uniform point cloud with a larger number of points or not.
    cloud = uniform_larger_pc(pc=pc, alpha=uniform_pc_alpha, nsub=3, nclus=20000) if uniform_pc else pc.copy()

    if cs_method == "marching_cube":
        _cs_args = {"levelset": 0, "mc_scale_factor": 1}
        if not (cs_args is None):
            _cs_args.update(cs_args)

        surf = marching_cube_mesh(pc=cloud, levelset=_cs_args["levelset"], mc_scale_factor=_cs_args["mc_scale_factor"])

    else:
        raise ValueError(
            "`cs_method` value is wrong."
            "\nAvailable `cs_method` are: `'pyvista'`, `'alpha_shape'`, `'ball_pivoting'`, `'poisson'`, `'marching_cube'`."
        )

    # Removes unused points and degenerate cells.
    csurf = clean_mesh(mesh=surf)

    uniform_surfs = []
    for sub_surf in csurf.split_bodies():
        # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
        sub_fix_surf = fix_mesh(mesh=sub_surf.extract_surface())

        # Get a uniformly meshed surface using voronoi clustering.
        sub_uniform_surf = uniform_mesh(mesh=sub_fix_surf, nsub=nsub, nclus=nclus)
        uniform_surfs.append(sub_uniform_surf)
    uniform_surf = merge_models(models=uniform_surfs)
    uniform_surf = uniform_surf.extract_surface().triangulate().clean()

    # Adjust point coordinates using Laplacian smoothing.
    if not (smooth is None):
        uniform_surf = smooth_mesh(mesh=uniform_surf, n_iter=smooth)

    # Scale the surface mesh.
    uniform_surf = scale_model(model=uniform_surf, distance=scale_distance, scale_factor=scale_factor)

    # Add labels and the colormap of the surface mesh.
    labels = np.asarray([label] * uniform_surf.n_cells, dtype=str)
    _, plot_cmap = add_model_labels(
        model=uniform_surf,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
        inplace=True,
    )

    # Clip the original pc using the reconstructed surface and reconstruct new point cloud.
    select_pc = pc.select_enclosed_points(surface=uniform_surf, check_surface=False)
    select_pc1 = select_pc.threshold(0.5, scalars="SelectedPoints").extract_surface()
    select_pc2 = select_pc.threshold(0.5, scalars="SelectedPoints", invert=True).extract_surface()
    inside_pc = select_pc1 if select_pc1.n_points > select_pc2.n_points else select_pc2

    return uniform_surf, inside_pc, plot_cmap

def voxelize_mesh(
    mesh: Union[PolyData, UnstructuredGrid],
    voxel_pc: Union[PolyData, UnstructuredGrid] = None,
    key_added: str = "groups",
    label: str = "voxel",
    color: Optional[str] = "gainsboro",
    alpha: Union[float, int] = 1.0,
    density: Optional[int] = None,
) -> Tuple[Union[UnstructuredGrid], Optional[str]]:
    """
    Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh model.
        voxel_pc: A voxel model which contains the ``voxel_pc.cell_data['obs_index']`` and ``voxel_pc.cell_data[key_added]``.
        key_added: The key under which to add the labels.
        label: The label of reconstructed voxel model.
        color: Color to use for plotting mesh. The default color is ``'gainsboro'``.
        alpha: The opacity of the color to use for plotting model. The default alpha is ``0.8``.
        smooth: The smoothness of the voxel model.

    Returns:
        voxel_model: A reconstructed voxel model, which contains the following properties:
            `voxel_model.cell_data[key_added]`, the `label` array;
            `voxel_model.cell_data[f'{key_added}_rgba']`,  the rgba colors of the `label` array.
            `voxel_model.cell_data['obs_index']`, the cell labels if not (voxel_pc is None).
        plot_cmap: Recommended colormap parameter values for plotting.
    """

    voxel_model = pv.voxelize(mesh, density=density, check_surface=False)

    # Add labels and the colormap of the volumetric mesh
    labels = np.array([label] * voxel_model.n_cells).astype(str)
    _, plot_cmap = add_model_labels(
        model=voxel_model,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
        inplace=True,
    )
    if not (voxel_pc is None):
        voxel_model.cell_data["obs_index"] = np.asarray(["no_cell"] * voxel_model.n_cells).astype(str)
        voxel_model = merge_models(models=[voxel_model, voxel_pc])

    return voxel_model, plot_cmap

def voxelize_zslice(
    mesh: Union[PolyData, UnstructuredGrid],
    voxel_pc: Union[PolyData, UnstructuredGrid] = None,
    zstep: Optional[int] = 2, 
    key_added: str = "groups",
    label: str = "voxel",
    color: Optional[str] = "gainsboro",
    alpha: Union[float, int] = 1.0,
    density: Optional[int] = None,
) -> Tuple[Union[UnstructuredGrid], Optional[str]]:
    """
    Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh model.
        voxel_pc: A voxel model which contains the ``voxel_pc.cell_data['obs_index']`` and ``voxel_pc.cell_data[key_added]``.
        key_added: The key under which to add the labels.
        label: The label of reconstructed voxel model.
        color: Color to use for plotting mesh. The default color is ``'gainsboro'``.
        alpha: The opacity of the color to use for plotting model. The default alpha is ``0.8``.
        smooth: The smoothness of the voxel model.

    Returns:
        voxel_model: A reconstructed voxel model, which contains the following properties:
            `voxel_model.cell_data[key_added]`, the `label` array;
            `voxel_model.cell_data[f'{key_added}_rgba']`,  the rgba colors of the `label` array.
            `voxel_model.cell_data['obs_index']`, the cell labels if not (voxel_pc is None).
        plot_cmap: Recommended colormap parameter values for plotting.
    """

    voxel_model = pv.voxelize(mesh, density=density, check_surface=False)

    # Add labels and the colormap of the volumetric mesh
    # labels = np.array([label] * voxel_model.n_cells).astype(str)
    # _, plot_cmap = add_model_labels(
    #     model=voxel_model,
    #     labels=labels,
    #     key_added=key_added,
    #     where="cell_data",
    #     colormap=color,
    #     alphamap=alpha,
    #     inplace=True,
    # )
    if not (voxel_pc is None):
        voxel_model.cell_data["obs_index"] = np.asarray(["no_cell"] * voxel_model.n_cells).astype(str)
        voxel_model = merge_models(models=[voxel_model, voxel_pc])

    return voxel_model

def smooth_mesh(mesh: PolyData, n_iter: int = 100, **kwargs) -> PolyData:
    """
    Adjust point coordinates using Laplacian smoothing.
    https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.smooth.html#pyvista.PolyData.smooth

    Args:
        mesh: A mesh model.
        n_iter: Number of iterations for Laplacian smoothing.
        **kwargs: The rest of the parameters in pyvista.PolyData.smooth.

    Returns:
        smoothed_mesh: A smoothed mesh model.
    """

    smoothed_mesh = mesh.smooth(n_iter=n_iter, **kwargs)

    return smoothed_mesh

def fix_mesh(mesh: PolyData) -> PolyData:
    """Repair the mesh where it was extracted and subtle holes along complex parts of the mesh."""

    # Check pymeshfix package
    try:
        import pymeshfix as mf
    except ImportError:
        raise ImportError(
            "You need to install the package `pymeshfix`. \nInstall pymeshfix via `pip install pymeshfix`"
        )

    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=False)
    fixed_mesh = meshfix.mesh.triangulate().clean()

    if fixed_mesh.n_points == 0:
        raise ValueError(
            f"The surface cannot be Repaired. " f"\nPlease change the method or parameters of surface reconstruction."
        )

    return fixed_mesh

def clean_mesh(mesh: PolyData) -> PolyData:
    """Removes unused points and degenerate cells."""

    sub_meshes = mesh.split_bodies()
    n_mesh = len(sub_meshes)

    if n_mesh == 1:
        return mesh
    else:
        inside_number = []
        for i, main_mesh in enumerate(sub_meshes[:-1]):
            main_mesh = pv.PolyData(main_mesh.points, main_mesh.cells)
            for j, check_mesh in enumerate(sub_meshes[i + 1 :]):
                check_mesh = pv.PolyData(check_mesh.points, check_mesh.cells)
                inside = check_mesh.select_enclosed_points(main_mesh, check_surface=False).threshold(0.5)
                inside = pv.PolyData(inside.points, inside.cells)
                if check_mesh == inside:
                    inside_number.append(i + 1 + j)

        cm_number = list(set([i for i in range(n_mesh)]).difference(set(inside_number)))
        if len(cm_number) == 1:
            cmesh = sub_meshes[cm_number[0]]
        else:
            cmesh = merge_models([sub_meshes[i] for i in cm_number])

        return pv.PolyData(cmesh.points, cmesh.cells)

def merge_models(
    models: List[PolyData or UnstructuredGrid or DataSet],
) -> PolyData or UnstructuredGrid:
    """Merge all models in the `models` list. The format of all models must be the same."""

    merged_model = models[0]
    for model in models[1:]:
        merged_model = merged_model.merge(model)

    return merged_model

def uniform_larger_pc(
    pc: PolyData,
    alpha: Union[float, int] = 0,
    nsub: Optional[int] = 5,
    nclus: int = 20000,
) -> PolyData:
    """
    Generates a uniform point cloud with a larger number of points.
    If the number of points in the original point cloud is too small or the distribution of the original point cloud is
    not uniform, making it difficult to construct the surface, this method can be used for preprocessing.

    Args:
        pc: A point cloud model.
        alpha: Specify alpha (or distance) value to control output of this filter.
               For a non-zero alpha value, only edges or triangles contained within a sphere centered at mesh vertices
               will be output. Otherwise, only triangles will be output.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        new_pc: A uniform point cloud with a larger number of points.
    """
    coords = np.asarray(pc.points)
    coords_z = np.unique(coords[:, 2])

    slices = []
    for z in coords_z:
        slice_coords = coords[coords[:, 2] == z]
        slice_cloud = pv.PolyData(slice_coords)
        if len(slice_coords) >= 3:
            slice_plane = slice_cloud.delaunay_2d(alpha=alpha).triangulate().clean()
            uniform_plane = uniform_mesh(mesh=slice_plane, nsub=nsub, nclus=nclus)
            slices.append(uniform_plane)
        else:
            slices.append(slice_cloud)

    slices_mesh = merge_models(models=slices)
    new_pc = pv.PolyData(slices_mesh.points).clean()
    return new_pc

def merge_models(
    models: List[PolyData or UnstructuredGrid or DataSet],
) -> PolyData or UnstructuredGrid:
    """Merge all models in the `models` list. The format of all models must be the same."""

    merged_model = models[0]
    for model in models[1:]:
        merged_model = merged_model.merge(model)

    return merged_model

def uniform_mesh(mesh: PolyData, nsub: Optional[int] = 3, nclus: int = 20000) -> PolyData:
    """
    Generate a uniformly meshed surface using voronoi clustering.

    Args:
        mesh: A mesh model.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        new_mesh: A uniform mesh model.
    """
    # Check pyacvd package
    try:
        import pyacvd
    except ImportError:
        raise ImportError("You need to install the package `pyacvd`. \nInstall pyacvd via `pip install pyacvd`")

    # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
    if not (nsub is None):
        mesh.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

    # Uniformly remeshing.
    clustered = pyacvd.Clustering(mesh)
    clustered.cluster(nclus)

    new_mesh = clustered.create_mesh().triangulate().clean()
    return new_mesh

def marching_cube_mesh(pc: PolyData, levelset: Union[int, float] = 0, mc_scale_factor: Union[int, float] = 1.0):
    """
    Computes a triangle mesh from a point cloud based on the marching cube algorithm.
    Algorithm Overview:
        The algorithm proceeds through the scalar field, taking eight neighbor locations at a time (thus forming an
        imaginary cube), then determining the polygon(s) needed to represent the part of the iso-surface that passes
        through this cube. The individual polygons are then fused into the desired surface.

    Args:
        pc: A point cloud model.
        levelset: The levelset of iso-surface. It is recommended to set levelset to 0 or 0.5.
        mc_scale_factor: The scale of the model. The scaled model is used to construct the mesh model.

    Returns:
        A mesh model.
    """
    try:
        import mcubes
    except ImportError:
        raise ImportError(
            "You need to install the package `mcubes`." "\nInstall mcubes via `pip install --upgrade PyMCubes`"
        )

    pc = pc.copy()

    # Move the model so that the coordinate minimum is at (0, 0, 0).
    raw_points = np.asarray(pc.points)
    pc.points = new_points = raw_points - np.min(raw_points, axis=0)

    # Generate new models for calculatation.
    dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
    row, col = np.diag_indices_from(dist)
    dist[row, col] = None
    max_dist = np.nanmin(dist, axis=1).max()
    mc_sf = max_dist * mc_scale_factor

    scale_pc = scale_model(model=pc, scale_factor=1 / mc_sf)
    scale_pc_points = scale_pc.points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)

    # Generate grid for calculatation based on new model.
    volume_array = np.zeros(
        shape=[
            scale_pc_points[:, 0].max() + 3,
            scale_pc_points[:, 1].max() + 3,
            scale_pc_points[:, 2].max() + 3,
        ]
    )
    volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1

    # Extract the iso-surface based on marching cubes algorithm.
    # volume_array = mcubes.smooth(volume_array)
    vertices, triangles = mcubes.marching_cubes(volume_array, levelset)

    if len(vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `marching_cube` method.")

    v = np.asarray(vertices).astype(np.float64)
    f = np.asarray(triangles).astype(np.int64)
    f = np.c_[np.full(len(f), 3), f]

    # Generate mesh model.
    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    mesh.clean(inplace=True)
    mesh = scale_model(model=mesh, scale_factor=mc_sf)

    # Transform.
    scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf)
    mesh.points = rigid_transform(
        coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
    )
    return mesh

def scale_model(
    model: Union[PolyData, UnstructuredGrid],
    distance: Union[float, int, list, tuple] = None,
    scale_factor: Union[float, int, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
    inplace: bool = False,
) -> Union[PolyData, UnstructuredGrid, None]:
    """
    Scale the model around the center of the model.

    Args:
        model: A 3D reconstructed model.
        distance: The distance by which the model is scaled. If `distance` is float, the model is scaled same distance
                  along the xyz axis; when the `scale factor` is list, the model is scaled along the xyz axis at
                  different distance. If `distance` is None, there will be no scaling based on distance.
        scale_factor: The scale by which the model is scaled. If `scale factor` is float, the model is scaled along the
                      xyz axis at the same scale; when the `scale factor` is list, the model is scaled along the xyz
                      axis at different scales. If `scale_factor` is None, there will be no scaling based on scale factor.
        scale_center: Scaling center. If `scale factor` is None, the `scale_center` will default to the center of the model.
        inplace: Updates model in-place.

    Returns:
        model_s: The scaled model.
    """

    model_s = model.copy() if not inplace else model

    if not (distance is None):
        model_s = _scale_model_by_distance(model=model_s, distance=distance, scale_center=scale_center)

    if not (scale_factor is None):
        model_s = _scale_model_by_scale_factor(model=model_s, scale_factor=scale_factor, scale_center=scale_center)

    model_s = model_s.triangulate()

    return model_s if not inplace else None

def _scale_model_by_distance(
    model: DataSet,
    distance: Union[int, float, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
) -> DataSet:
    # Check the distance.
    distance = distance if isinstance(distance, (tuple, list)) else [distance] * 3
    if len(distance) != 3:
        raise ValueError(
            "`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the distance.
    for i, (d, c) in enumerate(zip(distance, scale_center)):
        p2c_bool = np.asarray(model.points[:, i] - c) > 0
        model.points[:, i][p2c_bool] += d
        model.points[:, i][~p2c_bool] -= d

    return model


def _scale_model_by_scale_factor(
    model: DataSet,
    scale_factor: Union[int, float, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
) -> DataSet:
    # Check the scaling factor.
    scale_factor = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 3
    if len(scale_factor) != 3:
        raise ValueError(
            "`scale_factor` value is wrong."
            "\nWhen `scale_factor` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the scale center.
    for i, (f, c) in enumerate(zip(scale_factor, scale_center)):
        model.points[:, i] = (model.points[:, i] - c) * f + c

    return model


def add_model_labels(
    model: Union[PolyData, UnstructuredGrid],
    labels: np.ndarray,
    key_added: str = "groups",
    where: Literal["point_data", "cell_data"] = "cell_data",
    colormap: Union[str, list, dict, np.ndarray] = "rainbow",
    alphamap: Union[float, list, dict, np.ndarray] = 1.0,
    mask_color: Optional[str] = "gainsboro",
    mask_alpha: Optional[float] = 0.0,
    inplace: bool = False,
) -> Tuple[Optional[PolyData or UnstructuredGrid], Optional[Union[str]]]:
    """
    Add rgba color to each point of model based on labels.

    Args:
        model: A reconstructed model.
        labels: An array of labels of interest.
        key_added: The key under which to add the labels.
        where: The location where the label information is recorded in the model.
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Color to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
        inplace: Updates model in-place.

    Returns:
        A model, which contains the following properties:
            ``model.cell_data[key_added]`` or ``model.point_data[key_added]``, the labels array;
            ``model.cell_data[f'{key_added}_rgba']`` or ``model.point_data[f'{key_added}_rgba']``, the rgba colors of the labels.
        plot_cmap: Recommended colormap parameter values for plotting.
    """

    model = model.copy() if not inplace else model
    labels = np.asarray(labels).flatten()

    if not np.issubdtype(labels.dtype, np.number):
        cu_arr = np.sort(np.unique(labels), axis=0).astype(object)
        raw_labels_hex = labels.copy().astype(object)
        raw_labels_alpha = labels.copy().astype(object)
        raw_labels_hex[raw_labels_hex == "mask"] = mpl.colors.to_hex(mask_color)
        raw_labels_alpha[raw_labels_alpha == "mask"] = mask_alpha

        # Set raw hex.
        if isinstance(colormap, str):
            if colormap in list(mpl.colormaps()):
                lscmap = mpl.cm.get_cmap(colormap)
                raw_hex_list = [mpl.colors.to_hex(lscmap(i)) for i in np.linspace(0, 1, len(cu_arr))]
                for label, color in zip(cu_arr, raw_hex_list):
                    raw_labels_hex[raw_labels_hex == label] = color
            else:
                raw_labels_hex[raw_labels_hex != "mask"] = mpl.colors.to_hex(colormap)
        elif isinstance(colormap, dict):
            for label, color in colormap.items():
                raw_labels_hex[raw_labels_hex == label] = mpl.colors.to_hex(color)
        elif isinstance(colormap, list) or isinstance(colormap, np.ndarray):
            raw_hex_list = np.array([mpl.colors.to_hex(color) for color in colormap]).astype(object)
            for label, color in zip(cu_arr, raw_hex_list):
                raw_labels_hex[raw_labels_hex == label] = color
        else:
            raise ValueError("`colormap` value is wrong." "\nAvailable `colormap` types are: `str`, `list` and `dict`.")

        # Set raw alpha.
        if isinstance(alphamap, float) or isinstance(alphamap, int):
            raw_labels_alpha[raw_labels_alpha != "mask"] = alphamap
        elif isinstance(alphamap, dict):
            for label, alpha in alphamap.items():
                raw_labels_alpha[raw_labels_alpha == label] = alpha
        elif isinstance(alphamap, list) or isinstance(alphamap, np.ndarray):
            raw_labels_alpha = np.asarray(alphamap).astype(object)
        else:
            raise ValueError(
                "`alphamap` value is wrong." "\nAvailable `alphamap` types are: `float`, `list` and `dict`."
            )

        # Set rgba.
        labels_rgba = [mpl.colors.to_rgba(c, alpha=a) for c, a in zip(raw_labels_hex, raw_labels_alpha)]
        labels_rgba = np.array(labels_rgba).astype(np.float32)

        # Added rgba of the labels.
        if where == "point_data":
            model.point_data[f"{key_added}_rgba"] = labels_rgba
        else:
            model.cell_data[f"{key_added}_rgba"] = labels_rgba

        plot_cmap = None
    else:
        plot_cmap = colormap

    # Added labels.
    if where == "point_data":
        model.point_data[key_added] = labels
    else:
        model.cell_data[key_added] = labels

    return model if not inplace else None, plot_cmap

def rigid_transform(
    coords: np.ndarray,
    coords_refA: np.ndarray,
    coords_refB: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal transformation based on the two sets of points and apply the transformation to other points.

    Args:
        coords: Coordinate matrix needed to be transformed.
        coords_refA: Referential coordinate matrix before transformation.
        coords_refB: Referential coordinate matrix after transformation.

    Returns:
        The coordinate matrix after transformation
    """
    # Check the spatial coordinates

    coords, coords_refA, coords_refB = (
        coords.copy(),
        coords_refA.copy(),
        coords_refB.copy(),
    )
    assert (
        coords.shape[1] == coords_refA.shape[1] == coords_refA.shape[1]
    ), "The dimensions of the input coordinates must be uniform, 2D or 3D."
    coords_dim = coords.shape[1]
    if coords_dim == 2:
        coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]
        coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
        coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

    # Compute optimal transformation based on the two sets of points.
    coords_refA = coords_refA.T
    coords_refB = coords_refB.T

    centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
    centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

    Am = coords_refA - centroid_A
    Bm = coords_refB - centroid_B
    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # Apply the transformation to other points
    new_coords = (R @ coords.T) + t
    new_coords = np.asarray(new_coords.T)
    return new_coords[:, :2] if coords_dim == 2 else new_coords