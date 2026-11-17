import os
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ._colors import color_palette, ListedColormap, cmap1, cmap1a, to_rgba, hex_to_rgb, random_colors
from ..utilis._arrays import list_iter, vartype
from ._utilis import colrows, clipdata, get_minmax

def surface(adata, meshlist=None, use_raw=False, groupby = None, splitby=None,
            basis='spatial3d', cmap=None, gcolors = None, outpre =None, format ='mp4', 
            titles=None, legend_titles=None, last_as_default=False,
            save=None,
            **kargs):
    """Plot AnnData coordinates colored by observations or gene expression.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing three-dimensional coordinates in obsm[basis], observation
        labels in obs, and optional gene expression in X or raw.
    meshlist : sequence of pyvista.DataSet or None, default: None
        Surface meshes added to the active renderer. None entries are skipped. Meshes are
        copied before thresholding or rotation.
    use_raw : bool, default: False
        Read gene-expression values from adata.raw instead of adata.X. Applies only when
        groupby is recognized in adata.var_names and requires adata.raw.
    groupby : str, sequence of str, or None, default: None
        Observation-column names or gene names to plot when splitby is None. One plotter
        result is produced per recognized item. None and unknown names are skipped by the
        current implementation.
    splitby : str, sequence of str, or None, default: None
        Observation columns whose categories define separate point subsets. Each nonempty
        category is plotted with surfaces_df; this branch takes precedence over groupby.
    basis : str, default: 'spatial3d'
        Key in adata.obsm holding an (N, 3) coordinate array.
    cmap : object or None, default: None
        Retained parameter; currently unused. Use gcolors to supply continuous colormaps or
        categorical palettes.
    gcolors : dict, list, color, colormap, or None, default: None
        Color overrides for the groupby branch: a dictionary keyed by group name, a list of
        palettes/colormaps in groupby order, or one shared value. For one categorical group,
        wrap its palette in a one-item list or a group-keyed dictionary. Leave this as None
        in the current splitby branch.
    outpre : str or None, default: None
        Output prefix used when save is None. Files are named outpre.group.format or
        outpre.category.format.
    format : str, default: 'mp4'
        Filename extension appended to outpre; it has no effect when an explicit save path
        is supplied.
    titles : str, sequence of str, or None, default: None
        Plot titles expanded by list_iter. Indexed by groupby item or by splitby column; a
        missing/false value falls back to the current group/category name.
    legend_titles : str, sequence, or None, default: None
        Scalar-bar titles expanded by list_iter and indexed like titles. Forwarded as
        legend_title to the lower-level plotting function.
    last_as_default : bool, default: False
        Controls list_iter expansion of titles and legend_titles in this wrapper.
    save : str, path-like, or None, default: None
        Explicit output path, taking precedence over outpre. The same path is passed to
        every generated plot, so multiple outputs can overwrite it.
    **kargs : dict
        Additional arguments passed unchanged to surfaces3d or surfaces_df, such as pl,
        return_pl, point_psize, show, and show_scalar_bar.

    Returns
    -------
    plots : list
        Results from surfaces3d/surfaces_df, normally plotter objects in plotting
        order. Their open/closed state depends on return_pl and show.

    Notes
    -----
    Coordinates are taken directly from obsm[basis]; meshes must already use the same
    coordinate system. Category palettes default to adata.uns[group + "_colors"] when
    present. Continuous data default to cmap1a.
    The groupby branch calls surfaces3d; splitby calls surfaces_df. Currently, the splitby
    color-override branch references an unassigned group variable when gcolors is supplied.
    Use gcolors=None for that branch. No behavior is changed by this documentation.

    Examples
    --------
    >>> plots = surface(adata, groupby="Batch", basis="spatial3d",
    ...                 pl=pl, return_pl=True, show=False)
    """
    import pyvista as pv
    legend_titles = list_iter(legend_titles, last_as_default=last_as_default)
    titles = list_iter(titles, last_as_default=last_as_default)
    pls = []
    if not splitby is None:
        if (type(splitby) == str):
            splitby = [splitby]
        for k, isplit in enumerate(splitby):
            try:
                Order = adata.obs[isplit].cat.remove_unused_categories().cat.categories
            except:
                Order= adata.obs[isplit].unique()
            if (not gcolors is None) and (group in gcolors):
                my_colors = gcolors.get(group)
            elif f'{isplit}_colors' in adata.uns.keys():
                my_colors = adata.uns[f'{isplit}_colors']
            else:
                my_colors = color_palette(len(Order))

            for i, (icolor, iorder) in enumerate(zip(my_colors, Order)):
                idata = adata[adata.obs[isplit] == iorder]
                if not len(idata):
                    continue
                posdf = pd.DataFrame(idata.obsm[basis], columns=['x','y','z'])
                posdf[iorder] = iorder

                if not save is None:
                    isave = save
                elif outpre:
                    isave = f'{outpre}.{iorder}.{format}'
                else:
                    isave = outpre
                ipl = surfaces_df(
                                points=posdf, 
                                meshlist = meshlist, 
                                groupby=iorder, 
                                title= titles[k] or f'{iorder}',
                                legend_title = legend_titles[k],
                                points_colors=icolor, 
                                save = isave,
                                **kargs)
                pls.append(ipl)
    else:
        if (groupby is None) or (type(groupby) == str):
            groupby = [groupby]
        for i, group in enumerate(groupby):
            if group in adata.obs.columns:
                gdata = adata.obs[group].reset_index(drop=True)

            elif group in adata.var_names:
                if use_raw:
                    gdata = adata.raw.to_adata()[:, group].X
                else:
                    gdata = adata[:, group].X
                if issparse(gdata):
                    gdata = gdata.toarray().flatten()
                else:
                    gdata = gdata.flatten()
            else:
                continue
            gdtype = vartype(gdata)
            if gdtype == 'continuous':
                my_cmap = cmap1a
                if not gcolors is None:
                    if type(gcolors) == dict:
                        my_cmap = gcolors.get(group, cmap1a)
                    elif (type(gcolors) == list):
                        assert len(gcolors)>= len(groupby)
                        my_cmap = gcolors[i]
                    else:
                        my_cmap = gcolors

            else:
                if f'{group}_colors' in adata.uns.keys():
                    defa_col = adata.uns[f'{group}_colors']
                else:
                    defa_col = color_palette(np.unique(gdata).shape[0])
                my_cmap = defa_col
                if not gcolors is None:
                    if type(gcolors) == dict:
                        my_cmap = gcolors.get(group, defa_col)
                    elif (type(gcolors) == list):
                        assert len(gcolors)>= len(groupby)
                        my_cmap = gcolors[i]
                    else:
                        my_cmap = gcolors

            if group is None:
                posdf = pv.PolyData(adata.obsm[basis])
                gname='locs'
                posdf[gname] = 'locs'
            else:
                posdf = pd.DataFrame(adata.obsm[basis], columns=['x','y','z'])
                posdf[group] = gdata
                gname = group

            if not save is None:
                isave = save
            elif outpre:
                isave = f'{outpre}.{gname}.{format}'
            else:
                isave = outpre
            ipl = surfaces3d(meshlist = meshlist, 
                             pointslist=[posdf], 
                            groupbys=[gname], 
                            title= titles[i] or f'{gname}',
                            legend_title = legend_titles[i],
                            points_colors=[my_cmap], 
                            save = isave,
                            **kargs)
            pls.append(ipl)
    return pls

def surfaces3d(meshlist=None, pointslist = None, 
                groupbys= None,
                scalars = None,
                points_colors = None,
                opacity=0.2,
                point_psize=0.5, 
                point_opacity=1,
                mesh_psize=None,
                mesh_style=None,     
                edge_opacity=None,
                window_size=None,
                image_scale=None,
                edge_colors='#AAAAAA', 
                font_family = 'arial', #courier
                colors='#AAAAAA',
                theme_style=None,

                show_actor = False, actor_size = 15, 
                startpos = 10, actor_space=5,
                color_off='grey', color_bg='grey',
                mvmin = None, mvmax = None,
                pvmin = None, pvmax = None, ptmin=None,
                ppmin=None, ppmax=None,dropmin=True,
                symcolor_bar=False,
                clims = None,
                show_edges=True, 
                na_color='grey',

                jupyter_backend = None,
                line_smoothing=False,
                allow_empty_mesh=True,
                shade=False,
                # theme_name=None, 
                theme=None, 
                xyzscales=None,
                background_color='black', 
                font_color=None,
                legend_font_color = None,
                main_title_color=None, 
                font_size =15, title_size =15,
                main_title_font =None,main_title_size=15,
                cpos=None, 
                show_scalar_bar=True,    
                start_xvfb = False, 
                server_proxy_enabled=None, 
                server_proxy_prefix=None,
                notebook=None, 
                off_screen=None,

                view_angle = None, azimuth=None, roll=None, elevation=None, zoom=None,
                framerate=12, painter=True, view_up = None, viewup=(0, 0, 0),
                factor=2, n_points=120, step=0.1, line_width=None,
                shift=0, focus=None, quality=8, write_frames=True,
                show=True, raster=True,
                title=None, view_isometric=False,
                title_position='upper_edge', 

                points_to_glyph = False,
                point_radius =1,
                theta_resolution=16,
                phi_resolution=16,
                point_styles='points',
                render_points_as_spheres=True, 
                smooth_shading=True,
                anti_aliasing=True, 
                
                point_lighting=True, 
                mesh_lighting=False,

                aa_type ='fxaa', multi_samples=None, all_renderers=True,
                rotate_y = None, rotate_z = None, rotate_x = None, vtkz_format='json',
                save=None, save_type=None, 
                pl = None,
                return_pl = False,

                Light = None,
                legend_x_pos = 0.9,
                legend_y_pos = 0.1,
                legend_x_shift = 0.05,
                legend_y_shift = 0,
                legend_width = 0.025,
                legend_height = 0.7,
                legend_title_size = 12,
                legend_title = None,
                legend_label_size = 10,
                legend_nlabels = 5,  
                legend_ncolors= 256,
                legend_fmt = None,
                legend_shadow = False,
                legend_font_family = None,
                legend_vertical=True,

                last_as_default = False, 
                interactive = False,
                outline=False, 

                show_axes=False,
                axes_labels = dict(
                    zlabel='z', 
                    xlabel='x', 
                    ylabel='y',
                    labels_off=False,
                    label_size=(0.25, 0.1),
                ),
                show_bounds=False,
                bounds_args = dict(
                    grid='front',
                    location='outer',
                    all_edges=True,
                ),
                args_cbar = {},
                pkargs= dict(), 
                skargs={},
                fkargs={},
                mkargs={},

                showkargs={},
                ):
    """Add several meshes and point clouds to one active PyVista subplot.

    Parameters
    ----------
    meshlist : sequence of pyvista.DataSet or None, default: None
        Surface meshes added to the active renderer. None entries are skipped. Meshes are
        copied before thresholding or rotation.
    pointslist : sequence of DataFrame or (N, 3) arrays, or None, default: None
        Point clouds added to the active subplot. DataFrames must contain x, y, z and any
        groupbys columns. Arrays provide coordinates only. None entries are skipped.
    groupbys : str, sequence of str, or None, default: None
        Point-value column per cloud. None creates one category named group with value "1".
    scalars : str, sequence of str, or None, default: None
        Mesh scalar-array name, or one name per mesh. Continuous mesh scalars are
        thresholded using mvmin and mvmax.
    points_colors : sequence, color, callable colormap, or None, default: None
        Color specification per point cloud. A categorical cloud uses a color or a palette
        in categorical-level order; a continuous cloud requires a callable colormap. For one
        cloud, a typical value is [palette] or [colormap]. None selects built-in defaults.
    opacity : float or sequence of float, default: 0.2
        Mesh opacity in [0, 1], indexed in meshlist order. In the surfaces3d checkbox branch
        this argument is also indexed by category.
    point_psize : float or sequence of float, default: 0.5
        Point display size in pixels. Indexed per point cloud in surfaces3d and per category
        in the categorical surfaces_df branch; continuous surfaces_df uses the first value.
    point_opacity : float or sequence of float, default: 1
        Point opacity in [0, 1]. Indexed per cloud in surfaces3d and per category in
        categorical surfaces_df; continuous surfaces_df uses the first value.
    mesh_psize : float, sequence, or None, default: None
        Retained parameter. Values are expanded with list_iter but are not passed to
        add_mesh in this implementation.
    mesh_style : str, sequence of str, or None, default: None
        Mesh representation passed to add_mesh, such as 'surface', 'wireframe', or 'points',
        indexed per mesh. None uses PyVista's default.
    edge_opacity : float, sequence of float, or None, default: None
        Opacity of mesh polygon edges, passed to add_mesh per mesh. None delegates to the
        plotting backend.
    window_size : sequence of two int or None, default: None
        Render-window width and height in pixels when creating a plotter. Also forwarded to
        screenshot export when return_pl=False.
    image_scale : int or None, default: None
        Image-size multiplier supplied when creating the plotter and as scale for screenshot
        export. It does not change mesh coordinates.
    edge_colors : color or sequence of colors, default: '#AAAAAA'
        Mesh polygon-edge colors, indexed per mesh; used with show_edges.
    font_family : str, default: 'arial'
        Retained theme-font option. The current size/family assignment block is not reached
        after theme_style creates a theme, so this argument has no effect.
    colors : color or sequence of colors, default: '#AAAAAA'
        Base mesh colors, indexed in meshlist order. Mesh scalar coloring is configured
        separately with scalars and clims.
    theme_style : str or None, default: None
        None or "dark" builds a black DarkTheme; "white" builds a white Theme. Other values
        use a DarkTheme with background_color and font_color. This branch replaces the
        supplied theme.
    show_actor : bool, default: False
        For categorical clouds, add one actor and a visibility checkbox per category.
        Continuous coloring disables this branch. This branch indexes opacity and
        point_psize by category.
    actor_size : int, default: 15
        Checkbox side length in pixels when show_actor=True.
    startpos : float, default: 10
        Initial vertical checkbox position in display pixels.
    actor_space : int, default: 5
        Checkbox spacing divisor. The step is actor_size + actor_size // actor_space; use a
        nonzero value.
    color_off : color, default: 'grey'
        Checkbox color for a hidden category.
    color_bg : color, default: 'grey'
        Checkbox background color.
    mvmin : float or None, default: None
        Lower threshold for continuous mesh scalars. If omitted, the first applicable mesh
        supplies the minimum, which is reused for later meshes.
    mvmax : float or None, default: None
        Upper threshold for continuous mesh scalars. If omitted, the first applicable mesh
        supplies the maximum, which is reused for later meshes.
    pvmin : float, sequence of float, or None, default: None
        Lower continuous point-value bound passed to get_minmax. surfaces3d accepts one
        value per cloud; surfaces_df uses one shared value.
    pvmax : float, sequence of float, or None, default: None
        Upper continuous point-value bound passed to get_minmax. surfaces3d accepts one
        value per cloud; surfaces_df uses one shared value.
    ptmin : float, sequence, or None, default: None
        Retained threshold parameter; it does not currently affect the plotted data.
    ppmin : float, sequence of float, or None, default: None
        Lower distribution-based bound supplied as pmin to get_minmax; its interpretation
        follows that helper. Indexed per cloud in surfaces3d.
    ppmax : float, sequence of float, or None, default: None
        Upper distribution-based bound supplied as pmax to get_minmax; its interpretation
        follows that helper. Indexed per cloud in surfaces3d.
    dropmin : bool, default: True
        For continuous points, discard values below the computed lower bound before clipping
        colors. False initially keeps all points; later thresholding can still remove
        points.
    symcolor_bar : bool, default: False
        Write the selected lower display bound into one minimum-valued point before
        thresholding. This implementation does not construct a range symmetric about zero.
    clims : sequence of two-element ranges or None, default: None
        Scalar color limits per mesh, passed as clim. For one mesh, use [(low, high)] to
        keep the pair together when indexed by mesh.
    show_edges : bool or sequence of bool, default: True
        Display mesh polygon edges, indexed per mesh. This does not control subplot borders.
    na_color : color, default: 'grey'
        Color appended for missing categorical values in the surfaces3d palette-building
        branch.
    jupyter_backend : str or None, default: None
        Notebook backend forwarded to PyVista. A non-None value also calls
        set_jupyter_backend and changes the global notebook setting.
    line_smoothing : bool, default: False
        Line-smoothing option passed to the Plotter constructor when pl is None.
    allow_empty_mesh : bool, default: True
        Assigned to pv.global_theme.allow_empty_mesh; this is a global setting, not limited
        to the current plotter.
    shade : bool, default: False
        Retained parameter; unused in this implementation. Mesh shading is controlled by
        lighting and smooth_shading.
    theme : pyvista.Theme or None, default: None
        Retained input. The theme_style branches currently replace this object with a new
        theme before plotter construction.
    xyzscales : sequence of three float or None, default: None
        Display scale factors for x, y, and z passed to pl.set_scale. These change actor
        display, not the input coordinate arrays.
    background_color : color, default: 'black'
        Background color for the automatically built theme, subject to theme_style in
        surfaces3d.
    font_color : color or None, default: None
        Default text color for titles and, depending on theme_style, scalar bars. The theme
        font itself is fixed to white/black in the dark/white branches.
    legend_font_color : color or None, default: None
        Text color supplied to point scalar bars.
    main_title_color : color or None, default: None
        Color of the main title added with pl.add_title.
    font_size : int, default: 15
        Retained general font-size option; the current theme construction does not apply it.
    title_size : int, default: 15
        Retained theme-title-size option; the current theme construction does not apply it.
        Use main_title_size for the explicit title.
    main_title_font : str or None, default: None
        Font for the explicit main title; None uses the plotting default.
    main_title_size : int, default: 15
        Font size passed to pl.add_title.
    cpos : str, camera-position triple, or None, default: None
        Initial camera view passed to pl.camera_position. A triple contains position, focal
        point, and view-up vectors.
    show_scalar_bar : bool, default: True
        Show point scalar bars in branches that forward this flag. Mesh scalar bars are
        controlled through mkargs.
    start_xvfb : bool, default: False
        Call pv.start_xvfb before creating the plotter; relevant to compatible headless
        Linux environments.
    server_proxy_enabled : bool or None, default: None
        If provided, update pv.global_theme.trame.server_proxy_enabled.
    server_proxy_prefix : str or None, default: None
        If provided, update pv.global_theme.trame.server_proxy_prefix.
    notebook : bool or None, default: None
        Notebook display setting passed to the constructor of a new plotter. Ignored for an
        existing pl.
    off_screen : bool or None, default: None
        Off-screen rendering setting passed to the constructor of a new plotter. Ignored for
        an existing pl.
    view_angle : float or None, default: None
        Camera view angle in degrees; relevant to perspective projection.
    azimuth : float or None, default: None
        Value in degrees assigned to pl.camera.azimuth after setting cpos.
    roll : float or None, default: None
        Value in degrees assigned to pl.camera.roll.
    elevation : float or None, default: None
        Value in degrees assigned to pl.camera.elevation after setting azimuth.
    zoom : float or None, default: None
        Camera zoom factor. Values greater than 1 enlarge the view; values between 0 and 1
        shrink it.
    framerate : int, default: 12
        Movie frame rate forwarded to save_mesh.save_movie when exporting MP4.
    painter : bool, default: True
        Painter-order option forwarded to save_graphic for SVG/PDF/EPS/PS/TEX export; it
        does not affect PNG output.
    view_up : sequence of three float or None, default: None
        Static camera up vector passed to pl.set_viewup before cpos and the angular camera
        settings are applied.
    viewup : sequence of three float, default: (0, 0, 0)
        Up vector forwarded to orbital-path generation and movie rendering. Distinct from
        the static-camera view_up argument.
    factor : float, default: 2
        Orbital-path radius factor passed to generate_orbital_path for movie export.
    n_points : int, default: 120
        Number of positions in the generated movie camera path.
    step : float, default: 0.1
        Step argument forwarded to orbit_on_path for movie playback/rendering.
    line_width : float, sequence of float, or None, default: None
        Mesh line or edge width in pixels, indexed per mesh and passed to add_mesh.
    shift : float, default: 0
        Path shift passed to generate_orbital_path for movie export.
    focus : sequence of three float or None, default: None
        Camera focus forwarded to orbit_on_path; None uses the backend default.
    quality : int, default: 8
        Encoding quality forwarded to open_movie for MP4 export; not used by the GIF branch.
    write_frames : bool, default: True
        Whether orbit_on_path writes frames to the opened movie/GIF writer.
    show : bool or None, default: True
        Forwarded to save_mesh unless return_pl=True. In this file, True and None call show
        and save before close; False saves directly and closes.
    raster : bool, default: True
        Rasterization option forwarded to save_graphic. It does not affect PNG screenshots
        and does not guarantee fully vector SVG output.
    title : str or None, default: None
        Main title passed to pl.add_title when nonempty.
    view_isometric : bool, default: False
        Call pl.view_isometric after the other camera settings; this replaces their chosen
        viewing direction.
    title_position : str, default: 'upper_edge'
        Retained parameter; not forwarded to add_title in this implementation.
    points_to_glyph : bool, default: False
        Replace points by sphere glyph geometry. Implemented by surfaces_df only; currently
        unused by surfaces3d.
    point_radius : float, default: 1
        Sphere-glyph radius in coordinate units when surfaces_df uses points_to_glyph=True;
        unused by surfaces3d.
    theta_resolution : int, default: 16
        Sphere-glyph angular resolution in one direction; used only by the surfaces_df glyph
        branch.
    phi_resolution : int, default: 16
        Sphere-glyph angular resolution in the other direction; used only by the surfaces_df
        glyph branch.
    point_styles : str or sequence of str, default: 'points'
        Point representation passed to add_points, typically 'points' or 'points_gaussian'.
        Indexed per cloud in surfaces3d and per category in categorical surfaces_df.
    render_points_as_spheres : bool, default: True
        Sphere-like rendering for point primitives; this is separate from constructing
        sphere geometry with points_to_glyph.
    smooth_shading : bool, default: True
        Smooth-shading option for mesh actors and surfaces_df sphere glyphs. Ordinary point
        actors use pkargs.
    anti_aliasing : bool or None, default: True
        True enables anti-aliasing using aa_type; False disables it; None leaves the current
        setting unchanged.
    lighting : bool, default: False
        Lighting option for mesh actors and surfaces_df sphere glyphs. Ordinary point actors
        use pkargs, whose default disables lighting.
    aa_type : str, default: 'fxaa'
        Anti-aliasing method forwarded to enable_anti_aliasing, such as 'fxaa', 'ssaa', or
        'msaa'.
    multi_samples : int or None, default: None
        Multisample setting forwarded to enable_anti_aliasing.
    all_renderers : bool, default: True
        Whether the anti-aliasing enable operation applies to all renderers.
    rotate_y : float or None, default: None
        Rotation of copied meshes about y, in degrees, after rotate_x.
    rotate_z : float or None, default: None
        Rotation of copied meshes about z, in degrees, after rotate_y.
    rotate_x : float or None, default: None
        Rotation of copied meshes about x, in degrees. Mesh rotations are applied in x, y, z
        order; point clouds are not rotated here.
    vtkz_format : str, default: 'json'
        Format forwarded as format to VTKSZ export, normally 'json' or 'zip'.
    save : str, path-like, or None, default: None
        Output path forwarded to save_mesh. None skips writing a file but does not itself
        disable showing or closing the plotter.
    save_type : str or None, default: None
        Override the output format inferred from the filename extension; forwarded to the
        save_mesh constructor.
    pl : pyvista.Plotter or None, default: None
        Existing plotter. Actors are added to its active subplot; None creates a new
        plotter.
    return_pl : bool, default: False
        Return the plotter immediately after adding actors and camera settings, without
        calling save_mesh. Use True when composing several subplots.
    Light : pyvista.Light, dict, list, or None, default: None
        Additional light or lights. Each dictionary is expanded as pv.Light(**settings).
    legend_x_pos : float or None, default: 0.9
        Horizontal scalar-bar position in normalized viewport coordinates.
    legend_y_pos : float or None, default: 0.1
        Vertical scalar-bar position in normalized viewport coordinates.
    legend_x_shift : float, default: 0.05
        Horizontal scalar-bar offset per cloud in surfaces3d. Currently unused by
        surfaces_df.
    legend_y_shift : float, default: 0
        Vertical scalar-bar offset per cloud in surfaces3d. Currently unused by surfaces_df.
    legend_width : float, default: 0.025
        Scalar-bar width in normalized viewport coordinates.
    legend_height : float, default: 0.7
        Scalar-bar height in normalized viewport coordinates.
    legend_title_size : int, default: 12
        Scalar-bar title font size.
    legend_title : str, sequence, or None, default: None
        Point scalar-bar title. surfaces3d indexes titles per cloud and accepts False to
        suppress a title; None uses the group name.
    legend_label_size : int or sequence of int, default: 10
        Scalar-bar label font size. Indexed per cloud in surfaces3d; surfaces_df uses the
        first value.
    legend_nlabels : int, default: 5
        Number of labels for continuous point scalar bars. Categorical branches set the
        number of labels to zero.
    legend_ncolors : int, default: 256
        Number of scalar-bar colors for continuous data. surfaces3d uses the number of
        categories for categorical data.
    legend_fmt : str or None, default: None
        Numeric scalar-bar label format passed as fmt, for example a printf-style format.
    legend_shadow : bool, default: False
        Whether scalar-bar text has a shadow.
    legend_font_family : str or None, default: None
        Font family passed to point scalar bars.
    legend_vertical : bool, default: True
        True draws vertical scalar bars; False draws horizontal ones.
    last_as_default : bool, default: False
        Passed to list_iter for scalar/per-item options; controls reuse of the last supplied
        value when an item is missing.
    interactive : bool, default: False
        Interactivity of scalar bars. This is not the main Plotter.show interactive option,
        which belongs in showkargs.
    outline : bool, default: False
        Scalar-bar outline setting. The continuous surfaces3d branch explicitly turns it
        off.
    show_axes : bool, default: False
        Add and show orientation axes with axes_labels.
    axes_labels : dict, default: dict(zlabel='z', xlabel='x', ylabel='y', labels_off=False, label_size=(0.25, 0.1))
        Arguments passed to pl.add_axes, including axis-label text and label visibility.
    show_bounds : bool, default: False
        Add a coordinate-bounds display using pl.show_bounds.
    bounds_args : dict, default: dict(grid='front', location='outer', all_edges=True)
        Arguments passed to pl.show_bounds when show_bounds=True.
    args_cbar : dict, default: {}
        Extra scalar-bar settings merged into the base scalar-bar dictionary. Data-specific
        branches subsequently set titles and color/label counts.
    pkargs : dict, default: dict(lighting=False)
        Extra arguments for ordinary pl.add_points calls; the default sets lighting=False.
        Do not duplicate explicitly supplied keywords.
    skargs : dict, default: {}
        Extra arguments for save_mesh.__call__ when return_pl=False. Do not duplicate
        keywords already forwarded by this function.
    fkargs : dict, default: {}
        Extra arguments for a newly created pv.Plotter, for example shape or border=False.
        Ignored when pl is supplied.
    mkargs : dict, default: {}
        Extra arguments for mesh pl.add_mesh calls. Do not duplicate the explicitly supplied
        mesh-style keywords.
    showkargs : dict, default: {}
        Additional Plotter.show arguments passed through save_mesh, for example auto_close
        or interactive. Used when show is not False.

    Returns
    -------
    pl : pyvista.Plotter
        Existing or newly created plotter. return_pl=True bypasses display and
        export; otherwise save_mesh controls its final open/closed state.

    Notes
    -----
    Per-cloud settings follow pointslist order; per-mesh settings follow meshlist order.
    Categorical palettes follow pandas category order when available. Rendering may sort
    category labels, so category labels should be mutually sortable. The checkbox branch
    additionally indexes opacity and point_psize by category.
    Continuous point colors require a callable colormap and a nonconstant retained scalar
    range; this implementation explicitly calls that colormap during preprocessing.
    get_minmax supplies clipping bounds and a subsequent threshold selects the displayed
    range.
    The theme_style branch replaces a supplied theme. The later font-size/family assignment
    condition is not reached, so font_size, title_size, and font_family are currently
    inactive here. shade, title_position, and the sphere-glyph arguments are also inactive;
    use surfaces_df for sphere glyphs.
    With an existing pl, select the subplot before calling and use return_pl=True while
    assembling a multipanel figure. Constructor settings do not recreate that plotter.
    Global empty-mesh and notebook/proxy settings are still updated.

    Examples
    --------
    >>> pl = surfaces3d(meshlist=[mesh], pointslist=[points_df],
    ...                 groupbys=["celltype"], pl=pl, return_pl=True)
    """
    import pyvista as pv

    # if theme_name:
    #     pv.set_plot_theme(theme_name)
        # pv.set_plot_theme('default')
        # pv.global_theme

    # if theme is None:
    #     my_theme = pv.themes.Theme()
    #     # my_theme.background.color = background_color
    #     # my_theme.set_font(size=40, color='red')
    #     my_theme.font.color = 'red'
    #     my_theme.font.size = 40
    #     print(my_theme)
    #     pv.theme = my_theme
     
    if theme_style in [None,'dark']:
        font_color = font_color or 'white'
        main_title_color = main_title_color or font_color
        legend_font_color = legend_font_color or 'white'

        theme = pv.themes.DarkTheme()
        # theme.lighting = True
        # theme.show_edges = True
        # theme.edge_color = 'white'
        # theme.color = 'black'
        theme.background = 'black'
        theme.font.color = 'white'

    elif theme_style == 'white':
        font_color = font_color or 'black'
        main_title_color = main_title_color or font_color
        legend_font_color = legend_font_color or font_color

        theme = pv.themes.Theme()
        theme.background = 'white'
        theme.font.color = 'black'

    else:
        font_color = font_color or 'white'
        main_title_color = main_title_color or font_color
        legend_font_color = legend_font_color or font_color

        theme = pv.themes.DarkTheme()
        # theme.lighting = True
        # theme.show_edges = True
        # theme.edge_color = 'white'
        # theme.color = 'black'
        theme.background = background_color
        theme.font.color = font_color


    if theme is None:
        theme.font.size  = font_size
        theme.font.title_size = title_size
        theme.font.family = font_family

    pv.global_theme.allow_empty_mesh = allow_empty_mesh

    if server_proxy_enabled is not None:
        pv.global_theme.trame.server_proxy_enabled = server_proxy_enabled
    if not server_proxy_prefix is None:
        pv.global_theme.trame.server_proxy_prefix = server_proxy_prefix
    if start_xvfb:
        pv.start_xvfb() 
    if not jupyter_backend is None:
        pv.set_jupyter_backend(jupyter_backend)
    
    if pl is None:
        pl = pv.Plotter(notebook=notebook, window_size=window_size, 
                        off_screen=off_screen, image_scale =image_scale,
                        theme=theme, line_smoothing=line_smoothing, **fkargs)

    if not meshlist is None:
        edge_colors = list_iter(edge_colors, last_as_default=last_as_default)
        colors = list_iter(colors, last_as_default=last_as_default)
        opacity= list_iter(opacity, last_as_default=last_as_default)
        edge_opacity = list_iter(edge_opacity, last_as_default=last_as_default)
        show_edges = list_iter(show_edges, last_as_default=last_as_default)
        line_width = list_iter(line_width, last_as_default=last_as_default)
        scalars = list_iter(scalars, last_as_default=last_as_default)
        mesh_psize = list_iter(mesh_psize, last_as_default=last_as_default)
        clims = list_iter(clims, last_as_default=last_as_default)
        mesh_style = list_iter(mesh_style, last_as_default=last_as_default)
        mesh_lighting = list_iter(mesh_lighting, last_as_default=last_as_default)
        
        for j, mesh in enumerate(meshlist):
            if mesh is None:
                continue
            mesh = mesh.copy()
            groupby = scalars[j]
            if not groupby is None:
                groups = np.asarray(mesh[groupby])
                gdtype = vartype(groups)
                if gdtype == 'continuous':
                    try:
                        mesh.set_active_scalars(groupby)
                        if mvmin is None:
                            mvmin = mesh[groupby].min()
                        if mvmax is None:
                            mvmax = mesh[groupby].max()
                        mesh = mesh.threshold([mvmin, mvmax])
                    except:
                        pass

            if not rotate_x is None:
                mesh = mesh.rotate_x(rotate_x, inplace=False)
            if not rotate_y is None:
                mesh = mesh.rotate_y(rotate_y, inplace=False)
            if not rotate_z is None:
                mesh = mesh.rotate_z(rotate_z, inplace=False)
            pl.add_mesh(mesh, show_edges=show_edges[j], 
                        opacity=opacity[j], 
                        render_points_as_spheres=render_points_as_spheres,
                        edge_opacity=edge_opacity[j], 
                        scalars=scalars[j],
                        line_width=line_width[j],
                        edge_color=edge_colors[j], 
                        clim=clims[j],
                        color=colors[j],
                        lighting=mesh_lighting[j],
                        smooth_shading=smooth_shading,
                        style=mesh_style[j],
                        **mkargs)

    groupbys = list_iter(groupbys, last_as_default=last_as_default)
    points_colors = list_iter(points_colors, last_as_default=last_as_default)
    legend_label_size = list_iter(legend_label_size, last_as_default=last_as_default)
    point_opacity = list_iter(point_opacity, last_as_default=last_as_default)
    point_styles = list_iter(point_styles, last_as_default=last_as_default)
    point_psize =  list_iter(point_psize, last_as_default=last_as_default)
    point_lighting = list_iter(point_lighting, last_as_default=last_as_default)
    pvmins = list_iter(pvmin, last_as_default=last_as_default)
    pvmaxs = list_iter(pvmax, last_as_default=last_as_default)
    ppmins = list_iter(ppmin, last_as_default=last_as_default)
    ppmaxs = list_iter(ppmax, last_as_default=last_as_default)
    ptmins = list_iter(ptmin, last_as_default=last_as_default)
    legend_title = list_iter(legend_title, last_as_default=last_as_default)

    sclbar = dict(height=legend_height, width=legend_width, 
                  vertical = legend_vertical,
                    n_labels=legend_nlabels,
                    interactive=interactive,
                    outline =outline,
                    color = legend_font_color,
                    position_x=legend_x_pos, 
                    position_y=legend_y_pos,
                    fmt=legend_fmt,
                    shadow=legend_shadow,
                    font_family=legend_font_family,
                    title_font_size=legend_title_size,
                    label_font_size=legend_label_size[0])
    sclbar.update(args_cbar)
    if not pointslist is None:
        for idx, points in enumerate(pointslist):
            if points is None:
                continue
            if isinstance(points, (list, np.ndarray)):
                points =pd.DataFrame(points, columns=list('xyz'))
            assert isinstance(points, pd.DataFrame)

            groupby = groupbys[idx]
            igroup_color = points_colors[idx]
            ipkargs = pkargs.copy()
            if groupby is None:
                points = points.copy()
                groupby = 'group'
                points[groupby] = '1'

            assert groupby in points.columns.tolist()
            groups = points[groupby]

            gdtype = vartype(groups)

            if gdtype == 'discrete':
                pvdt = pv.PolyData(points[['x', 'y', 'z']].values.astype(np.float64))
                pvdt[groupby] = groups
                try:
                    # Order = groups.cat.remove_unused_categories().cat.categories
                    Order = groups.cat.categories #keep color order
                except:
                    Order= pd.unique(groups)
                uni_order = pd.unique(groups)
    
                if igroup_color is None:
                    color_list = color_palette(len(Order))
                else:
                    color_list = [igroup_color]*len(Order) if isinstance(igroup_color, str) else igroup_color

                # TODO sort legend bar
                nOrder =  Order[~pd.isna(Order)]
                na_nm =  groups[pd.isna(groups)].unique()

                Order = list(nOrder) + list(na_nm)
                na_colors = [na_color] * len(na_nm)
                color_list = list(color_list)+list(na_colors)
                assert len(color_list)>= len(Order)

                coldict = dict(zip(Order, color_list))
                Order = pd.Series(Order).sort_values()
                color_list = [ coldict[i] for i in Order if i in uni_order ]
                # TODO check
                # Order = pd.Series([i for i in Order if i in uni_order])
                # color_list = [ coldict[i] for i in Order ]
    
                # my_cmap = ListedColormap(color_list)
                my_cmap = color_list
                # point_opacity = [ None if point_opacity[i] is None else float(point_opacity[i]) for i in Order.index ]
                # point_psize = [ None if point_psize[i] is None else  float(point_psize[i]) for i in Order.index ]
                
                categories = True
                sclbar["n_labels"] = 0 #len(Order)
                sclbar["n_colors"] = len(Order)
                # sclbar["height"] =  0.85

            elif gdtype == 'continuous':
                pvmin  = pvmins[idx]
                pvmax  = pvmaxs[idx]
                ppmin  = ppmins[idx]
                ppmax  = ppmaxs[idx]
                ptmin  = ptmins[idx]

 
                groups = np.asarray(groups)
                imin, imax = get_minmax(groups, vmin=pvmin, vmax=pvmax, pmin=ppmin, pmax=ppmax)
                ipkargs.setdefault('clim', (imin, imax))
                if dropmin:
                    kidx = groups >= imin
                else:
                    kidx = np.ones(len(groups), dtype=bool)
                groups = np.clip(groups, imin, imax)

                pvdt = pv.PolyData(points[['x', 'y', 'z']].values.astype(np.float64)[kidx, :])
                pvdt[groupby] = groups[kidx]
                pvdt.set_active_scalars(groupby)

                # print(groups.shape, imin, imax, kidx.sum())
        
                # pvdt = pv.PolyData(points[['x', 'y', 'z']].values.astype(np.float64))
                # pvdt[groupby] = groups
                # pvdt.set_active_scalars(groupby)
                # pvdt[groupby] = clipdata(groups,  
                #                     vmin=pvmin, vmax=pvmax,
                #                     pmin=ppmin, pmax=ppmax, 
                #                     clips=None, tmin = ptmin, dropmin=False)
                avmin = pvdt[groupby].min()
                avmax = pvdt[groupby].max()

                for imin in [pvmin, ]:
                    if imin is not None:
                        avmin = min(imin, avmin)
                for imax in [pvmax, ]:
                    if imax is not None:
                        avmax = min(imax, avmax)
                if symcolor_bar:
                    minidx = np.where(pvdt[groupby] == pvdt[groupby].min())[0][:1]
                    pvdt[groupby][ minidx ] = avmin
                pvdt = pvdt.threshold([avmin, avmax])

                show_actor = False
                categories = False
                sclbar["n_labels"] = legend_nlabels 
                sclbar["n_colors"] = legend_ncolors
                sclbar["outline"] = False
                my_cmap = cmap1a if igroup_color is None else igroup_color
                scale_rgba = (pvdt[groupby] - pvdt[groupby].min() )/(pvdt[groupby].max()- pvdt[groupby].min())
                scale_rgba = my_cmap(scale_rgba)

            if (not show_actor):
                isclbar = sclbar.copy()
                if not isclbar['position_x'] is None:
                    isclbar['position_x'] = isclbar['position_x'] + legend_x_shift*idx
                if not isclbar['position_y'] is None:
                    isclbar['position_y'] = isclbar['position_y'] + legend_y_shift*idx
                isclbar['label_font_size'] = legend_label_size[idx]
                ilegend_title = legend_title[idx]
                if ilegend_title is False:
                    ilegend_title = None
                elif ilegend_title is None:
                    ilegend_title = None if groupby is None else str(groupby)
                else:
                    ilegend_title = str(ilegend_title)
                isclbar['title'] = ilegend_title
                actor = pl.add_points(pvdt, 
                            style=point_styles[idx],
                            cmap=my_cmap,
                            color=None, 
                            opacity = point_opacity[idx],
                            point_size = point_psize[idx],
                            # label =  None if groupby is None else str(groupby),
                            scalars = groupby,
                            scalar_bar_args=isclbar,
                            
                            # rgba=True,
                            # emissive=True,
                            render_points_as_spheres=render_points_as_spheres, 
                            show_scalar_bar=show_scalar_bar,
                            categories=categories,
                            name = str(idx),
                            lighting = point_lighting[idx],
                            # annotations=annotations,
                            **ipkargs)

            elif show_actor and (gdtype == 'discrete'):
                Startpos = startpos
                # opacity= [1,0.25]
                # point_psize = [3,1]

                for i, iorder in enumerate(Order):
                    fidx = groups == iorder
                    ipos = pvdt.points[fidx]
                    icolor = color_list[i]
                    ipos = pv.PolyData(ipos)
                    actor = pl.add_points(ipos, 
                                        # style='points', 
                                        style=point_styles[idx],
                                        # style='points_gaussian', 
                                        cmap=my_cmap,
                                        label = str(iorder), #groupby,
                                        name = str(iorder),
                                        color=icolor, 
                                        render_points_as_spheres=render_points_as_spheres,
                                        # scalars = scalars,
                                        # scalar_bar_args=sclbar,
                                        # label=str(i),
                                        categories=categories,
                                        # opacity = point_opacity[idx],
                                        # point_size=point_psize[idx],
                                        opacity=opacity[i],
                                        point_size=point_psize[i],
                                        lighting = point_lighting[idx],
                                        # rgba=True,
                                        **pkargs)

                    callback = SetVisibilityCallback(actor)
                    xyspace = actor_size + (actor_size // actor_space)
                    pl.add_checkbox_button_widget(
                        callback,
                        value=True,
                        position=(5.0 + idx*xyspace, Startpos),
                        size=actor_size,
                        border_size=1,
                        color_on=icolor,
                        color_off=color_off,
                        background_color=color_bg,
                    )
                    # pl.add_actor(actor, reset_camera=False, name=str(i), 
                    #              culling=False, pickable=True, 
                    #              render=True, remove_existing_actor=True)
                    Startpos = Startpos +xyspace
                # pl.add_legend(loc='center right', bcolor=None, size=[0.1,0.8], face='circle')
            # pl.update_scalar_bar_range([vmin, vmax], name=None)

    if anti_aliasing is None:
        pass
    elif anti_aliasing is True:
        pl.enable_anti_aliasing(aa_type, multi_samples=multi_samples, all_renderers=all_renderers)
    elif anti_aliasing is False:
        pl.disable_anti_aliasing()

    if not Light is None:
        if isinstance(Light, list):
            for iLight in Light:
                if isinstance(iLight, pv.Light):
                    pl.add_light(iLight)
                elif isinstance(iLight, dict):
                    pl.add_light(pv.Light(**iLight))
                else:
                    raise ValueError('Light must be a pv.Light object or a dictionary')
        elif isinstance(Light, pv.Light):
            pl.add_light(Light)
        elif isinstance(Light, dict):
            pl.add_light(pv.Light(**Light))
        else:
            raise ValueError('Light must be a pv.Light object, a list of pv.Light objects or a dictionary')
    
    if show_axes:
        pl.add_axes(**axes_labels)
        # pl.show_axes()

    if show_bounds:
        actor = pl.show_bounds(**bounds_args)
    
    # add_legend(pl)
    if not xyzscales is None:
        pl.set_scale(xscale=xyzscales[0], yscale=xyzscales[1], zscale=xyzscales[2])
    if not view_up is None:
        pl.set_viewup(view_up)
    if cpos is not None:
        pl.camera_position = cpos
    if azimuth is not None:
        pl.camera.azimuth = azimuth
    if elevation is not None:
        pl.camera.elevation = elevation
    if view_angle is not None:
        pl.camera.view_angle = view_angle
    if roll is not None:
        pl.camera.roll = roll
    if zoom is not None:
        pl.camera.zoom(zoom)

    if view_isometric:
        pl.view_isometric()

    if title:
        actor = pl.add_title(title, #position=title_position, 
                             font=main_title_font, color=main_title_color, font_size=main_title_size)

    # pv.theme.restore_defaults()
    if return_pl:
        return pl
    return save_mesh(pl, save_type=save_type)(save, show=show,
                    framerate=framerate, painter=painter,#view_up = view_up, 
                    viewup=viewup, factor=factor, n_points=n_points, 
                    step=step, shift=shift, focus=focus,
                    scale=image_scale, window_size=window_size,
                    quality=quality, write_frames=write_frames,
                    raster=raster, format=vtkz_format,
                     jupyter_backend=jupyter_backend, showkargs=showkargs, **skargs)

def surfaces_df(
                points = None, 
                meshlist=None, 
                groupby= None,
                scalars = None,
                points_colors = None,
                opacity=0.2,
                point_psize=0.5, 
                point_opacity=1,
                mesh_psize=None,
                edge_opacity=None,
                window_size=None,
                image_scale=None,
                edge_colors='#AAAAAA', 
                font_family = 'arial', #courier
                colors='#AAAAAA',
                mesh_style=None,
                show_actor = False, actor_size = 15, 
                startpos = 10, actor_space=5,
                color_off='grey', color_bg='grey',
                mvmin = None, mvmax = None,
                pvmin = None, pvmax = None, ptmin=None,
                ppmin=None, ppmax=None,
                clims = None,
                show_edges=True, 
                na_color='grey',

                jupyter_backend = None,
                line_smoothing=False,
                allow_empty_mesh=True,
                shade=False,
                # theme_name=None, 
                theme=None, 
                xyzscales=None,
                background_color='black', 
                font_color='white',
                legend_font_color = "white",
                main_title_color='white', 
                font_size =15, title_size =15,
                main_title_font =None,main_title_size=15,
                cpos=None, 
                show_scalar_bar=True,    
                start_xvfb = False, 
                server_proxy_enabled=None, 
                server_proxy_prefix=None,
                notebook=None, 
                off_screen=None,

                view_angle = None, azimuth=None, roll=None, elevation=None, zoom=None,
                framerate=12, painter=True, view_up = None, viewup=(0, 0, 0),
                factor=2, n_points=120, step=0.1, line_width=None,
                shift=0, focus=None, quality=8, write_frames=True,
                show=True, raster=True,
                title=None, view_isometric=False,
                title_position='upper_edge', 

                points_to_glyph = False,
                point_radius =1,
                theta_resolution=16,
                phi_resolution=16,

                point_styles='points',
                render_points_as_spheres=True, 
                smooth_shading=True,
                anti_aliasing=True, 
                
                point_lighting=True, 
                mesh_lighting=False,
            
                aa_type ='fxaa', multi_samples=None, all_renderers=True,
                rotate_y = None, rotate_z = None, rotate_x = None, vtkz_format='json',
                save=None, save_type=None, 
                pl = None,
                return_pl = False,

                Light = None,
                legend_x_pos = 0.9,
                legend_y_pos = 0.1,
                legend_x_shift = 0.05,
                legend_y_shift = 0,
                legend_width = 0.025,
                legend_height = 0.7,
                legend_title_size = 12,
                legend_title = None,
                legend_label_size = 10,
                legend_nlabels = 5,  
                legend_ncolors= 256,
                legend_fmt = None,
                legend_shadow = False,
                legend_font_family = None,
                legend_vertical=True,

                last_as_default = False, 
                interactive = False,
                outline=False, 

                show_axes=False,
                axes_labels = dict(
                    zlabel='z', 
                    xlabel='x', 
                    ylabel='y',
                    labels_off=False,
                    label_size=(0.25, 0.1),
                ),
                show_bounds=False,
                bounds_args = dict(
                    grid='front',
                    location='outer',
                    all_edges=True,
                ),

                args_cbar = {},
                pkargs= dict(lighting=False), 
                skargs={},
                fkargs={},
                mkargs={},
                showkargs={},
                ):
    """Add one labeled point DataFrame and optional meshes to a PyVista subplot.

    Parameters
    ----------
    points : pandas.DataFrame, (N, 3) array-like, or None, default: None
        One point cloud. DataFrames require x, y, z columns plus groupby when specified;
        coordinate-only lists/arrays are converted to a DataFrame.
    meshlist : sequence of pyvista.DataSet or None, default: None
        Surface meshes added to the active renderer. None entries are skipped. Meshes are
        copied before thresholding or rotation.
    groupby : str or None, default: None
        DataFrame column supplying categorical labels or continuous values. None creates one
        category named group with value "1".
    scalars : str, sequence of str, or None, default: None
        Mesh scalar-array name, or one name per mesh. Continuous mesh scalars are
        thresholded using mvmin and mvmax.
    points_colors : color, sequence of colors, colormap, or None, default: None
        Categorical palette in pandas category order, or first-appearance order for
        noncategorical labels. Continuous data use a callable colormap or a Matplotlib
        colormap name; the first expanded entry is used. None chooses a default
        palette/colormap.
    opacity : float or sequence of float, default: 0.2
        Mesh opacity in [0, 1], indexed in meshlist order. In the surfaces3d checkbox branch
        this argument is also indexed by category.
    point_psize : float or sequence of float, default: 0.5
        Point size in pixels, indexed per categorical level or using the first value for
        continuous data. Sphere glyphs instead use point_radius.
    point_opacity : float or sequence of float, default: 1
        Opacity in [0, 1], indexed by category or using the first value for continuous
        points. Categorical glyph alpha is replaced by this opacity; continuous glyph alpha
        is multiplied by it.
    mesh_psize : float, sequence, or None, default: None
        Retained parameter. Values are expanded with list_iter but are not passed to
        add_mesh in this implementation.
    edge_opacity : float, sequence of float, or None, default: None
        Opacity of mesh polygon edges, passed to add_mesh per mesh. None delegates to the
        plotting backend.
    window_size : sequence of two int or None, default: None
        Render-window width and height in pixels when creating a plotter. Also forwarded to
        screenshot export when return_pl=False.
    image_scale : int or None, default: None
        Image-size multiplier supplied when creating the plotter and as scale for screenshot
        export. It does not change mesh coordinates.
    edge_colors : color or sequence of colors, default: '#AAAAAA'
        Mesh polygon-edge colors, indexed per mesh; used with show_edges.
    font_family : str, default: 'arial'
        Font family for the automatically constructed theme. See the theme notes for this
        function.
    colors : color or sequence of colors, default: '#AAAAAA'
        Base mesh colors, indexed in meshlist order. Mesh scalar coloring is configured
        separately with scalars and clims.
    mesh_style : str, sequence of str, or None, default: None
        Mesh representation passed to add_mesh, such as 'surface', 'wireframe', or 'points',
        indexed per mesh. None uses PyVista's default.
    show_actor : bool, default: False
        Retained parameter; currently unused by surfaces_df.
    actor_size : int, default: 15
        Retained parameter; currently unused by surfaces_df.
    startpos : float, default: 10
        Retained parameter; currently unused by surfaces_df.
    actor_space : int, default: 5
        Retained parameter; currently unused by surfaces_df.
    color_off : color, default: 'grey'
        Retained parameter; currently unused by surfaces_df.
    color_bg : color, default: 'grey'
        Retained parameter; currently unused by surfaces_df.
    mvmin : float or None, default: None
        Lower threshold for continuous mesh scalars. If omitted, the first applicable mesh
        supplies the minimum, which is reused for later meshes.
    mvmax : float or None, default: None
        Upper threshold for continuous mesh scalars. If omitted, the first applicable mesh
        supplies the maximum, which is reused for later meshes.
    pvmin : float or None, default: None
        Lower continuous-value bound supplied to get_minmax. Values are clipped to the
        resulting range without dropping point rows.
    pvmax : float or None, default: None
        Upper continuous-value bound supplied to get_minmax. Values are clipped to the
        resulting range without dropping point rows.
    ptmin : float, sequence, or None, default: None
        Retained threshold parameter; it does not currently affect the plotted data.
    ppmin : float or None, default: None
        Lower distribution-based bound forwarded as pmin to get_minmax; interpreted by that
        helper.
    ppmax : float or None, default: None
        Upper distribution-based bound forwarded as pmax to get_minmax; interpreted by that
        helper.
    clims : sequence of two-element ranges or None, default: None
        Scalar color limits per mesh, passed as clim. For one mesh, use [(low, high)] to
        keep the pair together when indexed by mesh.
    show_edges : bool or sequence of bool, default: True
        Display mesh polygon edges, indexed per mesh. This does not control subplot borders.
    na_color : color, default: 'grey'
        Retained parameter; currently unused by surfaces_df.
    jupyter_backend : str or None, default: None
        Notebook backend forwarded to PyVista. A non-None value also calls
        set_jupyter_backend and changes the global notebook setting.
    line_smoothing : bool, default: False
        Line-smoothing option passed to the Plotter constructor when pl is None.
    allow_empty_mesh : bool, default: True
        Assigned to pv.global_theme.allow_empty_mesh; this is a global setting, not limited
        to the current plotter.
    shade : bool, default: False
        Retained parameter; unused in this implementation. Mesh shading is controlled by
        lighting and smooth_shading.
    theme : pyvista.Theme or None, default: None
        Theme passed to a newly created plotter. None constructs a DarkTheme using the
        background and font arguments; an explicit theme bypasses those assignments.
    xyzscales : sequence of three float or None, default: None
        Display scale factors for x, y, and z passed to pl.set_scale. These change actor
        display, not the input coordinate arrays.
    background_color : color, default: 'black'
        Background color for the automatically built theme, subject to theme_style in
        surfaces3d.
    font_color : color or None, default: 'white'
        Theme font color. The theme-handling branch determines whether this value is
        applied.
    legend_font_color : color or None, default: 'white'
        Text color supplied to point scalar bars.
    main_title_color : color or None, default: 'white'
        Color of the main title added with pl.add_title.
    font_size : int, default: 15
        General theme font size. See Notes for whether theme construction applies this
        value.
    title_size : int, default: 15
        Theme title font size. The explicit main title uses main_title_size instead.
    main_title_font : str or None, default: None
        Font for the explicit main title; None uses the plotting default.
    main_title_size : int, default: 15
        Font size passed to pl.add_title.
    cpos : str, camera-position triple, or None, default: None
        Initial camera view passed to pl.camera_position. A triple contains position, focal
        point, and view-up vectors.
    show_scalar_bar : bool, default: True
        Forwarded for continuous points and sphere glyphs. Ordinary categorical points are
        added as separately colored actors without this flag.
    start_xvfb : bool, default: False
        Call pv.start_xvfb before creating the plotter; relevant to compatible headless
        Linux environments.
    server_proxy_enabled : bool or None, default: None
        If provided, update pv.global_theme.trame.server_proxy_enabled.
    server_proxy_prefix : str or None, default: None
        If provided, update pv.global_theme.trame.server_proxy_prefix.
    notebook : bool or None, default: None
        Notebook display setting passed to the constructor of a new plotter. Ignored for an
        existing pl.
    off_screen : bool or None, default: None
        Off-screen rendering setting passed to the constructor of a new plotter. Ignored for
        an existing pl.
    view_angle : float or None, default: None
        Camera view angle in degrees; relevant to perspective projection.
    azimuth : float or None, default: None
        Value in degrees assigned to pl.camera.azimuth after setting cpos.
    roll : float or None, default: None
        Value in degrees assigned to pl.camera.roll.
    elevation : float or None, default: None
        Value in degrees assigned to pl.camera.elevation after setting azimuth.
    zoom : float or None, default: None
        Camera zoom factor. Values greater than 1 enlarge the view; values between 0 and 1
        shrink it.
    framerate : int, default: 12
        Movie frame rate forwarded to save_mesh.save_movie when exporting MP4.
    painter : bool, default: True
        Painter-order option forwarded to save_graphic for SVG/PDF/EPS/PS/TEX export; it
        does not affect PNG output.
    view_up : sequence of three float or None, default: None
        Static camera up vector passed to pl.set_viewup before cpos and the angular camera
        settings are applied.
    viewup : sequence of three float, default: (0, 0, 0)
        Up vector forwarded to orbital-path generation and movie rendering. Distinct from
        the static-camera view_up argument.
    factor : float, default: 2
        Orbital-path radius factor passed to generate_orbital_path for movie export.
    n_points : int, default: 120
        Number of positions in the generated movie camera path.
    step : float, default: 0.1
        Step argument forwarded to orbit_on_path for movie playback/rendering.
    line_width : float, sequence of float, or None, default: None
        Mesh line or edge width in pixels, indexed per mesh and passed to add_mesh.
    shift : float, default: 0
        Path shift passed to generate_orbital_path for movie export.
    focus : sequence of three float or None, default: None
        Camera focus forwarded to orbit_on_path; None uses the backend default.
    quality : int, default: 8
        Encoding quality forwarded to open_movie for MP4 export; not used by the GIF branch.
    write_frames : bool, default: True
        Whether orbit_on_path writes frames to the opened movie/GIF writer.
    show : bool or None, default: True
        Forwarded to save_mesh unless return_pl=True. In this file, True and None call show
        and save before close; False saves directly and closes.
    raster : bool, default: True
        Rasterization option forwarded to save_graphic. It does not affect PNG screenshots
        and does not guarantee fully vector SVG output.
    title : str or None, default: None
        Main title passed to pl.add_title when nonempty.
    view_isometric : bool, default: False
        Call pl.view_isometric after the other camera settings; this replaces their chosen
        viewing direction.
    title_position : str, default: 'upper_edge'
        Retained parameter; not forwarded to add_title in this implementation.
    points_to_glyph : bool, default: False
        Replace points by sphere glyph geometry. Implemented by surfaces_df only; currently
        unused by surfaces3d.
    point_radius : float, default: 1
        Sphere-glyph radius in coordinate units when surfaces_df uses points_to_glyph=True;
        unused by surfaces3d.
    theta_resolution : int, default: 16
        Sphere-glyph angular resolution in one direction; used only by the surfaces_df glyph
        branch.
    phi_resolution : int, default: 16
        Sphere-glyph angular resolution in the other direction; used only by the surfaces_df
        glyph branch.
    point_styles : str or sequence of str, default: 'points'
        Point representation passed to add_points, typically 'points' or 'points_gaussian'.
        Indexed per cloud in surfaces3d and per category in categorical surfaces_df.
    render_points_as_spheres : bool, default: True
        Sphere-like rendering for point primitives; this is separate from constructing
        sphere geometry with points_to_glyph.
    smooth_shading : bool, default: True
        Smooth-shading option for mesh actors and surfaces_df sphere glyphs. Ordinary point
        actors use pkargs.
    anti_aliasing : bool or None, default: True
        True enables anti-aliasing using aa_type; False disables it; None leaves the current
        setting unchanged.
    lighting : bool, default: False
        Lighting option for mesh actors and surfaces_df sphere glyphs. Ordinary point actors
        use pkargs, whose default disables lighting.
    aa_type : str, default: 'fxaa'
        Anti-aliasing method forwarded to enable_anti_aliasing, such as 'fxaa', 'ssaa', or
        'msaa'.
    multi_samples : int or None, default: None
        Multisample setting forwarded to enable_anti_aliasing.
    all_renderers : bool, default: True
        Whether the anti-aliasing enable operation applies to all renderers.
    rotate_y : float or None, default: None
        Rotation of copied meshes about y, in degrees, after rotate_x.
    rotate_z : float or None, default: None
        Rotation of copied meshes about z, in degrees, after rotate_y.
    rotate_x : float or None, default: None
        Rotation of copied meshes about x, in degrees. Mesh rotations are applied in x, y, z
        order; point clouds are not rotated here.
    vtkz_format : str, default: 'json'
        Format forwarded as format to VTKSZ export, normally 'json' or 'zip'.
    save : str, path-like, or None, default: None
        Output path forwarded to save_mesh. None skips writing a file but does not itself
        disable showing or closing the plotter.
    save_type : str or None, default: None
        Override the output format inferred from the filename extension; forwarded to the
        save_mesh constructor.
    pl : pyvista.Plotter or None, default: None
        Existing plotter. Actors are added to its active subplot; None creates a new
        plotter.
    return_pl : bool, default: False
        Return the plotter immediately after adding actors and camera settings, without
        calling save_mesh. Use True when composing several subplots.
    Light : pyvista.Light, dict, list, or None, default: None
        Additional light or lights. Each dictionary is expanded as pv.Light(**settings).
    legend_x_pos : float or None, default: 0.9
        Horizontal scalar-bar position in normalized viewport coordinates.
    legend_y_pos : float or None, default: 0.1
        Vertical scalar-bar position in normalized viewport coordinates.
    legend_x_shift : float, default: 0.05
        Retained parameter; currently unused by surfaces_df.
    legend_y_shift : float, default: 0
        Retained parameter; currently unused by surfaces_df.
    legend_width : float, default: 0.025
        Scalar-bar width in normalized viewport coordinates.
    legend_height : float, default: 0.7
        Scalar-bar height in normalized viewport coordinates.
    legend_title_size : int, default: 12
        Scalar-bar title font size.
    legend_title : str or None, default: None
        Continuous scalar-bar title; None uses the groupby column name.
    legend_label_size : int or sequence of int, default: 10
        Scalar-bar label font size. Indexed per cloud in surfaces3d; surfaces_df uses the
        first value.
    legend_nlabels : int, default: 5
        Number of labels for continuous point scalar bars. Categorical branches set the
        number of labels to zero.
    legend_ncolors : int, default: 256
        Number of scalar-bar colors for continuous data. surfaces3d uses the number of
        categories for categorical data.
    legend_fmt : str or None, default: None
        Numeric scalar-bar label format passed as fmt, for example a printf-style format.
    legend_shadow : bool, default: False
        Whether scalar-bar text has a shadow.
    legend_font_family : str or None, default: None
        Font family passed to point scalar bars.
    legend_vertical : bool, default: True
        True draws vertical scalar bars; False draws horizontal ones.
    last_as_default : bool, default: False
        Passed to list_iter for scalar/per-item options; controls reuse of the last supplied
        value when an item is missing.
    interactive : bool, default: False
        Interactivity of scalar bars. This is not the main Plotter.show interactive option,
        which belongs in showkargs.
    outline : bool, default: False
        Scalar-bar outline setting. The continuous surfaces3d branch explicitly turns it
        off.
    show_axes : bool, default: False
        Add and show orientation axes with axes_labels.
    axes_labels : dict, default: dict(zlabel='z', xlabel='x', ylabel='y', labels_off=False, label_size=(0.25, 0.1))
        Arguments passed to pl.add_axes, including axis-label text and label visibility.
    show_bounds : bool, default: False
        Add a coordinate-bounds display using pl.show_bounds.
    bounds_args : dict, default: dict(grid='front', location='outer', all_edges=True)
        Arguments passed to pl.show_bounds when show_bounds=True.
    args_cbar : dict, default: {}
        Extra scalar-bar settings merged into the base scalar-bar dictionary. Data-specific
        branches subsequently set titles and color/label counts.
    pkargs : dict, default: dict(lighting=False)
        Extra arguments for ordinary pl.add_points calls; the default sets lighting=False.
        Do not duplicate explicitly supplied keywords.
    skargs : dict, default: {}
        Extra arguments for save_mesh.__call__ when return_pl=False. Do not duplicate
        keywords already forwarded by this function.
    fkargs : dict, default: {}
        Extra arguments for a newly created pv.Plotter, for example shape or border=False.
        Ignored when pl is supplied.
    mkargs : dict, default: {}
        Extra arguments for mesh pl.add_mesh calls. Do not duplicate the explicitly supplied
        mesh-style keywords.
    showkargs : dict, default: {}
        Additional Plotter.show arguments passed through save_mesh, for example auto_close
        or interactive. Used when show is not False.

    Returns
    -------
    pl : pyvista.Plotter
        Existing or newly created plotter. return_pl=True returns before any
        display/export; otherwise the result follows save_mesh behavior.

    Notes
    -----
    Categorical points normally use one actor per observed category. Color, opacity, size,
    and style lists follow pandas categorical order, including unused levels, or first
    appearance for noncategorical labels. Continuous points use the first expanded
    style/color/opacity entry, clip colors with get_minmax, and retain point rows.
    points_to_glyph=True constructs sphere geometry with a physical radius. It uses explicit
    RGBA, enables depth peeling, and bypasses the ordinary pkargs point-actor path. The
    categorical glyph path does not supply a separate missing-value color; prepare missing
    labels before using it.
    show_actor and its checkbox-layout arguments, na_color, ptmin, shade, title_position,
    legend_x_shift, and legend_y_shift are retained but unused here. mesh_psize is expanded
    but not forwarded. A supplied theme is retained; otherwise background/font settings
    build a DarkTheme.
    Global empty-mesh and optional notebook/proxy settings are updated even when pl is
    supplied. Use return_pl=True to keep composing a figure without showing or exporting it.

    Examples
    --------
    >>> pl = surfaces_df(points=points_df, groupby="celltype",
    ...                  meshlist=[mesh], pl=pl, return_pl=True)
    """
    import pyvista as pv


    if theme is None:
        theme = pv.themes.DarkTheme()
        # theme.lighting = True
        # theme.show_edges = True
        # theme.edge_color = 'white'
        # theme.color = 'black'
        theme.background = background_color
        theme.font.color = font_color
        theme.font.size  = font_size
        theme.font.title_size = title_size
        theme.font.family = font_family
    pv.global_theme.allow_empty_mesh = allow_empty_mesh

    if server_proxy_enabled is not None:
        pv.global_theme.trame.server_proxy_enabled = server_proxy_enabled
    if not server_proxy_prefix is None:
        pv.global_theme.trame.server_proxy_prefix = server_proxy_prefix
    if start_xvfb:
        pv.start_xvfb() 
    if not jupyter_backend is None:
        pv.set_jupyter_backend(jupyter_backend)
    
    if pl is None:
        pl = pv.Plotter(
                notebook=notebook, window_size=window_size, 
                off_screen=off_screen, image_scale =image_scale,
                theme=theme, line_smoothing=line_smoothing, **fkargs
                )

    if not meshlist is None:
        edge_colors = list_iter(edge_colors, last_as_default=last_as_default)
        colors = list_iter(colors, last_as_default=last_as_default)
        opacity= list_iter(opacity, last_as_default=last_as_default)
        edge_opacity = list_iter(edge_opacity, last_as_default=last_as_default)
        show_edges = list_iter(show_edges, last_as_default=last_as_default)
        line_width = list_iter(line_width, last_as_default=last_as_default)
        scalars = list_iter(scalars, last_as_default=last_as_default)
        mesh_psize = list_iter(mesh_psize, last_as_default=last_as_default)
        clims = list_iter(clims, last_as_default=last_as_default)
        mesh_style = list_iter(mesh_style, last_as_default=last_as_default)
        mesh_lighting = list_iter(mesh_lighting, last_as_default=last_as_default)

        for j, mesh in enumerate(meshlist):
            if mesh is None:
                continue
            mesh = mesh.copy()
            igroup = scalars[j]
            if not igroup is None:
                groups = np.asarray(mesh[igroup])
                gdtype = vartype(groups)
                if gdtype == 'continuous':
                    try:
                        mesh.set_active_scalars(igroup)
                        if mvmin is None:
                            mvmin = mesh[igroup].min()
                        if mvmax is None:
                            mvmax = mesh[igroup].max()
                        mesh = mesh.threshold([mvmin, mvmax])
                    except:
                        pass

            if not rotate_x is None:
                mesh = mesh.rotate_x(rotate_x, inplace=False)
            if not rotate_y is None:
                mesh = mesh.rotate_y(rotate_y, inplace=False)
            if not rotate_z is None:
                mesh = mesh.rotate_z(rotate_z, inplace=False)
            pl.add_mesh(mesh, 
                        show_edges=show_edges[j], 
                        opacity=opacity[j], 
                        render_points_as_spheres=render_points_as_spheres,
                        edge_opacity=edge_opacity[j], 
                        scalars=scalars[j],
                        line_width=line_width[j],
                        edge_color=edge_colors[j], 
                        clim=clims[j],
                        color=colors[j],
                        lighting=mesh_lighting[j],
                        smooth_shading=smooth_shading,
                        style=mesh_style[j],
                        **mkargs
                        )

    points_colors = list_iter(points_colors, last_as_default=last_as_default)
    legend_label_size = list_iter(legend_label_size, last_as_default=last_as_default)
    point_opacity = list_iter(point_opacity, last_as_default=last_as_default)
    point_psize =  list_iter(point_psize, last_as_default=last_as_default)
    point_styles = list_iter(point_styles, last_as_default=last_as_default)
    point_lighting = list_iter(point_lighting, last_as_default=last_as_default)

    sclbar = dict(height=legend_height, width=legend_width, 
                  vertical = legend_vertical,
                    n_labels=legend_nlabels,
                    interactive=interactive,
                    outline =outline,
                    color = legend_font_color,
                    position_x=legend_x_pos, 
                    position_y=legend_y_pos,
                    fmt=legend_fmt,
                    shadow=legend_shadow,
                    font_family=legend_font_family,
                    title_font_size=legend_title_size,
                    label_font_size=legend_label_size[0])
    sclbar.update(args_cbar)

    if not points is None:
        if isinstance(points, (list, np.ndarray)):
            points =pd.DataFrame(points, columns=list('xyz'))
        assert isinstance(points, pd.DataFrame)
        if groupby is None:
            points = points.copy()
            groupby = 'group'
            points[groupby] = '1'

        assert groupby in points.columns.tolist()
        groups = points[groupby]
        gdtype = vartype(groups)

        pvdt = pv.PolyData(points[['x', 'y', 'z']].values.astype(np.float64))
        pvdt[groupby] = groups

        if gdtype == 'continuous':
            from matplotlib import colormaps
            from matplotlib.colors import Normalize
            imin, imax = get_minmax(np.asarray(groups, float), vmin=pvmin, vmax=pvmax,
                                    pmin=ppmin, pmax=ppmax)
            pvdt[groupby] = np.clip(np.asarray(groups, float), imin, imax)
            point_cmap = cmap1a if points_colors[0] is None else points_colors[0]
            if isinstance(point_cmap, str):
                point_cmap = colormaps[point_cmap]
        elif points_colors[0] is None:
            order = getattr(getattr(groups, 'dtype', None), 'categories', pd.unique(groups))
            points_colors = list_iter(color_palette(len(order)), last_as_default=last_as_default)

        if points_to_glyph:
            if gdtype == 'discrete':
                try:
                    # Order = groups.cat.remove_unused_categories().cat.categories
                    Order = groups.cat.categories #keep color order
                except:
                    Order= pd.unique(groups)
                n_cat = len(Order)
                Order_index = pd.Index(Order).get_indexer(groups)
                rgba   = []
                for i in range(n_cat):
                    icol = points_colors[i]
                    try:
                        icol = list(to_rgba(points_colors[i]))
                    except:
                        raise ValueError('Invalid hex color: {}'.format(icol))
                    icol[-1] = float(point_opacity[i])
                    rgba.append(icol)
                rgba = np.array(rgba)
            else:
                rgba = point_cmap(Normalize(imin, imax)(np.ma.masked_invalid(pvdt[groupby])))
                rgba[:, 3] *= point_opacity[0]
                Order_index = np.arange(pvdt.n_points)

            pvdt["point_rgba"] = rgba[Order_index]
            pvdt.set_active_scalars("point_rgba")

            sphere = pv.Sphere(radius=point_radius, theta_resolution=theta_resolution, 
                               phi_resolution=phi_resolution)

            glyphs = pvdt.glyph(
                scale=False,
                orient=False, 
                geom=sphere,
                color_mode="scalar",
            )
            pl.enable_depth_peeling()
            pl.add_mesh(
                glyphs,
                scalars="point_rgba",
                rgba=True,
                lighting=point_lighting[0],
                smooth_shading=smooth_shading,
                show_scalar_bar=show_scalar_bar,
            )
        else:
            if gdtype == 'discrete':
                try:
                    # Order = groups.cat.remove_unused_categories().cat.categories
                    Order = groups.cat.categories #keep color order
                except:
                    Order= pd.unique(groups)

                for idx, iorder in enumerate(Order):
                    fidx = groups == iorder
                    if np.sum(fidx) == 0:
                        continue
                    ipos = pvdt.points[fidx]
                    ipos = pv.PolyData(ipos)

                    sclbar["n_labels"] = 0 
                    sclbar["n_colors"] = 1

                    actor = pl.add_points(ipos, 
                                        label = str(iorder), #groupby,
                                        name = str(iorder),
                                        render_points_as_spheres=render_points_as_spheres,
                                        # scalars = scalars,
                                        # scalar_bar_args=sclbar,
                                        categories=True,
                                        style=point_styles[idx],
                                        color=points_colors[idx], 
                                        opacity=point_opacity[idx],
                                        point_size=point_psize[idx],
                                        lighting=point_lighting[idx],
                                        # rgba=True,
                                        **pkargs
                        )
            elif gdtype == 'continuous':
                sclbar.update(n_labels=legend_nlabels, n_colors=legend_ncolors,
                              title=str(groupby) if legend_title is None else legend_title)
                pl.add_points(pvdt, scalars=groupby, cmap=point_cmap, clim=(imin, imax),
                              style=point_styles[0], opacity=point_opacity[0],
                              point_size=point_psize[0], scalar_bar_args=sclbar,
                              lighting = point_lighting[0],
                              show_scalar_bar=show_scalar_bar,
                              render_points_as_spheres=render_points_as_spheres, **pkargs)

    if anti_aliasing is None:
        pass
    elif anti_aliasing is True:
        pl.enable_anti_aliasing(aa_type, multi_samples=multi_samples, all_renderers=all_renderers)
    elif anti_aliasing is False:
        pl.disable_anti_aliasing()

    if not Light is None:
        if isinstance(Light, list):
            for iLight in Light:
                if isinstance(iLight, pv.Light):
                    pl.add_light(iLight)
                elif isinstance(iLight, dict):
                    pl.add_light(pv.Light(**iLight))
                else:
                    raise ValueError('Light must be a pv.Light object or a dictionary')
        elif isinstance(Light, pv.Light):
            pl.add_light(Light)
        elif isinstance(Light, dict):
            pl.add_light(pv.Light(**Light))
        else:
            raise ValueError('Light must be a pv.Light object, a list of pv.Light objects or a dictionary')

    if show_axes:
        pl.add_axes(**axes_labels)
        # pl.show_axes()

    if show_bounds:
        actor = pl.show_bounds(**bounds_args)

    # add_legend(pl)
    if not xyzscales is None:
        pl.set_scale(xscale=xyzscales[0], yscale=xyzscales[1], zscale=xyzscales[2])
    if not view_up is None:
        pl.set_viewup(view_up)
    if cpos is not None:
        pl.camera_position = cpos
    if azimuth is not None:
        pl.camera.azimuth = azimuth
    if elevation is not None:
        pl.camera.elevation = elevation
    if view_angle is not None:
        pl.camera.view_angle = view_angle
    if roll is not None:
        pl.camera.roll = roll
    if zoom is not None:
        pl.camera.zoom(zoom)
    if view_isometric:
        pl.view_isometric()
    if show_axes:
        pl.show_axes()
    if title:
        actor = pl.add_title(title, #position=title_position, 
                             font=main_title_font, color=main_title_color, font_size=main_title_size)
    if return_pl:
        return pl
    return save_mesh(pl, save_type=save_type)(save, show=show,
                    framerate=framerate, painter=painter, #view_up = view_up, 
                    viewup=viewup, factor=factor, n_points=n_points, 
                    step=step, shift=shift, focus=focus,
                    scale=image_scale, window_size=window_size,
                    quality=quality, write_frames=write_frames,
                    raster=raster, format=vtkz_format,
                     jupyter_backend=jupyter_backend, showkargs=showkargs, **skargs)



def mesh_plane_section(mesh, origin, normal="x", clean_tolerance=0.0):
    """Return (filled cap, contour); process all loops together to retain holes."""
    import pyvista as pv
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator

    mesh = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()
    mesh = mesh if mesh.is_all_triangles else mesh.triangulate()
    contour = mesh.slice(normal=normal, origin=origin).clean(
        tolerance=clean_tolerance, absolute=True, lines_to_points=False)
    if not contour.n_lines:
        return pv.PolyData(), contour
    lines = contour.lines
    if lines.size != 3 * contour.n_lines or np.any(lines[::3] != 2):
        raise ValueError("Expected two-endpoint intersection segments")
    edges = lines.reshape(-1, 3)[:, 1:]
    edges = edges[edges[:, 0] != edges[:, 1]]
    _, keep = np.unique(np.sort(edges, axis=1), axis=0, return_index=True)
    edges = edges[np.sort(keep)]
    if not len(edges):
        return pv.PolyData(), contour
    degree = np.bincount(edges.ravel(), minlength=contour.n_points)
    if np.any(degree[degree > 0] != 2):
        raise ValueError("Open or branching contour; cannot fill this section")
    contour = pv.PolyData(np.asarray(contour.points, float),
                         lines=np.column_stack((np.full(len(edges), 2), edges)))
    triangulator = vtkContourTriangulator()
    triangulator.SetInputData(contour)
    triangulator.Update()
    cap = pv.wrap(triangulator.GetOutput()).copy()
    if triangulator.GetTriangulationError() or not cap.GetNumberOfPolys():
        raise ValueError("Section triangulation failed")
    return cap, contour


def surface_slice(
    pl, points=None, point_value=None, points_colors=None,
    meshlist=None, meshnames=None, mesh_colors=None,
    origin=None, normal="x", thickness=20.0, mode="cutaway", invert=True,
    point_mode="both", project_points=True, point_psize=3.0, point_opacity=1.0,
    mesh_opacity=1.0, cap_opacity=1.0, show_edges=True,
    edge_opacity=1.0, edge_colors=None, show_contours=True,
    interactive=True, assign_to_axis=None, interaction_event="end",

    point_lighting=True, 
    mesh_lighting=False,
    show_scalar_bar=False,
    render_points_as_spheres=True,
    render=True,
    smooth_shading=True,

    clean_tolerance=0.0, margs=None, seed =0,
    cap_colors=None, contour_colors=None, contour_width=2.0, edge_width=None,
    widget_color="white", widget_plane_color=None,
    widget_plane_opacity=0.025, widget_line_width=1.0,
    widget_outline_color=None, widget_outline_opacity=0.0,
    widget_outline_width=None, widget_factor=1.05,
    show_widget_outline=True, show_widget_normal=True,
    widget_normal_color=None, widget_normal_opacity=1.0,
    widget_normal_width=None, widget_normal_length_ratio=0.3,
    widget_handle_size=None,
):
    """Section point clouds and meshes with a shared, optionally interactive plane.

    Parameters
    ----------
    pl : pyvista.Plotter
        Existing plotter; add the section to its currently active subplot.
    points : array-like of shape (N, 3) or None, default: None
        Finite point coordinates. Points and meshes must use the same coordinate system and
        units.
    point_value : one-dimensional array-like or None, default: None
        One category or numeric value per point, classified by vartype. None assigns every
        point to the "points" category. Missing categorical values raise ValueError.
    points_colors : dict, sequence of colors, callable colormap, or None, default: None
        Categorical colors: a label-to-color dictionary or a list in pandas category order
        (otherwise first-appearance order). None generates random category colors.
        Continuous data require a callable colormap and a nonzero value range.
    meshlist : sequence of pyvista.DataSet or None, default: None
        Surface meshes to section. Each is converted to a surface if needed, triangulated,
        and cleaned; each entry must contain polygons.
    meshnames : sequence or None, default: None
        Names used in state["mesh_names"] and error reporting, with one entry per mesh. None
        uses integer indices.
    mesh_colors : sequence of colors or None, default: None
        One explicit solid color per mesh, in meshlist order. Existing mesh scalar arrays
        are not used for coloring. None generates colors using seed.
    origin : sequence of three float or None, default: None
        A point on the slicing plane. None uses the point bounding-box center, or the
        combined mesh bounding-box center when no points are present. With an interactive
        widget, VTK constrains the origin to its placement box; the section and returned
        state use the effective widget origin.
    normal : str or sequence of three float, default: 'x'
        Plane normal: x/y/z, a signed axis such as -x, or a finite nonzero vector. Vectors
        are normalized before point selection and mesh cutting.
    thickness : float, default: 20.0
        Positive full thickness of the selected point slab in coordinate units. For
        point_mode="slice" or "both", retain abs((points-origin) @ normal) <= thickness/2.
        This thickness applies only to the section points, not the 3D context. Mesh caps
        have zero thickness.
    mode : {"section", "context", "cutaway"}, default: 'cutaway'
        section displays selected points, caps, and optional contours. context also displays
        complete meshes. cutaway also displays the retained mesh halves. With
        point_mode="both", context also shows the full 3D cloud and cutaway shows its
        retained half-space; section shows only the slab points.
    invert : bool, default: True
        True retains signed distance <= 0; False retains >= 0. Applies to cutaway meshes and
        point_mode="clip" or the cutaway 3D cloud in "both", not to symmetric point slabs.
    point_mode : {"slice", "clip", "both"}, default: 'slice'
        slice selects a slab of thickness; clip selects one entire half-space according to
        invert. both keeps the slice display and adds a separate 3D point actor matching
        mode (full cloud for context, retained half-space for cutaway). Original slice and
        clip behavior is unchanged.
    project_points : bool, default: True
        Project selected slab points onto the plane when point_mode="slice" or "both".
        The 3D context cloud always retains its original coordinates. Ignored for
        point_mode="clip".
    point_psize : float, default: 3.0
        Point display diameter in pixels. At very small sizes (for example 3 pixels),
        spherical points may look blocky because the round outline is undersampled.
    point_opacity : float, default: 1.0
        Point actor opacity in [0, 1]. Per-color alpha is retained separately by to_rgba or
        the continuous colormap; this actor opacity also affects the rendered transparency.
    mesh_opacity : float or sequence of float, default: 1.0
        Opacity in [0, 1] for complete/context or clipped mesh actors; indexed per mesh.
    cap_opacity : float or sequence of float, default: 1.0
        Opacity in [0, 1] of the filled mesh-plane intersections, indexed per mesh.
    show_edges : bool or sequence of bool, default: True
        Draw triangulation edges on mesh surfaces and filled caps. This is separate from
        section contours, widget borders, and subplot borders.
    edge_opacity : float or sequence of float, default: 1.0
        Opacity of mesh/cap triangle edges and section contour actors, indexed per mesh.
    edge_colors : color, sequence of colors, or None, default: None
        Mesh/cap triangle-edge colors. A None entry uses that mesh color; contours also
        inherit this color unless contour_colors is supplied.
    show_contours : bool, default: True
        Display the line intersections between the plane and meshes independently of
        show_edges.
    interactive : bool, default: True
        Create a draggable plane widget. False still computes the section and provides
        state["update"] for programmatic updates.
    assign_to_axis : {"x", "y", "z", 0, 1, 2} or None, default: None
        Lock the plane normal to an axis. The supplied normal must already be parallel to
        that axis. Axis locking also hides the normal arrows/center handle in this
        implementation.
    interaction_event : str, default: 'end'
        Plane-widget event used to update geometry, such as "end" or "always". "end"
        recomputes after a drag; "always" updates continuously.
    lighting : bool, default: False
        Default lighting setting for section actors; margs may override it.
    show_scalar_bar : bool, default: False
        Default scalar-bar flag passed to actor creation. Direct point RGBA and solid mesh
        colors do not define a continuous scalar scale.
    render_points_as_spheres : bool, default: True
        Default sphere-like rendering of point primitives; margs may override it. This
        uses GPU point sprites rather than sphere mesh geometry. With lighting=False,
        spheres appear as flat round markers; lighting=True adds spherical shading.
    render : bool, default: True
        Render after the initial section update. Widget events and state["update"] render by
        default; use state["update"](normal, origin, render=False) to suppress a later
        render.
    smooth_shading : bool, default: True
        Default smooth-shading flag for section actors; margs may override it.
    clean_tolerance : float, default: 0.0
        Nonnegative absolute point-merging distance in coordinate units, used for mesh and
        contour cleaning. Zero merges exactly coincident vertices.
    margs : dict or None, default: None
        Shared add_mesh options. These override base rendering defaults, while role-specific
        arguments such as color, opacity, and point_size override margs.
    seed : int or None, default: 0
        Random seed for automatically generated category and mesh palettes.
    cap_colors : color, sequence of colors, or None, default: None
        Colors of the actual filled mesh sections; a None entry inherits the corresponding
        mesh color. This does not color the widget guide plane.
    contour_colors : color, sequence of colors, or None, default: None
        Section-outline colors; a None entry inherits the corresponding edge color.
    contour_width : float or sequence of float, default: 2.0
        Section-outline width in display pixels, indexed per mesh.
    edge_width : float, sequence of float, or None, default: None
        Triangle-edge width in display pixels for mesh and cap actors. None uses
        margs["line_width"] when present, otherwise 1.0.
    widget_color : color, default: 'white'
        Plane-widget border color and fallback color for its guide face, outer box, and
        normal handles.
    widget_plane_color : color or None, default: None
        Color of the translucent widget guide face, including its selected state. None
        inherits widget_color; use cap_colors for the filled tissue sections.
    widget_plane_opacity : float, default: 0.025
        Opacity in [0, 1] of the widget guide face, including its selected state.
    widget_line_width : float, default: 1.0
        Widget plane-border width in pixels and fallback width for the outer box and normal-
        arrow shafts.
    widget_outline_color : color or None, default: None
        Color of the widget outer bounding box; None inherits widget_color.
    widget_outline_opacity : float, default: 0.0
        Opacity in [0, 1] of the widget outer box. The default zero makes the box invisible
        even when show_widget_outline=True.
    widget_outline_width : float or None, default: None
        Outer-box line width in pixels; None inherits widget_line_width.
    widget_factor : float or sequence of three floats, default: 1.05
        Widget placement-box scale relative to the combined point/mesh bounds. A scalar
        scales all axes equally and preserves the original scalar placement behavior.
        A list, tuple, or 1D array (sx, sy, sz) scales the x, y, z side lengths separately
        around the bounding-box center. Values must be positive and finite; 1.0 keeps a
        side length, values above 1 enlarge it, and values below 1 shrink it. These are
        data-coordinate axes, not screen axes. Point/mesh coordinates and data bounds are
        unchanged. A smaller box restricts the interactive plane-origin range; any clamped
        origin is also used to update the section geometry.
    show_widget_outline : bool, default: True
        False hides the widget outer box regardless of widget_outline_opacity; True uses the
        configured opacity.
    show_widget_normal : bool, default: True
        Display normal arrows and the center handle when assign_to_axis is None. False also
        disables normal rotation through the widget.
    widget_normal_color : color or None, default: None
        Normal-arrow/handle color, including the selected state; None inherits widget_color.
    widget_normal_opacity : float, default: 1.0
        Normal-arrow/handle opacity in [0, 1] when they are enabled.
    widget_normal_width : float or None, default: None
        Normal-arrow shaft width in pixels; None inherits widget_line_width.
    widget_normal_length_ratio : float, default: 0.3
        Normal-line length relative to the widget bounding-box diagonal, passed to VTK
        SetDiagonalRatio.
    widget_handle_size : float or None, default: None
        Relative handle-size setting passed to VTK SetHandleSize, affecting arrowheads and
        the center sphere. None retains the VTK default.

    Returns
    -------
    pl : pyvista.Plotter
        The supplied plotter.
    state : dict
        Mutable section state. Empty when no drawable inputs are supplied.
        Keys include widget, mesh_names, bounds, normal, origin, point_indices,
        point_categories, point_section, caps, contours, clipped_meshes, errors,
        actors, and update. bounds contains xmin, xmax, ymin, ymax, zmin, zmax.
        point_indices indexes original input rows; nonempty point_section carries
        RGB and source_index arrays. Geometry lists follow mesh_names order;
        clipped_meshes is populated only for mode="cutaway".
        point_context and point_context_indices contain the additional 3D points in
        "both" mode, with an actor at actors["point_context"]. They are empty otherwise.
        Both point datasets carry RGB and source_index arrays when nonempty. Duplicate
        points at the same position are omitted from point_context when already drawn
        by point_section; projected slab points may also appear at their original 3D
        positions. point_indices continues to index point_section only.
        state["update"](normal, origin, render=True) moves/recomputes the section.

    Notes
    -----
    For unit normal n and plane origin o, signed distance is d = (points - o) @ n. The point
    selection is a finite-thickness slab or a half-space; filled mesh caps are exact plane
    sections. Closed contour loops are triangulated together to retain holes. Failed mesh
    filling clears that cap, records the reason in errors, and leaves its current contour
    available.
    Mesh/edge/cap styling arguments expanded with list_iter reuse their last value when
    needed. mesh_colors and meshnames instead require one entry per input mesh. Six-digit
    categorical colors receive the alpha returned by to_rgba (normally 1); eight-digit/RGBA
    colors retain their own alpha. point_opacity is applied to the point actor rather than
    replacing color alpha. Continuous values are normalized by their raw minimum and
    maximum; constant or invalid ranges are not handled specially.
    Widget options take effect only when interactive=True. Keep the widget enabled and a
    live Python rendering session to recompute slices. Exported standalone HTML does not
    execute this Python update callback. The guide-plane face, its border, its outer
    bounding box, and the filled mesh cap have separate styling parameters.
    Input arrays and meshes are not modified. Actor user-matrix transforms, such as those
    applied by link_views_angles, do not automatically transform the slicing coordinates or
    plane widget.

    Examples
    --------
    >>> pl, state = surface_slice(pl, points=points, point_value=labels,
    ...                           meshlist=[mesh], normal="x", thickness=20.0,
    ...                           cap_colors="orange", contour_colors="white",
    ...                           contour_width=2.0, interactive=True)
    >>> state["update"]("x", (10.0, 0.0, 0.0))
    """
    import warnings
    import pyvista as pv

    if mode not in ("section", "context", "cutaway") or point_mode not in ("slice", "clip", "both"):
        raise ValueError("Invalid mode or point_mode")
    if (not np.isfinite([thickness, clean_tolerance, point_opacity]).all()
            or thickness <= 0 or clean_tolerance < 0 or not 0 <= point_opacity <= 1):
        raise ValueError("Check thickness, clean_tolerance and point_opacity")
    meshlist = [] if meshlist is None else list(meshlist)
    if points is None and not meshlist:
        return pl, {}
    meshnames = list(range(len(meshlist))) if meshnames is None else meshnames
    if mesh_colors is None:
        mesh_colors = random_colors(len(meshlist), seed=seed)
    if len(meshnames) != len(meshlist) or len(mesh_colors) != len(meshlist):
        raise ValueError("meshlist, meshnames and mesh_colors must have equal lengths")

    mesh_opacity = list_iter(mesh_opacity, last_as_default=True)
    cap_opacity = list_iter(cap_opacity, last_as_default=True)
    edge_opacity = list_iter(edge_opacity, last_as_default=True)
    edge_colors = list_iter(edge_colors, last_as_default=True)
    show_edges = list_iter(show_edges, last_as_default=True)
    cap_colors = list_iter(cap_colors, last_as_default=True)
    contour_colors = list_iter(contour_colors, last_as_default=True)
    contour_width = list_iter(contour_width, last_as_default=True)
    mesh_lighting = list_iter(mesh_lighting, last_as_default=True)
    edge_width = list_iter((margs or {}).get("line_width", 1.0) if edge_width is None
                           else edge_width, last_as_default=True)

    cloud = None
    if points is not None and len(points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite (N, 3) array")
        if point_value is None:
            point_value = np.full(len(points), "points")
        if len(point_value) != len(points):
            raise ValueError("point_value must have one value per point")
        gdtype = vartype(point_value)
        if gdtype == 'discrete':
            codes, labels = pd.factorize(point_value, sort=False)
            # AnnData color lists follow category order, not first appearance.
            color_order = getattr(getattr(point_value, "dtype", None), "categories", labels)
            point_value = np.asarray(point_value)
            if points_colors is None:
                points_colors = random_colors(len(color_order), seed=seed)
            category_colors = (points_colors if isinstance(points_colors, dict)
                               else dict(zip(color_order, points_colors)))
            if np.any(codes < 0):
                raise ValueError("point_value contains missing categories; fill or remove them first")
            palette = np.array([ to_rgba(category_colors[k]) for k in labels] ) 
            assert min(codes) >= 0
            rgb = palette[codes]
        else:
            point_value = np.asarray(point_value)
            color_scale = (point_value-point_value.min()) / (point_value.max() - point_value.min())
            ## points_colors :matplotlib.colors.ListedColormap('jet')
            rgb = points_colors(color_scale)
        cloud = pv.PolyData(points.copy())
        cloud["RGB"] = rgb

    surfaces = [
        (m if isinstance(m, pv.PolyData) else m.extract_surface()).triangulate().clean(
            tolerance=clean_tolerance, absolute=True) for m in meshlist
    ]
    for mesh in surfaces:
        if not mesh.GetNumberOfPolys():
            raise ValueError("Each mesh must contain surface polygons")
        mesh.points = np.asarray(mesh.points, float)

    boxes = ([cloud.bounds] if cloud is not None else []) + [m.bounds for m in surfaces]
    if not len(boxes):
        return pl, {}
    
    boxes = np.asarray(boxes).reshape(-1, 3, 2)
    bounds = np.column_stack((boxes[:, :, 0].min(0), boxes[:, :, 1].max(0))).ravel()
    if origin is None:
        origin = (points.min(0) + points.max(0)) / 2 if cloud is not None else bounds.reshape(3, 2).mean(1)
    state = dict(pl=pl, widget=None, mesh_names=meshnames, bounds=bounds, errors={})
    renderer, prefix = pl.renderer, f"section_{id(state)}"
    axis = "xyz".index(assign_to_axis.lower()) if isinstance(assign_to_axis, str) else assign_to_axis
    if axis is not None and axis not in (0, 1, 2):
        raise ValueError("assign_to_axis must be x/y/z, 0/1/2, or None")

    def warp_actor(data, name, **options):
        kargs = dict(show_scalar_bar=show_scalar_bar,
                     render_points_as_spheres=render_points_as_spheres,
                     smooth_shading=smooth_shading,
                     reset_camera=False, render=False)
        kargs.update(margs or {})
        kargs.update(options)
        obj = pl.add_mesh(data, name=f"{prefix}_{name}", **kargs)
        if "color" in options:
            obj.mapper.scalar_visibility = False
        return obj

    def display(obj, data):
        obj.mapper.dataset = data
        obj.SetVisibility(bool(data.n_cells))

    point_actor = context_point_actor = None
    if cloud is not None:
        point_actor = warp_actor(cloud, "points",
                                 lighting=point_lighting, scalars="RGB", rgb=True, 
                                 #opacity=1.0, 
                                 opacity=point_opacity,
                                 style="points", point_size=point_psize)
        point_actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        point_actor.mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-4)
        if point_mode == "both" and mode != "section":
            context_point_actor = warp_actor(
                cloud, "point_context", 
                lighting=point_lighting, scalars="RGB", rgb=True, opacity=point_opacity,
                style="points", point_size=point_psize)
    cap_actors, line_actors, mesh_actors = [], [], []
    for i, (mesh, color) in enumerate(zip(surfaces, mesh_colors)):
        edge_color = color if edge_colors[i] is None else edge_colors[i]
        cap_color = color if cap_colors[i] is None else cap_colors[i]
        contour_color = edge_color if contour_colors[i] is None else contour_colors[i]
        edges = dict(show_edges=show_edges[i], edge_color=edge_color,
                     edge_opacity=edge_opacity[i], line_width=edge_width[i])
        cap_actors.append(warp_actor(mesh, f"cap_{i}", color=cap_color,
                                    lighting = mesh_lighting[i],
                                    opacity=cap_opacity[i], **edges))
        if show_contours:
            line = warp_actor(mesh, f"contour_{i}", color=contour_color,
                            lighting = mesh_lighting[i],
                              opacity=edge_opacity[i], line_width=contour_width[i], show_edges=False)
            line.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            line.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
            line_actors.append(line)
        if mode != "section":
            mesh_actors.append(warp_actor(mesh, f"mesh_{i}", color=color, 
                                            lighting = mesh_lighting[i],
                                           opacity=mesh_opacity[i], **edges))

    def update_pl(normal, origin, render=True):
        if isinstance(normal, str):
            normal = np.eye(3)["xyz".index(normal[-1].lower())] * (-1 if normal[0] == "-" else 1)
        n, o = np.asarray(normal, float), np.asarray(origin, float)
        if n.shape != (3,) or o.shape != (3,) or not np.isfinite([n, o]).all() or np.linalg.norm(n) == 0:
            raise ValueError("Use finite 3-vectors and a nonzero normal")
        n = n / np.linalg.norm(n)
        if axis is not None and not np.isclose(abs(n[axis]), 1):
            raise ValueError("normal must be parallel to assign_to_axis")
        if state["widget"] is not None:
            state["widget"].SetNormal(n)
            # VTK clamps this argument in place; never pass the caller's array directly.
            state["widget"].SetOrigin(o.copy())
            o = np.asarray(state["widget"].GetOrigin(), dtype=float)
        ids, selected = np.empty(0, dtype=int), pv.PolyData()
        context_ids, context_points = np.empty(0, dtype=int), pv.PolyData()
        if cloud is not None:
            d = (points - o) @ n
            slab = point_mode in ("slice", "both")
            mask = np.abs(d) <= thickness / 2 if slab else (d <= 0 if invert else d >= 0)
            ids = np.flatnonzero(mask)
            xyz = points[ids].copy()
            if project_points and slab:
                xyz -= d[ids, None] * n
            selected = pv.PolyData(xyz)
            if len(ids):
                selected["RGB"], selected["source_index"] = rgb[ids], ids
            display(point_actor, selected)
            if context_point_actor is not None:
                context_mask = ((d <= 0 if invert else d >= 0) if mode == "cutaway"
                                else np.ones(len(points), dtype=bool))
                # Keep the 3D cloud unprojected; do not draw coincident points twice.
                context_mask &= (d != 0) if project_points else ~mask
                context_ids = np.flatnonzero(context_mask)
                context_points = pv.PolyData(points[context_ids].copy())
                if len(context_ids):
                    context_points["RGB"] = rgb[context_ids]
                    context_points["source_index"] = context_ids
                display(context_point_actor, context_points)
        caps, contours, clipped, errors = [], [], [], {}
        for i, mesh in enumerate(surfaces):
            try:
                cap, contour = mesh_plane_section(mesh, o, n, clean_tolerance)
            except ValueError as error:
                cap, contour = pv.PolyData(), mesh.slice(normal=n, origin=o)
                errors[meshnames[i]] = str(error)
                if state["errors"].get(meshnames[i]) != str(error):
                    warnings.warn(f"{meshnames[i]}: {error}; showing contour only", stacklevel=2)
            caps.append(cap)
            contours.append(contour)
            display(cap_actors[i], cap)
            if show_contours:
                display(line_actors[i], contour)
            if mode == "cutaway":
                half = mesh.clip(normal=n, origin=o, invert=invert)
                clipped.append(half)
                display(mesh_actors[i], half)
        state.update(normal=n.copy(), origin=o.copy(), point_indices=ids,
                     point_categories=point_value[ids] if cloud is not None else np.array([]),
                     point_section=selected, caps=caps, contours=contours,
                     point_context=context_points, point_context_indices=context_ids,
                     clipped_meshes=clipped, errors=errors)
        renderer.ResetCameraClippingRange()
        if render:
            pl.render()

    state["actors"] = dict(points=point_actor, caps=cap_actors, contours=line_actors, meshes=mesh_actors,
                           point_context=context_point_actor)
    state["update"] = update_pl
    update_pl(normal, origin, render=render)
    if interactive:
        factor = np.asarray(widget_factor, dtype=float)
        if (factor.shape not in ((), (3,)) or not np.isfinite(factor).all()
                or np.any(factor <= 0)):
            raise ValueError("widget_factor must be a positive finite scalar or three positive finite values")
        widget_bounds = bounds
        if factor.ndim == 1:
            box = bounds.reshape(3, 2)
            center = box.mean(axis=1)
            half_size = (box[:, 1] - box[:, 0]) * factor / 2
            widget_bounds = np.column_stack((center - half_size, center + half_size)).ravel()
            factor = 1.0  # Per-axis scaling is already included in the bounds.
        else:
            factor = float(factor)
        normal_visible = show_widget_normal and axis is None
        widget = pl.add_plane_widget(update_pl, normal=state["normal"], origin=state["origin"].copy(),
            bounds=widget_bounds, factor=factor, color=widget_color,
            assign_to_axis=None if axis is None else "xyz"[axis], normal_rotation=normal_visible,
            interaction_event=interaction_event, test_callback=False)
        widget.SetDiagonalRatio(widget_normal_length_ratio)
        if widget_handle_size is not None:
            widget.SetHandleSize(widget_handle_size)
        widget.SetNormal(state["normal"])
        widget.UpdatePlacement()
        widget.SizeHandles()
        plane_color = pv.Color(widget_color if widget_plane_color is None else widget_plane_color).float_rgb
        for prop in (widget.GetPlaneProperty(), widget.GetSelectedPlaneProperty()):
            prop.SetColor(plane_color)
            prop.SetOpacity(widget_plane_opacity)
        widget.GetEdgesProperty().SetColor(pv.Color(widget_color).float_rgb)
        widget.GetEdgesProperty().SetLineWidth(widget_line_width)
        normal_color = pv.Color(widget_color if widget_normal_color is None else widget_normal_color).float_rgb
        for prop in (widget.GetNormalProperty(), widget.GetSelectedNormalProperty()):
            prop.SetColor(normal_color)
            prop.SetOpacity(widget_normal_opacity if normal_visible else 0)
            prop.SetLineWidth(widget_line_width if widget_normal_width is None else widget_normal_width)
        outline_color = pv.Color(widget_color if widget_outline_color is None else widget_outline_color).float_rgb
        for prop in (widget.GetOutlineProperty(), widget.GetSelectedOutlineProperty()):
            prop.SetColor(outline_color)
            prop.SetOpacity(widget_outline_opacity if show_widget_outline else 0)
            prop.SetLineWidth(widget_line_width if widget_outline_width is None else widget_outline_width)
        # VTK edge scalars otherwise override widget_color with red.
        for actor in renderer.GetActors():
            if actor.GetProperty() == widget.GetEdgesProperty():
                actor.GetMapper().ScalarVisibilityOff()
        state["widget"] = widget
        if not np.array_equal(widget.GetOrigin(), state["origin"]):
            update_pl(widget.GetNormal(), widget.GetOrigin(), render=render)
    return pl, state


def link_views_angles(pl, angles, centers=None, zoom=1.0, padding=1.05):
    """Apply per-subplot actor orientations and link one orthographic camera.

    Parameters
    ----------
    pl : pyvista.Plotter
        Plotter whose existing renderers and 3D actors are adjusted in place. Add all
        intended geometry before calling.
    angles : sequence of (roll, azimuth, elevation)
        One angle triple per renderer, in pl.renderers order, with all values in degrees.
        Each triple starts from the yz view and is applied in roll, azimuth, elevation
        order.
    centers : sequence of three-vectors or None, default: None
        One rotation/centering point per renderer in its original coordinates. None uses
        each renderer.center before transforming its actors.
    zoom : float, default: 1.0
        Positive shared magnification factor. Increasing zoom reduces parallel_scale and
        enlarges every subplot; a large value can crop the framing.
    padding : float, default: 1.05
        Positive framing multiplier. Values greater than 1 add space around the transformed
        geometry; values below 1 tighten the view and can crop it.

    Returns
    -------
    pl : pyvista.Plotter
        The input plotter, modified and rendered in place.

    Notes
    -----
    This function rotates/translates each existing vtkProp3D actor through its user matrix,
    placing the requested center at a common origin. It then shares one parallel-projection
    camera across all renderers and uses transformed bounds to choose its scale. Mesh point
    arrays are not rewritten.
    Call once after adding geometry. Repeated calls compose additional actor transforms;
    actors added later do not receive the earlier transform. Plane widgets and Python
    slicing calculations are not automatically updated to match actor user matrices.
    All subplots share parallel_scale, orientation, pan, and zoom. Equal viewport heights
    give equal world-units per pixel; differing viewport heights do not. The smallest
    viewport aspect ratio is used when fitting the horizontal extent. A transparent,
    nonpickable bounds actor named _linked_view_bounds is added to each renderer.
    Raises ValueError if angle/center counts do not match the renderers or if zoom/padding
    is nonpositive.

    Examples
    --------
    >>> pl = link_views_angles(pl, [(-90, 0, 0), (-90, 0, 0),
    ...                            (-90, -20, 0)], zoom=1.0, padding=1.05)
    """
    import pyvista as pv
    renderers = list(pl.renderers)

    if centers is None:
        centers = [renderer.center for renderer in renderers]

    if len(angles) != len(renderers) or len(centers) != len(renderers):
        raise ValueError("Provide one angle triple and one center per subplot.")

    if zoom <= 0 or padding <= 0:
        raise ValueError("zoom and padding must be positive.")

    for renderer, center, (roll, azimuth, elevation) in zip(
        renderers, centers, angles
    ):
        camera = pv.Camera()

        camera.position = (1.0, 0.0, 0.0)
        camera.focal_point = (0.0, 0.0, 0.0)
        camera.up = (0.0, 0.0, 1.0)

        camera.roll = roll
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.OrthogonalizeViewUp()

        rotation = pv.array_from_vtkmatrix(
            camera.GetViewTransformMatrix()
        )[:3, :3]

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ np.asarray(center)

        for actor in renderer.actors.values():
            if not actor.IsA("vtkProp3D"):
                continue

            old_matrix = actor.GetUserMatrix()
            old_matrix = (
                np.eye(4)
                if old_matrix is None
                else pv.array_from_vtkmatrix(old_matrix)
            )

            actor.SetUserMatrix(
                pv.vtkmatrix_from_array(transform @ old_matrix)
            )

    bounds = np.asarray(
        [renderer.bounds for renderer in renderers]
    ).reshape(-1, 3, 2)

    half_size = np.max(np.abs(bounds), axis=(0, 2))
    common_bounds = np.column_stack((-half_size, half_size)).ravel()

    width, height = pl.window_size
    viewports = np.asarray([renderer.viewport for renderer in renderers])

    aspects = (
        width * (viewports[:, 2] - viewports[:, 0])
        / (height * (viewports[:, 3] - viewports[:, 1]))
    )

    parallel_scale = (
        padding
        * max(half_size[1], half_size[0] / aspects.min())
        / zoom
    )

    distance = max(4.0 * np.linalg.norm(half_size), 1.0)

    bounds_box = pv.Box(bounds=common_bounds)

    for renderer in renderers:
        actor = pv.Actor(
            mapper=pv.DataSetMapper(dataset=bounds_box)
        )
        actor.prop.opacity = 0.0
        actor.pickable = False

        renderer.add_actor(
            actor,
            reset_camera=False,
            name="_linked_view_bounds",
        )

    pl.link_views()

    camera = renderers[0].camera
    camera.position = (0.0, 0.0, distance)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = (0.0, 1.0, 0.0)

    camera.parallel_projection = True
    camera.parallel_scale = float(parallel_scale)

    renderers[0].ResetCameraClippingRange(common_bounds)
    pl.render()

    return pl

class save_mesh:
    """Callable wrapper for exporting an existing PyVista plotter.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to display/export. The same object is returned by __call__, including when
        it has been closed.
    save_type : str or None, default: None
        Output-format override. None selects the lowercase filename extension when the
        object is called.

    Returns
    -------
    saver : save_mesh
        Callable object; use saver(filename, ...) to display/export the plotter.

    Notes
    -----
    The constructor only stores plotter and save_type. All export/display options are
    parameters of __call__. In this uploaded implementation, show=True and show=None display
    first and export through before_close_callback. show=False exports directly and closes.
    Supported dispatch branches are MP4/GIF, HTML, VTKSZ, VRML, OBJ, GLTF, VTK, and
    SVG/EPS/PS/PDF/TEX; other extensions use screenshot. The VTK branch specifically calls
    plotter.save and therefore requires a plotter-like object exposing that method; it is
    not a general multi-actor scene export. HTML exports current geometry without Python
    widget callbacks.

    Examples
    --------
    >>> saver = save_mesh(pl)
    >>> _ = saver("view.png", show=True, showkargs={"auto_close": True})
    """
    def __init__(self, plotter, save_type=None):
        self.plotter = plotter
        self.save_type = save_type

    def __call__(self, filename, show=None, transparent_background=None, viewup=None,
                 return_img=False, window_size=None, scale=None, raster=True, painter=True,
                 format='zip', jupyter_backend=None, showkargs=None, close=None, **kargs):
        """Display and export on close, or export the current scene directly.

        Parameters
        ----------
        filename : str, path-like, or None
            Output filename. None or another false value skips file export while retaining the
            selected show/close behavior.
        show : bool or None, default: None
            True and None call Plotter.show with a before-close saving callback. False runs
            saving immediately and then closes the plotter. None is not a save-only mode in this
            implementation.
        transparent_background : bool or None, default: None
            Screenshot background transparency, forwarded only to screenshot export. None uses
            the PyVista/theme setting.
        viewup : sequence of three float or None, default: None
            Camera up vector forwarded to save_movie for MP4/GIF export; unused for static
            images.
        return_img : bool, default: False
            Retained parameter. It is captured by the callback but currently forwarded to
            neither show nor screenshot, and __call__ always returns the plotter.
        window_size : sequence of two int or None, default: None
            Screenshot width and height in pixels. Not forwarded to save_graphic or
            Plotter.show; use showkargs for the display window size.
        scale : int or None, default: None
            Screenshot resolution multiplier, forwarded to screenshot; unused for vector and
            scene exports.
        raster : bool, default: True
            Rasterization flag forwarded to save_graphic for SVG/EPS/PS/PDF/TEX; ignored by PNG
            screenshots. Actual vector/raster content depends on the exporter.
        painter : bool, default: True
            Painter-order flag forwarded to save_graphic; ignored by PNG screenshots.
        format : str, default: 'zip'
            VTKSZ serialization format forwarded to export_vtksz, such as "zip" or "json". This
            does not select the filename extension.
        jupyter_backend : str or None, default: None
            Notebook backend passed to Plotter.show when show is not False. Desktop plotting
            follows the plotter configuration.
        showkargs : dict or None, default: None
            Arguments merged into Plotter.show options; interactive defaults to True. For
            example, use {"auto_close": True} to close after interaction. Do not include
            before_close_callback or jupyter_backend here because they are passed explicitly.
        close : bool or None, default: None
            Retained parameter; unused by the current implementation. Use
            showkargs["auto_close"] for the display branch. The show=False branch always closes.
        **kargs : dict
            Extra keyword arguments for save_movie only: framerate, factor, n_points, step,
            shift, focus, quality, and write_frames. They are not forwarded to static-image or
            scene exporters.

        Returns
        -------
        plotter : pyvista.Plotter
            The same plotter object. It may be closed after this call; no image array
            or viewer is returned by this wrapper.

        Notes
        -----
        The callback refreshes rendering without resetting the camera, then exports the current
        scene. To capture the final desktop view, use show=True and finish interaction with q.
        On Windows, the window close button can destroy the render context before export.
        With show=False, the plotter must already be rendered or configured for off-screen
        screenshots. With showkargs={"auto_close": False}, saving through the callback waits for
        actual plotter closure rather than merely returning from interaction. A closed render
        window cannot be used for a new scene/vector export.
        Notebook display may return before interaction finishes, so this call does not
        necessarily create the file immediately. The callback must remain registered until
        close. close and return_img are accepted but currently inactive; this documentation does
        not change that behavior.

        Examples
        --------
        >>> _ = save_mesh(pl)("view.svg", show=True, raster=False,
        ...                     painter=False, showkargs={"auto_close": True})
        """
        plotter=self.plotter
        window_size = plotter.window_size if window_size is None else window_size
        scale = plotter.scale if scale is None else scale

        def save_call(plotter, filename=filename, viewup=viewup, kargs=kargs,
                      format=format, raster=raster, painter=painter,
                      transparent_background=transparent_background, return_img=return_img,
                      window_size=window_size, scale=scale
                      ):
            if filename:
                filename = os.fspath(filename)
                save_type = (filename.split('.')[-1] if self.save_type is None else self.save_type).lower()
                if plotter.render_window is not None:
                    plotter.render()  # Refresh the current camera/scene without resetting it.
                if save_type in ('mp4', 'gif'):
                    self.save_movie(plotter, filename,  viewup=viewup,**kargs)
                elif save_type == 'html':
                    self.save_html(plotter, filename)
                elif save_type == 'vtksz':
                    self.save_vtksz(plotter, filename, format=format)
                elif save_type == 'vrml':
                    plotter.export_vrml(filename)
                elif save_type == 'obj':
                    plotter.export_obj(filename)
                elif save_type == 'gltf':
                    plotter.export_gltf(filename)
                elif save_type == 'vtk':
                    plotter.save(filename)
                elif save_type in ['svg', 'eps', 'ps', 'pdf', 'tex']:
                    plotter.save_graphic(filename, raster=raster, painter=painter)
                else:
                    plotter.screenshot(filename, transparent_background=transparent_background, 
                                       window_size=window_size, scale=scale)


        if not show is False:
            _showkargs =  {}
            # _showkargs['auto_close'] = False
            _showkargs['interactive'] =True
            # _showkargs['interactive_update'] =True
            # _showkargs['return_viewer'] =True
            # _showkargs['return_img'] =return_img
            # _showkargs['return_cpos'] =True
            if not showkargs is None:
                _showkargs.update(showkargs)

            if plotter.notebook:
                # _showkargs['auto_close'] = True
                plotter.show(jupyter_backend=jupyter_backend, **_showkargs)
                save_call(plotter)
            else:
                plotter.show(jupyter_backend=jupyter_backend, before_close_callback=save_call, **_showkargs)

        if show is False:
            save_call(plotter)
            plotter.close()
        return plotter

    @staticmethod
    def save_movie(plotter, filename, framerate=24, view_up = None, viewup=(0, 0, 0),
                    factor=2, n_points=120, step=0.1, 
                    shift=0, focus=None, quality=10, write_frames=True):
        path = plotter.generate_orbital_path(factor=factor, shift=shift, viewup=viewup, n_points=n_points)
        if filename.endswith('gif'):
            plotter.open_gif(filename)
        else:
            plotter.open_movie(filename, framerate=framerate, quality=quality)
        try:
            plotter.orbit_on_path(path, write_frames=write_frames, viewup=viewup, step=step, focus=focus)
        except Exception as error:
            raise RuntimeError('Movie export failed') from error
    @staticmethod
    def save_html(plotter, filename):
        try:
            plotter.trame.export_html(filename)
        except:
            plotter.export_html(filename)

    @staticmethod
    def save_vtksz(plotter, filename, format='zip'):
        plotter.export_vtksz(filename, format=format)

class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, state):
        self.actor.SetVisibility(state)

    def add_legend(
            self,
            labels=None,
            bcolor=(0.5, 0.5, 0.5),
            border=False,
            size=(0.2, 0.2),
            name=None,
            loc='upper right',
            face='triangle',
            font_family='courier',
        ):
        import pyvista as pv
        ###################### - Added - ########################
        legend_text = self._legend.GetEntryTextProperty()
        legend_text.SetFontFamily(pv.parse_font_family(font_family))
        #########################################################

        self.add_actor(self._legend, reset_camera=False, name=name, pickable=False)
        return self._legend
