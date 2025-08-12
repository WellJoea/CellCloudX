import os
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ._colors import color_palette, ListedColormap, cmap1
from ..utilis._arrays import list_iter, vartype
from ._utilis import colrows, clipdata

# interactive conda install conda-forge::glib

def surface(adata, meshlist=None, use_raw=False, groupby = None, splitby=None,
            basis='spatial3d', cmap=None, gcolors = None, outpre =None, format ='mp4', 
            titles=None, legend_titles=None, last_as_default=False,
            save=None,
            **kargs):
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

            gdtype = vartype(gdata)
            if gdtype == 'continuous':
                my_cmap = cmap1
                if not gcolors is None:
                    if type(gcolors) == dict:
                        my_cmap = gcolors.get(group, cmap1)
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
                edge_opacity=None,
                window_size=None,
                image_scale=None,
                edge_colors='#AAAAAA', 
                font_family = 'arial', #courier
                colors='#AAAAAA',
                show_actor = False, actor_size = 15, 
                startpos = 10, actor_space=5,
                color_off='grey', color_bg='grey',
                mvmin = None, mvmax = None,
                pvmin = None, pvmax = None, ptmin=None,
                ppmin=None, ppmax=None,
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

                view_angle = None, azimuth=None, roll=None, elevation=None,
                framerate=12, view_up = None, viewup=(0, 0, 0),
                factor=2, n_points=120, step=0.1, line_width=None,
                shift=0, focus=None, quality=8, write_frames=True,
                show=True, raster=True,
                title=None, view_isometric=False,
                title_position='upper_edge', 
                point_styles='points',
                render_points_as_spheres=True, 
                smooth_shading=True,
                anti_aliasing=True, 
                lighting=False, 
                aa_type ='fxaa', multi_samples=None, all_renderers=True,
                rotate_y = None, rotate_z = None, rotate_x = None, vtkz_format='json',
                save=None, save_type=None, 

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

    if server_proxy_enabled:
        pv.global_theme.trame.server_proxy_enabled = server_proxy_enabled
    if not server_proxy_prefix is None:
        pv.global_theme.trame.server_proxy_prefix = server_proxy_prefix
    if start_xvfb:
        pv.start_xvfb() 
    if not jupyter_backend is None:
        pv.set_jupyter_backend(jupyter_backend)
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

            pl.add_mesh(mesh, show_edges=show_edges[j], 
                        opacity=opacity[j], 
                        edge_opacity=edge_opacity[j], 
                        scalars=scalars[j],
                        line_width=line_width[j],
                        edge_color=edge_colors[j], 
                        clim=clims[j],
                        color=colors[j], lighting=lighting,
                        smooth_shading=smooth_shading,
                        **mkargs)

            if not rotate_x is None:
                mesh = mesh.rotate_x(rotate_x, inplace=False)
            if not rotate_y is None:
                mesh = mesh.rotate_y(rotate_y, inplace=False)
            if not rotate_z is None:
                mesh = mesh.rotate_z(rotate_z, inplace=False)
    groupbys = list_iter(groupbys, last_as_default=last_as_default)
    points_colors = list_iter(points_colors, last_as_default=last_as_default)
    legend_label_size = list_iter(legend_label_size, last_as_default=last_as_default)
    point_opacity = list_iter(point_opacity, last_as_default=last_as_default)
    point_styles = list_iter(point_styles, last_as_default=last_as_default)
    point_psize =  list_iter(point_psize, last_as_default=last_as_default)
    sclbar = dict(height=legend_height, width=legend_width, 
                  vertical = legend_vertical,
                    n_labels=5,
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
            if isinstance(points, (list, np.ndarray)):
                points =pd.DataFrame(points, columns=list('xyz'))
            assert isinstance(points, pd.DataFrame)

            pvdt = pv.PolyData(points[['x', 'y', 'z']].values.astype(np.float64))

            groupby = groupbys[idx]
            igroup_color = points_colors[idx]
            if groupby is None:
                groupby = 'group'
                points[groupby] = '1'

            assert groupby in points.columns.tolist()
            groups = points[groupby]
            pvdt[groupby] = groups
            gdtype = vartype(groups)

            if gdtype == 'discrete':
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
                # my_cmap = ListedColormap(color_list)
                my_cmap = color_list
                # point_opacity = [ None if point_opacity[i] is None else float(point_opacity[i]) for i in Order.index ]
                # point_psize = [ None if point_psize[i] is None else  float(point_psize[i]) for i in Order.index ]
                
                categories = True
                sclbar["n_labels"] = 0 #len(Order)
                sclbar["n_colors"] = len(Order)
                # sclbar["height"] =  0.85

            elif gdtype == 'continuous':
                pvdt.set_active_scalars(groupby)
                pvdt[groupby] = clipdata(pvdt[groupby],  
                                    vmin=pvmin, vmax=pvmax,
                                    pmin=ppmin, pmax=ppmax, 
                                    clips=None, tmin = ptmin, dropmin=False)
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
                my_cmap = cmap1 if igroup_color is None else igroup_color
                scale_rgba = (pvdt[groupby] - pvdt[groupby].min() )/(pvdt[groupby].max()- pvdt[groupby].min())
                scale_rgba = my_cmap(scale_rgba)

            if (not show_actor):
                sclbar['position_x'] = sclbar['position_x'] + legend_x_shift*idx
                sclbar['position_y'] = sclbar['position_y'] + legend_y_shift*idx
                sclbar['label_font_size'] = legend_label_size[idx]
                if legend_title is False:
                    legend_title = None
                elif legend_title is None:
                    legend_title = None if groupby is None else str(groupby)
                else:
                    legend_title = legend_title
                sclbar['title'] = legend_title
                actor = pl.add_points(pvdt, 
                            style=point_styles[idx],
                            cmap=my_cmap,
                            color=None, 
                            opacity = point_opacity[idx],
                            point_size = point_psize[idx],
                            # label =  None if groupby is None else str(groupby),
                            scalars = groupby,
                            scalar_bar_args=sclbar,
                            
                            # rgba=True,
                            # emissive=True,
                            render_points_as_spheres=render_points_as_spheres, 
                            show_scalar_bar=show_scalar_bar,
                            categories=categories,
                            name = str(idx),
                            # annotations=annotations,
                            **pkargs)

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
        pl.set_scale(xscale=xyzscales[0], yscale=xyzscales[1], zscale=[2])
    if not view_up is None:
        pl.set_viewup(view_up)
    if cpos:
        pl.camera_position = cpos
    if azimuth:
        pl.camera.azimuth = azimuth
    if elevation:
        pl.camera.elevation = elevation
    if view_angle:
        pl.camera.view_angle = view_angle
    if roll:
        pl.camera.roll = roll
    if view_isometric:
        pl.view_isometric()

    if title:
        actor = pl.add_title(title, #position=title_position, 
                             font=main_title_font, color=main_title_color, font_size=main_title_size)

    # pv.theme.restore_defaults()
    return save_mesh(pl, save_type=save_type)(save, show=show,
                    framerate=framerate, #view_up = view_up, 
                    viewup=viewup, factor=factor, n_points=n_points, 
                    step=step, shift=shift, focus=focus,
                    scale=image_scale, window_size=window_size,
                    quality=quality, write_frames=write_frames,
                    raster=raster,
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

                view_angle = None, azimuth=None, roll=None, elevation=None,
                framerate=12, view_up = None, viewup=(0, 0, 0),
                factor=2, n_points=120, step=0.1, line_width=None,
                shift=0, focus=None, quality=8, write_frames=True,
                show=True, raster=True,
                title=None, view_isometric=False,
                title_position='upper_edge', 
                point_styles='points',
                render_points_as_spheres=True, 
                smooth_shading=True,
                anti_aliasing=True, 
                lighting=False, 
                aa_type ='fxaa', multi_samples=None, all_renderers=True,
                rotate_y = None, rotate_z = None, rotate_x = None, vtkz_format='json',
                save=None, save_type=None, 

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
    
                args_cbar = {},
                pkargs= dict(lighting=False), 
                skargs={},
                fkargs={},
                mkargs={},
                showkargs={},
                ):
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

    if server_proxy_enabled:
        pv.global_theme.trame.server_proxy_enabled = server_proxy_enabled
    if not server_proxy_prefix is None:
        pv.global_theme.trame.server_proxy_prefix = server_proxy_prefix
    if start_xvfb:
        pv.start_xvfb() 
    if not jupyter_backend is None:
        pv.set_jupyter_backend(jupyter_backend)
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
                        lighting=lighting,
                        smooth_shading=smooth_shading,
                        **mkargs
                        )
    
            if not rotate_x is None:
                mesh = mesh.rotate_x(rotate_x, inplace=False)
            if not rotate_y is None:
                mesh = mesh.rotate_y(rotate_y, inplace=False)
            if not rotate_z is None:
                mesh = mesh.rotate_z(rotate_z, inplace=False)

    points_colors = list_iter(points_colors, last_as_default=last_as_default)
    legend_label_size = list_iter(legend_label_size, last_as_default=last_as_default)
    point_opacity = list_iter(point_opacity, last_as_default=last_as_default)
    point_psize =  list_iter(point_psize, last_as_default=last_as_default)
    point_styles = list_iter(point_styles, last_as_default=last_as_default)

    sclbar = dict(height=legend_height, width=legend_width, 
                  vertical = legend_vertical,
                    n_labels=5,
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
            groupby = 'group'
            points[groupby] = '1'

        assert groupby in points.columns.tolist()
        groups = points[groupby]
        gdtype = vartype(groups)

        pvdt = pv.PolyData(points[['x', 'y', 'z']].values.astype(np.float64))
        pvdt[groupby] = groups

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
                                    # rgba=True,
                                    **pkargs
                    )

    if anti_aliasing is None:
        pass
    elif anti_aliasing is True:
        pl.enable_anti_aliasing(aa_type, multi_samples=multi_samples, all_renderers=all_renderers)
    elif anti_aliasing is False:
        pl.disable_anti_aliasing()

    # add_legend(pl)
    if not xyzscales is None:
        pl.set_scale(xscale=xyzscales[0], yscale=xyzscales[1], zscale=[2])
    if not view_up is None:
        pl.set_viewup(view_up)
    if cpos:
        pl.camera_position = cpos
    if azimuth:
        pl.camera.azimuth = azimuth
    if elevation:
        pl.camera.elevation = elevation
    if view_angle:
        pl.camera.view_angle = view_angle
    if roll:
        pl.camera.roll = roll
    if view_isometric:
        pl.view_isometric()
    if show_axes:
        pl.show_axes()
    if title:
        actor = pl.add_title(title, #position=title_position, 
                             font=main_title_font, color=main_title_color, font_size=main_title_size)  
    return save_mesh(pl, save_type=save_type)(save, show=show,
                    framerate=framerate, #view_up = view_up, 
                    viewup=viewup, factor=factor, n_points=n_points, 
                    step=step, shift=shift, focus=focus,
                    scale=image_scale, window_size=window_size,
                    quality=quality, write_frames=write_frames,
                    raster=raster,
                     jupyter_backend=jupyter_backend, showkargs=showkargs, **skargs)

class save_mesh:
    def __init__(self, plotter, save_type=None):
        self.plotter = plotter
        self.save_type = save_type

    def __call__(self, filename, show=False, transparent_background=None, viewup=None,
                 return_img=False, window_size=None, scale=None, raster=True, painter=True,
                 format='zip', jupyter_backend=None, showkargs={}, **kargs):
        plotter=self.plotter

        if filename:
            save_type = filename.split('.')[-1] if self.save_type is None else self.save_type
            if save_type == 'mp4':
                self.save_movie(plotter, filename,  viewup=viewup,**kargs)
            elif save_type == 'html':
                self.save_html(plotter, filename)
            elif save_type == 'vtksz':
                self.save_vtksz(plotter, filename)
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
                                   return_img=return_img)
                                   #window_size=window_size)
        if show:
            plotter.show(jupyter_backend=jupyter_backend, **showkargs)
            return plotter
        elif show is False:
            plotter.close()
        else:
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
        except:
            raise('please install imageio and imageio-ffmpeg like this:```pip install imageio imageio-ffmpeg```')
    @staticmethod
    def save_html(plotter, filename):
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

