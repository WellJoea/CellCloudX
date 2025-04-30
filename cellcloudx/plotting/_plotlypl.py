import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scanpy as sc
import plotly


from ..plotting._utilis import colrows
from ..plotting._colors import *
from ..plotting._spatial3d import get_spatial_info
from ..plotting._utilis import image2batyes
from ..utilis._arrays import list_iter, vartype

class Fatch():
    def __init__(self, adata = None,
                    basis='X_umap',
                    groupby=None, splitby=None, use_raw=False,
                    dimsname=None, obs='_ALL',
                    cmap = None, palette = None,
                    sample = None,
                    seed = 200504,
                    axes = None):
        np.random.seed(seed)

        self.adata = adata #adata.copy() add color to uns

        self.groupby=groupby
        self.splitby=splitby
        self.fetchdata( use_raw=use_raw,
                        basis= basis, dimsname=dimsname,
                        cmap = cmap, palette = palette,
                        sample = sample,
                        obs=obs,
                        axes = axes)

    def fetchdata(self, use_raw=False,
                    basis='X_umap', dimsname=None,
                    cmap = None, palette = None,
                    axes = None, sample = None,
                    obs=None,
                     **kargs):
        adata=self.adata
        if (not adata is None) and use_raw:
            adataX = adata.raw.to_adata()
        else:
            adataX = adata

        try:
            corr_df = pd.DataFrame(adata.obsm[basis],
                                    index =adata.obs_names).copy()
            corr_df.columns = [ f"{basis.strip('X_')}_{i}" for i in range(corr_df.shape[1])]
        except ValueError:
            print("Error: Please input valid basis name or corr_df.")


        if axes is not None:
            self.dims = len(axes)
            self.axes = axes
        else:
            self.dims = corr_df.shape[1]
            self.axes  = range(self.dims)


        groups = [self.groupby] if type(self.groupby) in [str, int, float, bool] else self.groupby
        groups = list(dict.fromkeys(groups))
        if not self.splitby is None:
            groups.append(self.splitby)
        
        if not obs is None:
            if (type(obs) in [str]) and (obs =='_ALL'):
                obs = [ i for i in adata.obs.columns if i not in groups]
                groups += list(obs)
            elif  type(self.groupby) in [str, int, float, bool]:
                groups.append(obs)
            else:
                groups += list(obs)

        var_groups = adataX.var_names.intersection(groups).tolist()
        obs_groups = adataX.obs.columns.intersection(groups).tolist()

        group_df = []
        if len(obs_groups)>0:
            group_df.append(adataX.obs[obs_groups])
        if len(var_groups)>0:
            try:
                var_arr = adataX[:,var_groups].X.toarray()
            except:
                var_arr = adataX[:,var_groups].X
            var_arr = pd.DataFrame(var_arr, 
                                    index=adataX.obs_names,
                                    columns=var_groups)
            group_df.append(var_arr)
        assert len(group_df)>0, 'No group was fond in adata.'
        group_df = pd.concat(group_df, axis=1)
        groups  = [i for i in groups if i in group_df.columns.tolist()]
        group_df = group_df[groups]

        dimsname = corr_df.columns.tolist() if dimsname is None else dimsname
        corr_df.columns = dimsname

        colors = {}
        gdtype = {}
        #print(group_df.head(), group_df.dtypes)
        for col in group_df.columns:
            igroup = group_df[col]
            gtype = vartype(igroup)
            if gtype == 'discrete':
                try:
                    iorders = igroup.cat.categories.tolist()
                except:
                    iorders = igroup.unique().tolist()
                group_df[col] = pd.Categorical(igroup, categories=iorders)
    
                if type(palette) in [list, np.ndarray]:
                    icolor = palette
                elif type(cmap) in [dict]:
                    icolor = palette[col]
                else:
                    if adata is None:
                        icolor = color_palette(len(iorders))
                    else:
                        iuns_cor = f'{col}_colors'
                        if (iuns_cor in adata.uns.keys()) and len(iorders)<= len(adata.uns[iuns_cor]):
                            icolor = adata.uns[iuns_cor]
                        else:
                            adata.obs[col] = pd.Categorical(igroup, categories=iorders)
                            adata_color(adata, value = col, cmap=cmap, palette=palette) 
                            icolor = adata.uns[iuns_cor]
                            adata.uns[iuns_cor] = icolor
            else:
                if type(cmap) == str:
                    icolor = cmap
                # elif type(cmap) in [list, np.ndarray]:
                #     cmap = list(cmap)
                #     icolor = cmap.pop(0)
                elif type(cmap) in [dict]:
                    icolor = cmap[col]
                elif cmap is None:
                    icolor = cmap1px
                else:
                    icolor = cmap
            colors[col] = icolor
            gdtype[col] = gtype

        if sample is not None:
            if 0<=sample <=1:
                isize = int(adata.shape[0]*sample)
            else:
                assert type(sample) == int
                isize = sample
            obs_idex = np.random.choice(adata.obs_names, 
                            size=isize, 
                            replace=False, p=None)
            corr_df = corr_df.loc[obs_idex,:]
            group_df = group_df.loc[obs_idex,:]
        data = pd.concat([corr_df, group_df], axis=1)

        self.dimsname = dimsname
        self.dims = corr_df.shape[1]
        self.colors = colors
        self.gdtype = gdtype
        self.corr_df = corr_df
        self.group_df = group_df
        self.groupby = [i for i in self.groupby if i in groups] 
        self.data = data
        self.data['index'] = data.index
        self.ngroup = len(self.groupby)
        return self

    def fetchimages(self, images=None, 
                    img_key="hires", basis = None, 
                    library_id=None, rescale=None):
        
        if images is None:
            adata = self.adata
            basis = 'spatial' if basis is None else basis
            rescale = 1 if rescale is None else rescale
        
            library_ids = list(adata.uns[basis].keys()) if (library_id is None) else library_id    
            if isinstance(library_id, (str, int)):
                library_ids = [library_id]

            images = []
            for lid in library_ids:
                try:
                    img_dict = adata.uns[basis][lid]
                    iimg = img_dict['images'][img_key]
                    scale_factor = img_dict['scalefactors'].get(f'tissue_{img_key}_scalef', 1)
                    spot_diameter_fullres = img_dict['scalefactors'].get('spot_diameter_fullres',1)
                
                    rescale = scale_factor*rescale
                    if (not np.allclose(rescale, 0.9999, rtol=1e-04, atol=1e-08)) and (rescale != 0):
                        import skimage.transform as skitf
                        if iimg.ndim == 2:
                            iimg = skitf.rescale(iimg, 1/rescale)
                        else:
                            iimg = skitf.rescale(iimg, (1/rescale,1/rescale,1))
                        iimg = np.uint8(iimg*255)
                except:
                    print('Warning: no image found for library_id:', lid)
                    iimg= None
                images.append(iimg)
        return images
    
    @staticmethod
    def add_layout_images(fig, images, ncols, x_reverse=False, y_reverse=False, 
                            sizing='stretch', image_opacity=None,
                            xanchor="left",
                            yanchor="bottom",
                            **kargs):

        if (not images is None) and len(images)>0:
            ncells = len(images)
            nrows, ncols = colrows(ncells,  ncols=ncols, soft=False)
            idx_arr = np.arange(nrows*ncols).reshape(nrows,ncols, order='C')[::-1,]
            for i, image in enumerate(images):
                if image is None:
                    continue
                sizey, sizex = image.shape[:2]
                x = sizex if x_reverse else 0
                y = sizey if y_reverse else 0
                imagebt = image2batyes(image)
                # import PIL
                # imagebt = PIL.Image.fromarray(image)
                row = (i // ncols)
                col = (i % ncols)
                xy_idx = idx_arr[row, col] + 1
                if xy_idx==1:
                    ix, iy = 'x', 'y'
                else:
                    ix, iy = f'x{xy_idx}', f'y{xy_idx}'
                fig.add_layout_image(
                        source=imagebt,
                        # row=row,
                        # col=col,
                        xref=ix,
                        yref=iy,
                        x=x, 
                        y=y,
                        xanchor=xanchor,
                        yanchor=yanchor,
                        layer="below",
                        sizing=sizing,
                        sizex=sizex,
                        sizey=sizey,
                        opacity = image_opacity,
                        **kargs
                )

    @staticmethod
    def add_layout_image(fig, image, x_reverse=False, y_reverse=False, 
                            sizing='stretch', image_opacity=None,
                            xanchor="left",
                            yanchor="bottom",
                            **kargs):

        if (not image is None):
            sizey, sizex = image.shape[:2]
            x = sizex if x_reverse else 0
            y = sizey if y_reverse else 0
            imagebt = image2batyes(image)
            # import PIL
            # imagebt = PIL.Image.fromarray(image)
            xy_axes = set([ (ifig['xaxis'], ifig['yaxis']) for ifig in fig.data ])
            for (ix, iy) in xy_axes:
                fig.add_layout_image(
                        source=imagebt,
                        # row=row,
                        # col=col,
                        xref=ix,
                        yref=iy,
                        x=x, 
                        y=y,
                        xanchor=xanchor,
                        yanchor=yanchor,
                        layer="below",
                        sizing=sizing,
                        sizex=sizex,
                        sizey=sizey,
                        opacity = image_opacity,
                        **kargs
                )

    @staticmethod
    def add_image(fig, images, image_opacity=None,**kargs):
        if (not images is None) and len(images)>0:
            xyaxis = []
            for i, ifig in enumerate(fig.data):
                ix, iy =  ifig['xaxis'], ifig['yaxis']
                if (len(images)>=i+1):
                    image = images[i] 
                    if image is None:
                        continue
                    imagebt = image2batyes(image)
                if not [ix, iy] in xyaxis:
                    xyaxis.append([ix, iy])
                    fig.add_image(
                        source=imagebt,
                        xaxis =ix,
                        yaxis =iy,
                        opacity=image_opacity,
                        **kargs
                    )

        # for i in fig.data:
        #     print(ixy)
        #     if type(i) == plotly.graph_objs._scattergl.Scattergl:
        #         fig.add_image(
        #             source=imagebt,
        #             xaxis =f'x{ixy}',
        #             yaxis =f'y{ixy}', 
        #             opacity=image_opacity,
        #             **kargs
        #         )
        #         ixy +=1

def splitscatter2d(adata, groupby, splitby,  basis='X_umap', use_raw=False,
               cmap = None, show_background=False,
               save=None,  size=None, size_max =20, scale=1, show=True,
                template='none', scene_aspectmode='data',
                y_reverse=False, x_reverse=False,  
                img_x_reverse=False, img_y_reverse=False, same_scale=True, 
                scene_aspectratio=dict(x=1, y=1), return_fig =False,
                ncols=2, soft=False, 
                figscale=600, werror=20, 
                width = None, height=None,
                clips = None, vmin = None, vmax = None, showlegend=True,
                show_grid =True, 
                showticklabels = False,
                itemsizing='constant',
                itemwidth=None,
                show_image=False,
                sample=None,
                opacity=None,
                render_mode='auto', 
                image_opacity = None,
                facet_col_spacing=None,
                facet_row_spacing=None,
    
                thicknessmode="pixels", 
                cbar_width=25,
                lenmode="pixels", 
                cbar_height=300,
                cbar_yanchor="middle",
                cbar_ypos=0.5,
                cbar_ypad=10,
                ticks="outside", 

                images=None, 
                img_key="hires", 
                img_basis = None, 
                library_id=None, 

                legdict={},
                dargs={},
                **kargs, ):
    Fd = Fatch(adata = adata,
                    basis=basis,
                    groupby=groupby,
                    splitby=splitby,
                    use_raw=use_raw,
                    cmap = cmap, sample=sample,
                    **dargs)
    data_df = Fd.data.copy()
    Order = data_df[splitby].cat.categories.tolist()
    nrows, ncols = colrows(len(Order), ncols=ncols, soft=soft)
    dimdict= dict(zip(list('xy'), Fd.dimsname))

    clips = None if clips is None else list(clips)
    width = ncols*figscale+werror  if width is None else width
    height= nrows*figscale if height is None else height

    gtype = Fd.gdtype[groupby]
    if gtype=='discrete':
        category_orders = data_df[groupby].cat.categories.tolist()
        color_discrete_sequence = Fd.colors[groupby]
    elif gtype =='continuous':            
        cmap = Fd.colors[groupby]
        cmin = data_df[groupby].min()
        if not vmin is None:
            data_df = data_df.loc[(data_df[groupby]>=vmin), :]
        if not vmax is None:
            data_df[groupby] = data_df[groupby].clip(None, vmax)
 
        if not clips is None:
            if np.ndim(clips)==1:
                data_df[groupby] = np.clip(data_df[groupby] , clips[0], clips[1])
            elif np.ndim(clips) > 1:
                data_df[groupby]  = np.clip(data_df[groupby] , clips.pop(0)[0], clips.pop(0)[1])
        
        cmin = data_df[groupby].min()
        cmax = data_df[groupby].max()
        dimdict['range_color'] = [float(cmin), float(cmax)]


    data_df.sort_values(by = groupby, ascending=False, inplace=True)
    data_df['index'] = data_df.index

    fig = px.scatter(data_df, color=groupby, 
                        facet_col=splitby, 
                        facet_col_wrap=ncols,
                        facet_col_spacing=facet_col_spacing,
                        facet_row_spacing=facet_row_spacing,
                        size = size,
                        size_max=size_max,
                        render_mode=render_mode, 
                        width=width, height=height,
                        color_discrete_sequence=color_discrete_sequence,
                        color_continuous_scale = cmap,
                        hover_name="index", hover_data=["index"],
                        category_orders={groupby: category_orders, splitby: Order},
                        **dimdict, **kargs)
    # for i in range(len(fig.data)):
    #     if hasattr(fig.data[i], 'showlegend'):
    #         fig.data[i].showlegend = True
    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")
    if show_image:
        images = Fd.fetchimages(images=images, 
                                img_key=img_key, 
                                basis = img_basis or basis, 
                                library_id=Order)
        Fd.add_layout_images(fig, images, ncols,
                    image_opacity=image_opacity,
                    xanchor="right" if x_reverse else "left",
                    yanchor="top" if y_reverse else "bottom",
                    x_reverse=img_x_reverse,
                    y_reverse=img_y_reverse)

    itemwidth = itemwidth or 30
    fig.update_layout(  
                        legend=dict( itemsizing = itemsizing, itemwidth=itemwidth, **legdict),
                        coloraxis_colorbar=dict(
                            # title="",
                            thicknessmode=thicknessmode, 
                            thickness=cbar_width,
                            lenmode=lenmode, 
                            len=cbar_height,
                            yanchor=cbar_yanchor,
                            y=cbar_ypos,
                            ypad=cbar_ypad,
                            ticks=ticks, 
                        ),

                        showlegend=showlegend,
                        scene_aspectmode=scene_aspectmode,
                        template=template,
                        scene_aspectratio=scene_aspectratio,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                        # images=add_images,
                        plot_bgcolor='#FFFFFF',) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_bgcolor='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)
    if not size is None:
        fig.update_traces(marker=dict(opacity=opacity, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    else:
        fig.update_traces(marker=dict(opacity=opacity, size=scale, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    if same_scale:
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )


    if show:
        fig.show()
    if save:
        fig.write_html(save)
    if return_fig:
        return fig

def scatter2ds(adata, groupbys=None, basis='X_umap', use_raw=False,
               cmap = None,
               save=None, outpre=None, size=None, size_max =20, scale=1, show=True,
                template='none', scene_aspectmode='data',
                render_mode='auto', 
                y_reverse=False, x_reverse=False,  
                img_x_reverse=False, img_y_reverse=False, same_scale=True, 
                scene_aspectratio=dict(x=1, y=1), return_fig =False,
                ncols=2, soft=False, 
                figscale=600, werror=20, 
                width = None, height=None,
                clips = None, vmin = None, vmax = None, showlegend=True,
                show_grid =True, 
                showticklabels = False,
                itemsizing='constant',
                itemwidth=None,
                show_image=False,
                sample=None,
                opacity = None,
                image_opacity = None,
                facet_col_spacing=None,
                facet_row_spacing=None,
                
                thicknessmode="pixels", 
                cbar_width=25,
                lenmode="pixels", 
                cbar_height=300,
                cbar_yanchor="middle",
                cbar_ypos=0.5,
                cbar_ypad=10,
                ticks="outside", 

                images=None, 
                img_key="hires", 
                img_basis = None, 
                library_id=None, 

                legdict={},
                dargs={},
                **kargs, ):
    Fd = Fatch(adata = adata,
                    basis=basis,
                    groupby=groupbys, use_raw=use_raw,
                    cmap = cmap, sample=sample,
                    **dargs)
        
    nrows, ncols = colrows(Fd.ngroup, ncols=ncols, soft=soft)
    dimdict= dict(zip(list('xy'), Fd.dimsname))
    data_df = Fd.data.copy()

    clips = None if clips is None else list(clips)
    width = ncols*figscale+werror  if width is None else width
    height= nrows*figscale if height is None else height

    category_orders = []
    color_discrete_sequence = []
    dis_groups = []
    con_groups = []
    for col in Fd.groupby:
        gtype = Fd.gdtype[col]
        if gtype=='discrete':
            category_orders.extend(data_df[col].cat.categories.tolist())
            color_discrete_sequence.extend(Fd.colors[col])
            dis_groups.append(col)
        elif gtype =='continuous':            
            cmap = Fd.colors[col]
            con_groups.append(col)
    # if len(dis_groups)<1:
    #     raise ValueError('Only discrete groups will be ploted!!')

    assert not ((len(dis_groups)>0) & (len(con_groups)>0))
    #plot
    SData = pd.melt(data_df, id_vars= Fd.dimsname, 
                            value_vars=dis_groups+ con_groups,
                            var_name='groups', value_name='Type')
    
    if len(con_groups)>0:
        if not vmin is None:
            SData = SData.loc[(SData['Type']>=vmin), :]
        if not vmax is None:
            SData['Type'] = SData['Type'].clip(None, vmax)
        # idata = idata[idata[group]<=vmax]
        if not clips is None:
            if np.ndim(clips)==1:
                SData['Type'] = np.clip(SData['Type'] , clips[0], clips[1])
            elif np.ndim(clips) > 1:
                SData['Type']  = np.clip(SData['Type'] , clips.pop(0)[0], clips.pop(0)[1])

    SData.sort_values(by = 'Type', ascending=False, inplace=True)
    SData['index'] = SData.index
    fig = px.scatter(SData, color="Type", 
                        facet_col="groups", 
                        facet_col_wrap=ncols,
                        facet_col_spacing=facet_col_spacing,
                        facet_row_spacing=facet_row_spacing,
                        size = size,
                        size_max=size_max,
                        render_mode=render_mode,
                        width=width, height=height,
                        color_discrete_sequence=color_discrete_sequence,
                        color_continuous_scale = cmap,
                        hover_name="index", hover_data=["index"],
                        category_orders={'Type': category_orders},
                        **dimdict, **kargs)
    # for i in range(len(fig.data)):
    #     if hasattr(fig.data[i], 'showlegend'):
    #         fig.data[i].showlegend = True
    itemwidth = itemwidth or (30+len(dis_groups))
    fig.update_layout(  
                        legend=dict( itemsizing = itemsizing, itemwidth=itemwidth, **legdict),
                        coloraxis_colorbar=dict(
                            # title="",
                            thicknessmode=thicknessmode, 
                            thickness=cbar_width,
                            lenmode=lenmode, 
                            len=cbar_height,
                            yanchor=cbar_yanchor,
                            y=cbar_ypos,
                            ypad=cbar_ypad,
                            ticks=ticks, 
                        ),

                        showlegend=showlegend,
                        scene_aspectmode=scene_aspectmode,
                        template=template,
                        scene_aspectratio=scene_aspectratio,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                        plot_bgcolor='#FFFFFF',) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_bgcolor='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)
    if not size is None:
        fig.update_traces(marker=dict(opacity=opacity, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    else:
        fig.update_traces(marker=dict(size=scale, 
                                      opacity=opacity,
                                      line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    if same_scale:
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")
    if show_image:
        images = Fd.fetchimages(images=images, 
                                img_key=img_key, 
                                basis = img_basis or basis, 
                                library_id=library_id)
        Fd.add_layout_image(fig, images[0],
                    image_opacity=image_opacity,
                    xanchor="right" if x_reverse else "left",
                    yanchor="top" if y_reverse else "bottom",
                    x_reverse=img_x_reverse,
                    y_reverse=img_y_reverse)

    if show:
        fig.show()
    if outpre or save:
        save= outpre + '.'+ '.'.join(dis_groups) + '.2d.html' if outpre else save
        fig.write_html(save)
    if return_fig:
        return fig

def scatter3d(adata, groupby, basis='X_umap', use_raw=False,
               cmap = None, select=None, nacolor='lightgrey', show=True,
                xyz= [0,1,2],
                size=None, size_max=20, scale=1,
                width=None, height=None,
                scene_aspectmode='data', keep_all=False, order=True, 
                ascending=False, return_fig =False, 
                show_grid =True, 
                showticklabels = False,
                y_reverse=False, x_reverse=False, z_reverse=False,
                scene_aspectratio=dict(x=1, y=1, z=1),
                itemsizing='constant',
                template='none', save=None, 
                vmin = None, vmax = None, clip =None,
                sample=None,
                gridwidth=1, gridcolor='#AAAAAA',
                
            thicknessmode="pixels", 
            cbar_width=25,
            lenmode="pixels", 
            cbar_height=300,
            cbar_yanchor="middle",
            cbar_ypos=0.5,
            cbar_ypad=10,
            ticks="outside", 
            # ticksuffix=" bills",
            # dtick=5

            legdict={}, dargs={},
            **kargs):
    Fd = Fatch(adata = adata,
                    basis=basis,
                    groupby=[groupby], use_raw=use_raw,
                    cmap = cmap, sample=sample, **dargs)

    if Fd.dims <3:
        raise ValueError('The dims must be larger than 2!!')
    dimdict = dict(zip(list('xyz'), np.array(Fd.dimsname)[xyz]))
    # ctype
    ctype = Fd.gdtype[groupby]
    color = Fd.colors[groupby]
    idata = Fd.data.copy()

    dimdict.update(pxsetcolor(color, ctype=ctype,  cmap=cmap))

    if ctype == 'discrete':
        order = idata[groupby].cat.categories.tolist()
        if not keep_all:
            idata[groupby] = idata[groupby].cat.remove_unused_categories()
            keep_order = idata[groupby].cat.categories.tolist()
            colors = dimdict['color_discrete_sequence']
            if type(colors)==list:
                colors =[c for c,o in zip(colors, order) if o in keep_order]
                dimdict['color_discrete_sequence'] = colors
            order = keep_order
        category_orders={groupby: order}
        if not select is None:
            select = [select] if type(select) == str else select
            if type(dimdict['color_discrete_sequence'])==list:
                colors = [  cs if co in select else nacolor
                            for co,cs in  zip(category_orders[groupby], dimdict['color_discrete_sequence']) ]
                dimdict['color_discrete_sequence'] = colors
        dimdict.update({'category_orders': category_orders}) #'animation_frame': group

    elif ctype == 'continuous':
        cmin = idata[groupby].min()
        if not vmin is None:
            idata = idata[idata[groupby]>=vmin]
        if not vmax is None:
            idata[groupby] = idata[groupby].clip(None, vmax)
            # idata = idata[idata[group]<=vmax]
        if not clip is None:
            idata[groupby] = np.clip(idata[groupby], clip[0], clip[1])

        if order =='guess' or  order == True:
            idata.sort_values(by = groupby, ascending=ascending, inplace=True)
        cmax = idata[groupby].max()
        dimdict['range_color'] = [float(cmin), float(cmax)]


    fig = px.scatter_3d(idata, 
                        color=groupby, 
                        size=size,
                        size_max=size_max,
                        width=width, height=height,
                        hover_name="index", hover_data=["index"],
                            **dimdict, **kargs)
    
    fig.update_layout(legend=dict(itemsizing = itemsizing, **legdict),
                        coloraxis_colorbar=dict(
                            # title="",
                            thicknessmode=thicknessmode, 
                            thickness=cbar_width,
                            lenmode=lenmode, 
                            len=cbar_height,
                            yanchor=cbar_yanchor,
                            y=cbar_ypos,
                            ypad=cbar_ypad,
                            ticks=ticks, 
                        ),
                        scene_aspectmode=scene_aspectmode,
                        scene_aspectratio=scene_aspectratio,
                        template=template,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels, gridwidth=gridwidth, gridcolor=gridcolor),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels, gridwidth=gridwidth, gridcolor=gridcolor),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels, gridwidth=gridwidth, gridcolor=gridcolor),
                        ),
                        plot_bgcolor='#FFFFFF',) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)
    markerdict=dict(marker  = {'line':{'width':0, 'color':'DarkSlateGrey' }},
                    selector={'mode':'markers'})
    if size is None:
        markerdict['marker']['size'] = scale
    fig.update_traces(**markerdict)

    if y_reverse:
        fig.update_scenes(yaxis_autorange="reversed")
    if x_reverse:
        fig.update_scenes(xaxis_autorange="reversed")
    if z_reverse:
        fig.update_scenes(zaxis_autorange="reversed")
    if show:
        fig.show()
    if save:
        fig.write_html(save)
    if return_fig:
        return fig

def scatter3ds(adata, groupbys, outpre=None,  **kargs):
    for i in groupbys:
        save = None if outpre is None else '%s.%s.3d.html'%(outpre, i.replace(' ','.'))
        scatter3d(adata, i, save=save, **kargs)

def scatters(adata, groupbys, basis='X_umap', use_raw=False,
               cmap = None, matches = None, 
                line_color='gray',  line_width=1, line_alpha = 1, color='b',
                line_weight = None, line_cmap = None,
                out=None, outpre=None, show=True, ncols=2,
                figscale=400, werror=30,  width =None, height=None,
                aspectmode='data', shared_xaxes=True, shared_yaxes=True, 
                aspectratio=dict(x=1, y=1, z=1), y_reverse=False, x_reverse=False,
                xyz = [0,1,2],
                clips = None, image_opacity=None,
                vmin = None, vmax = None, cmid=None,
                error=10, scale=1, 
                legendwscape=0.1, lengedxloc = 1.05, keep_all=False,
                ascending=False, return_fig =False, legend_tracegroupgap = 25, legend_font_size=14,
                thickness=20, cblen=0.5, cby=0.5, ticks='outside', tickmode='auto', 
                template='none',
                itemwidth = None,
                order ='guess', soft=False, 
                clickmode='event+select',
                same_scale=True, 
                show_grid =True, 
                showticklabels = False,
                legend_groupclick='toggleitem',
                legend_itemclick='toggle',
                legend_itemdoubleclick='toggleothers',
                show_image=False,
                imageaspos=False, 
                sample=None,

                margin=None,
                showlegend=True, 
                subplot_dict = {}, 
                dargs={},
                **kargs):
    Fd = Fatch(adata = adata,
                    basis=basis,
                    groupby=groupbys, use_raw=use_raw,
                    cmap = cmap, sample=sample, **dargs)
    
    groups= Fd.groupby
    clips = None if clips is None else list(clips)
    idata = Fd.data.copy()
    idata['index'] = idata.index

    ncell = len(groups)
    nrows, ncols = colrows(ncell, ncols=ncols, soft=soft)
    width = ncols*figscale+werror  if width is None else width
    height= nrows*figscale if height is None else height


    if Fd.dims==2:
        GoS = go.Scatter
        fig = make_subplots(rows=nrows, cols=ncols, 
                            shared_xaxes=shared_xaxes, 
                            shared_yaxes=shared_yaxes, 
                            print_grid =False,
                            subplot_titles=groups,
                            **subplot_dict)
    elif Fd.dims==3:
        GoS = go.Scatter3d
        specs = np.array([{"type": "scene"},]*(nrows * ncols)).reshape(nrows, ncols)
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=groups, specs=specs.tolist())

    legendps = lengedxloc+legendwscape if 'discrete' in list(Fd.gdtype.values()) else lengedxloc

    for n,group in enumerate(groups):
        irow, icol = n//ncols+1, n%ncols+1
        ctype = Fd.gdtype[group]
        color = Fd.colors[group]
        colors = pxsetcolor(color, ctype=ctype)

        # plot
        if ctype == 'discrete':
            order = idata[group].cat.categories.tolist()
            if not keep_all:
                idata[group] = idata[group].cat.remove_unused_categories()
                keep_order = idata[group].cat.categories.tolist()
                if type(color)==list:
                    color =[c for c,o in zip(color, order) if o in keep_order]
                order = keep_order
            cordict = dict(zip(order,color))
            for _n in order:
                iidata  = idata.loc[(idata[group]==_n),:]
                dimdict = { i[0]: iidata[i[1]] for i in zip(list('xyz'), Fd.dimsname) }
                dimdict.update({'name':_n, 'legendgrouptitle':{'text':group, 'font': {'size':14}}, 'legendgroup' : str(n+1)})
                #if Colors:
                dimdict.update({'marker': dict(color=cordict[_n],
                                                size=scale, 
                                                line=dict(width=0,color='DarkSlateGrey'))})
                fig.append_trace(GoS(mode="markers", showlegend=showlegend,  **dimdict, **kargs), 
                                 row=irow, col=icol)
        elif ctype == 'continuous':
            if np.ndim(clips)==1:
                idata[group] = np.clip(idata[group], clips[0], clips[1])
            elif np.ndim(clips) > 1:
                idata[group] = np.clip(idata[group], clips.pop(0)[0], clips.pop(0)[1])

            if not vmin is None:
                idata = idata[idata[group]>=vmin]
            else:
                vmin = idata[group].min()

            if not vmax is None:
                idata[group] = idata[group].clip(None, vmax)
            else:
                vmax = idata[group].max()

            if order =='guess' or  order == True:
                idata.sort_values(by = group, ascending=ascending, inplace=True)
            dimdict = { i[0]: idata[i[1]] for i in zip(list('xyz'), Fd.dimsname)}
            dimdict.update({'name': group, 'legendgroup' : str(n+1)})
            colorscale = colors['color_continuous_scale'] if cmap is None else cmap
            colorbar=dict(thickness=thickness, title=group,
                        len=cblen, x=legendps,y=cby, 
                        tickmode=tickmode,
                        ticks= ticks,
                        outlinewidth=0)

            dimdict.update({'marker': dict(colorscale=colorscale, 
                                            showscale=showlegend,
                                            color=idata[group],
                                            size=scale,
                                            cmin=vmin,cmax=vmax,
                                            cmid=cmid,
                                            line=dict(width=0,color='DarkSlateGrey'),
                                            colorbar=colorbar)})
            fig.append_trace(GoS(mode="markers", showlegend=False, 
                                 marker_coloraxis=None, 
                                #  hoverlabel =idata['index'],
                                    # hover_name="index",
                                    # hover_data=["index"],
                                    **dimdict, **kargs), row=irow, col=icol)
            legendps += legendwscape

        if same_scale:
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                row=irow, col=icol,
            )
        fig.update_scenes(aspectmode=aspectmode, 
                            aspectratio=aspectratio,
                            row=irow, col=icol)
    # fig.update_traces(marker=dict(size=scale, line=dict(width=0,color='DarkSlateGrey')),
    #                     showlegend=showlegend,
    #                     selector=dict(mode='markers'))

    if not matches is None:
        assert matches.shape[1] >= 6
        if not line_weight is None:
            line_weight = np.array(line_weight)
            line_weight = line_weight/line_weight.max()
            line_widths = line_weight * line_width
        else:
            line_widths = np.ones(matches.shape[0]) * line_width

        if not line_weight is None:
            line_colors = cmaptolist(line_cmap, spaces=line_weight)
        else:
            line_colors = [line_color] * matches.shape[0]

        XYZ = np.array([matches[:,[0,3]], matches[:,[1,4]], matches[:,[2,5]]])[Fd.axes]
        for i in range(matches.shape[0]):
            fig.append_trace(GoS( x=XYZ[0][i], y=XYZ[1][i], z=XYZ[2][i], 
                                mode="lines", showlegend = False,
                                line={'color': line_color, 
                                        'width': line_width},),
                                row=irow, col=icol)

    fig.update_layout(
                        height=height, width=width,
                        #showlegend=showlegend,
                        #scene_aspectmode=aspectmode,
                        #scene_aspectratio=aspectratio,
                        legend=dict( itemsizing = 'constant', 
                                     itemwidth = (itemwidth or 30+len(groups)) ),
                        # legendwidth =
                        legend_tracegroupgap = legend_tracegroupgap,
                        legend_groupclick=legend_groupclick,
                        legend_itemclick=legend_itemclick,
                        legend_itemdoubleclick=legend_itemdoubleclick,
                        legend_font_size=legend_font_size,
                        clickmode=clickmode,
                        template=template,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            aspectmode=aspectmode, 
                            aspectratio=aspectratio,
                        ),
                        #autosize=False,
                        margin=margin,
                        plot_bgcolor='#FFFFFF',) 
                        #template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)

    if show_image and (not Fd.image is None):
        if imageaspos:
            Fd.add_image(fig, Fd.image, image_opacity=image_opacity,)
        else:
            Fd.add_layout_image(fig, Fd.image,
                        image_opacity=image_opacity,
                        x_reverse=x_reverse,
                        y_reverse=y_reverse)

    #fig = go.FigureWidget(fig)
    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")

    if show:
        fig.show()
    if outpre or out :
        out = '%s.%s.%sd.html'%(outpre, '.'.join(groups), Fd.dims) if outpre else out
        fig.write_html(out)
    if return_fig:
        return fig

def qscatter(Data, X=None, Y=None, Z=None, group=None, save=None, show=True, scale=1,
                xwidth=800, ywidth=800, zwidth=800, scene_aspectmode='data',
                scene_aspectratio=dict(x=1, y=1, z=1), clip=None, sample=None,
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                colormap=None, template='none', **kargs):
    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data, columns=list('xyz'))
    if (X is None) and (Y is None) and (Z is None):
        X, Y, Z = Data.columns[:3].tolist()

    Data['index'] = Data.index

    if isinstance(group, pd.Series) or \
        isinstance(group, pd.core.arrays.categorical.Categorical) or \
        isinstance(group, np.ndarray) or \
        isinstance(group, list):
        try:
            Order = group.cat.categories
            Data['group'] = pd.Categorical(np.array(group), categories=Order)
        except:
            Data['group'] = np.array(group)
        group = 'group'
    if not sample is None:
        if (type(sample) == int) and (sample>1):
            Data = Data.sample(n=sample, 
                                    replace=False,
                                    random_state=random_state)
        if (type(sample) == float) and (sample<=1):
            Data = Data.sample(frac=sample, 
                                    replace=False,
                                    random_state=random_state)

    dimdict = dict(zip(list('xyz'), (X,Y,Z)))

    if not group is None:
        ctype   = vartype(Data[group])
        dimdict['color']=group
        if ctype == 'discrete':
            try:
                order = Data[group].cat.categories.tolist()
            except:
                order = Data[group].unique().tolist()
            color_seq = 'color_discrete_sequence'
            colormap = color_palette(len(order))
            category_orders={group: order, color_seq : colormap}
            dimdict.update({'category_orders': category_orders}) #'animation_frame': group
        elif ctype == 'continuous':
            if not clip is None:
                Data[group] = np.clip(Data[group], clip[0], clip[1])
            if order =='guess' or  order == True:
                Data = Data.sort_values(by = group, ascending=True)
            color_seq = 'color_continuous_scale'
            colormap = 'Viridis' if colormap is None else colormap
            dimdict.update({color_seq:colormap})

    fig = px.scatter_3d(Data, hover_name="index", hover_data=["index"],
                         **dimdict, **kargs) #width=width, height=height,
    fig.update_layout(legend=dict(itemsizing = 'constant'),
                    scene_aspectmode=scene_aspectmode,
                        scene_aspectratio=scene_aspectratio,
                        template=template,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                        plot_bgcolor='#FFFFFF',) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_traces(marker=dict(size=scale,
                        line=dict(width=0,color='DarkSlateGrey')),
                        selector=dict(mode='markers'))

    fig.update_traces(#hovertemplate="Sepal Width: %{x}<br>Sepal Length: %{y}<br>%{text}<br>Petal Width: %{customdata[1]}",
                    text=[{"index": Data.index}])

    if save:
        fig.write_html(save)
    if show:
        fig.show()
    else:
        return fig

def qscatter2d(Data, X=None, Y=None, group=None, save=None, show=True, 
               scale=1,  same_scale=True, 
               size=None, opacity=None,
               sample=None,
               figscale=1000, werror=100, 
                width = None, height=None, scene_aspectmode='data',  
                reverse_y=False, reverse_x=False,  
                showlegend=True,
                scene_aspectratio=dict(x=1, y=1, z=1), clip=None,
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                colordict = {},
                render_mode='auto', 
                colormap=None, template='none',
                itemsizing='constant',
                itemwidth=None,
                legdict={},
                  **kargs):
    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data, columns=list('xy'))
    if (X is None) and (Y is None):
        X, Y = Data.columns[:2].tolist()

    Data['index'] = Data.index

    if isinstance(group, pd.Series) or \
        isinstance(group, pd.core.arrays.categorical.Categorical) or \
        isinstance(group, np.ndarray) or \
        isinstance(group, list):
        try:
            Order = group.cat.categories
            Data['group'] = pd.Categorical(np.array(group), categories=Order)
        except:
            Data['group'] = np.array(group)
        group = 'group'
    if not sample is None:
        if (type(sample) == int) and (sample>1):
            Data = Data.sample(n=sample, 
                                    replace=False,
                                    random_state=random_state)
        if (type(sample) == float) and (sample<=1):
            Data = Data.sample(frac=sample, 
                                    replace=False,
                                    random_state=random_state)

    dimdict = dict(zip(list('xy'), (X,Y)))

    if not group is None:
        ctype   = vartype(Data[group])
        dimdict['color']=group
        if ctype == 'discrete':
            try:
                order = Data[group].cat.categories.tolist()
            except:
                order = Data[group].unique().tolist()

            colormap = colordict.get(group, color_palette(len(order)))
            category_orders={group: order}
            
            dimdict.update({'category_orders': category_orders, 
                            'color_discrete_sequence':colormap}) #'animation_frame': group
        elif ctype == 'continuous':
            if not clip is None:
                Data[group] = np.clip(Data[group], clip[0], clip[1])
            if order =='guess' or  order == True:
                Data = Data.sort_values(by = group, ascending=True)
            colormap = 'Viridis' if colormap is None else colormap
            dimdict.update({'color_continuous_scale':colormap})

    width = figscale+werror  if width is None else width
    height= figscale if height is None else height
    fig = px.scatter(Data, hover_name="index", hover_data=["index"],
                     width=width, height=height,
                     render_mode=render_mode,
                    **dimdict, **kargs) 
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    if reverse_x:
        fig.update_xaxes(autorange="reversed")

    itemwidth or 30
    fig.update_layout(#legend=dict(itemsizing = 'constant'),
                    legend=dict( itemsizing = itemsizing, itemwidth=itemwidth, **legdict),
                        showlegend=showlegend,
                        scene_aspectmode=scene_aspectmode,
                        template=template,
                        scene_aspectratio=scene_aspectratio,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                        plot_bgcolor='#FFFFFF',) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)

    fig.update_traces(text=[{"index": Data.index}])
    if not size is None:
        fig.update_traces(marker=dict(opacity=opacity, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    else:
        fig.update_traces(marker=dict(opacity=opacity, size=scale, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    if same_scale:
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

    if save:
        fig.write_html(save)
    if show:
        fig.show()
    else:
        return fig
