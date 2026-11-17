"""Plotly scatter utilities with configurable light/dark 3D themes."""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc
from plotly.subplots import make_subplots

from ..plotting._colors import *
from ..plotting._spatial3d import get_spatial_info
from ..plotting._utilis import colrows, image2batyes
from ..utilis._arrays import list_iter, vartype


class Fatch:
    def __init__(
        self,
        adata=None,
        basis="X_umap",
        groupby=None,
        splitby=None,
        use_raw=False,
        dimsname=None,
        obs=None,
        cmap=None,
        palette=None,
        sample=None,
        seed=200504,
        axes=None,
    ):
        np.random.seed(seed)

        self.adata = adata
        self.groupby = groupby
        self.splitby = splitby
        self.fetchdata(
            use_raw=use_raw,
            basis=basis,
            dimsname=dimsname,
            cmap=cmap,
            palette=palette,
            sample=sample,
            obs=obs,
            axes=axes,
        )

    def fetchdata(
        self,
        use_raw=False,
        basis="X_umap",
        dimsname=None,
        cmap=None,
        palette=None,
        axes=None,
        sample=None,
        obs=None,
        **kargs,
    ):
        adata = self.adata
        if adata is not None and use_raw:
            adataX = adata.raw.to_adata()
        else:
            adataX = adata

        try:
            corr_df = pd.DataFrame(
                adata.obsm[basis],
                index=adata.obs_names,
            ).copy()
            corr_df.columns = [
                f"{basis.removeprefix('X_')}_{i}"
                for i in range(corr_df.shape[1])
            ]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "Please input a valid basis name."
            ) from error

        if axes is not None:
            self.dims = len(axes)
            self.axes = axes
        else:
            self.dims = corr_df.shape[1]
            self.axes = range(self.dims)

        if isinstance(self.groupby, (str, int, float, bool)):
            requested_groups = [self.groupby]
        elif self.groupby is None:
            requested_groups = []
        else:
            requested_groups = list(self.groupby)

        groups = list(dict.fromkeys(requested_groups))
        if self.splitby is not None:
            groups.append(self.splitby)

        if obs is not None:
            if isinstance(obs, str) and obs == "_ALL":
                obs = [i for i in adata.obs.columns if i not in groups]
                groups += list(obs)
            elif isinstance(obs, (str, int, float, bool)):
                groups.append(obs)
            else:
                groups += list(obs)

        var_groups = adataX.var_names.intersection(groups).tolist()
        obs_groups = adataX.obs.columns.intersection(groups).tolist()

        group_df = []
        if len(obs_groups) > 0:
            group_df.append(adataX.obs[obs_groups])
        if len(var_groups) > 0:
            var_arr = adataX[:, var_groups].X
            if hasattr(var_arr, "toarray"):
                var_arr = var_arr.toarray()
            var_arr = pd.DataFrame(
                var_arr,
                index=adataX.obs_names,
                columns=var_groups,
            )
            group_df.append(var_arr)

        assert len(group_df) > 0, "No group was found in adata."
        group_df = pd.concat(group_df, axis=1)
        groups = [i for i in groups if i in group_df.columns]
        group_df = group_df[groups]

        dimsname = corr_df.columns.tolist() if dimsname is None else dimsname
        corr_df.columns = dimsname

        colors = {}
        gdtype = {}
        for col in group_df.columns:
            igroup = group_df[col]
            gtype = vartype(igroup)

            if gtype == "discrete":
                if isinstance(igroup.dtype, pd.CategoricalDtype):
                    iorders = igroup.cat.categories.tolist()
                else:
                    iorders = igroup.unique().tolist()

                group_df[col] = pd.Categorical(
                    igroup,
                    categories=iorders,
                )

                if isinstance(palette, (list, np.ndarray)):
                    icolor = palette
                elif isinstance(palette, dict):
                    icolor = palette[col]
                elif isinstance(cmap, dict):
                    icolor = cmap[col]
                else:
                    if adata is None:
                        icolor = color_palette(len(iorders))
                    else:
                        iuns_cor = f"{col}_colors"
                        if (
                            iuns_cor in adata.uns
                            and len(iorders) <= len(adata.uns[iuns_cor])
                        ):
                            icolor = adata.uns[iuns_cor]
                        else:
                            adata.obs[col] = pd.Categorical(
                                igroup,
                                categories=iorders,
                            )
                            adata_color(
                                adata,
                                value=col,
                                cmap=cmap,
                                palette=palette,
                            )
                            icolor = adata.uns[iuns_cor]
                            adata.uns[iuns_cor] = icolor
            else:
                if isinstance(cmap, str):
                    icolor = cmap
                elif isinstance(cmap, dict):
                    icolor = cmap[col]
                elif cmap is None:
                    icolor = cmap1px
                else:
                    icolor = cmap

            colors[col] = icolor
            gdtype[col] = gtype

        if sample is not None:
            if 0 <= sample <= 1:
                isize = int(adata.shape[0] * sample)
            else:
                assert isinstance(sample, int)
                isize = sample

            isize = min(isize, adata.shape[0])
            obs_index = np.random.choice(
                adata.obs_names,
                size=isize,
                replace=False,
                p=None,
            )
            corr_df = corr_df.loc[obs_index, :]
            group_df = group_df.loc[obs_index, :]

        data = pd.concat([corr_df, group_df], axis=1)

        self.dimsname = dimsname
        self.dims = corr_df.shape[1]
        self.colors = colors
        self.gdtype = gdtype
        self.corr_df = corr_df
        self.group_df = group_df
        self.groupby = [i for i in requested_groups if i in groups]
        self.data = data
        self.data["index"] = data.index
        self.ngroup = len(self.groupby)
        return self

    def fetchimages(
        self,
        images=None,
        img_key="hires",
        basis=None,
        library_id=None,
        rescale=None,
    ):
        if images is None:
            adata = self.adata
            basis = "spatial" if basis is None else basis
            base_rescale = 1 if rescale is None else rescale

            library_ids = (
                list(adata.uns[basis].keys())
                if library_id is None
                else library_id
            )
            if isinstance(library_ids, (str, int)):
                library_ids = [library_ids]

            images = []
            for lid in library_ids:
                try:
                    img_dict = adata.uns[basis][lid]
                    iimg = img_dict["images"][img_key]
                    scale_factor = img_dict["scalefactors"].get(
                        f"tissue_{img_key}_scalef",
                        1,
                    )
                    img_rescale = scale_factor * base_rescale

                    if (
                        not np.allclose(
                            img_rescale,
                            0.9999,
                            rtol=1e-4,
                            atol=1e-8,
                        )
                        and img_rescale != 0
                    ):
                        import skimage.transform as skitf

                        if iimg.ndim == 2:
                            iimg = skitf.rescale(iimg, 1 / img_rescale)
                        else:
                            iimg = skitf.rescale(
                                iimg,
                                (1 / img_rescale, 1 / img_rescale, 1),
                            )
                        iimg = np.uint8(iimg * 255)
                except Exception:
                    print(
                        "Warning: no image found for library_id:",
                        lid,
                    )
                    iimg = None
                images.append(iimg)

        return images

    @staticmethod
    def add_layout_images(
        fig,
        images,
        ncols,
        x_reverse=False,
        y_reverse=False,
        sizing="stretch",
        image_opacity=None,
        xanchor="left",
        yanchor="bottom",
        **kargs,
    ):
        if images is not None and len(images) > 0:
            ncells = len(images)
            nrows, ncols = colrows(ncells, ncols=ncols, soft=False)
            idx_arr = np.arange(nrows * ncols).reshape(
                nrows,
                ncols,
                order="C",
            )[::-1, :]

            for i, image in enumerate(images):
                if image is None:
                    continue

                sizey, sizex = image.shape[:2]
                x = sizex if x_reverse else 0
                y = sizey if y_reverse else 0
                imagebt = image2batyes(image)
                row = i // ncols
                col = i % ncols
                xy_idx = idx_arr[row, col] + 1

                if xy_idx == 1:
                    ix, iy = "x", "y"
                else:
                    ix, iy = f"x{xy_idx}", f"y{xy_idx}"

                fig.add_layout_image(
                    source=imagebt,
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
                    opacity=image_opacity,
                    **kargs,
                )

    @staticmethod
    def add_layout_image(
        fig,
        image,
        x_reverse=False,
        y_reverse=False,
        sizing="stretch",
        image_opacity=None,
        xanchor="left",
        yanchor="bottom",
        **kargs,
    ):
        if image is not None:
            sizey, sizex = image.shape[:2]
            x = sizex if x_reverse else 0
            y = sizey if y_reverse else 0
            imagebt = image2batyes(image)
            xy_axes = {
                (ifig["xaxis"], ifig["yaxis"])
                for ifig in fig.data
            }

            for ix, iy in xy_axes:
                fig.add_layout_image(
                    source=imagebt,
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
                    opacity=image_opacity,
                    **kargs,
                )

    @staticmethod
    def add_image(fig, images, image_opacity=None, **kargs):
        if images is not None and len(images) > 0:
            xyaxis = []
            for i, ifig in enumerate(fig.data):
                ix, iy = ifig["xaxis"], ifig["yaxis"]
                if len(images) < i + 1:
                    continue

                image = images[i]
                if image is None:
                    continue
                imagebt = image2batyes(image)

                if [ix, iy] not in xyaxis:
                    xyaxis.append([ix, iy])
                    fig.add_image(
                        source=imagebt,
                        xaxis=ix,
                        yaxis=iy,
                        opacity=image_opacity,
                        **kargs,
                    )


def _resolve_plotly_theme(
    theme="white",
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
):
    theme = "white" if theme is None else str(theme).lower()

    if theme in ("black", "dark", "plotly_dark"):
        default_bg = "#000000"
        default_fg = "#FFFFFF"
        default_grid = "#3A3A3A"
    elif theme in ("white", "light", "plotly_white"):
        default_bg = "#FFFFFF"
        default_fg = "#222222"
        default_grid = "#D0D0D0"
    else:
        raise ValueError(
            "theme must be 'white'/'light' or 'black'/'dark'."
        )

    plot_bgcolor = default_bg if plot_bgcolor is None else plot_bgcolor
    paper_bgcolor = default_bg if paper_bgcolor is None else paper_bgcolor
    scene_bgcolor = plot_bgcolor if scene_bgcolor is None else scene_bgcolor
    fontcolor = default_fg if fontcolor is None else fontcolor
    axiscolor = default_fg if axiscolor is None else axiscolor
    gridcolor = default_grid if gridcolor is None else gridcolor

    return (
        gridcolor,
        axiscolor,
        scene_bgcolor,
        plot_bgcolor,
        paper_bgcolor,
        fontcolor,
    )


def _set_plotly_2d_theme(
    fig,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
):
    """Apply a consistent light or dark style to all 2D axes."""
    (
        gridcolor,
        axiscolor,
        _,
        plot_bgcolor,
        paper_bgcolor,
        fontcolor,
    ) = _resolve_plotly_theme(
        theme=theme,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
    )
    axis_ticks = "" if axis_ticks is None else axis_ticks

    axis_style = dict(
        visible=show_axes,
        showgrid=show_grid,
        showticklabels=showticklabels,
        ticks=axis_ticks,
        showline=show_axisline,
        zeroline=show_zeroline,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        linecolor=axiscolor,
        tickcolor=axiscolor,
        color=axiscolor,
    )
    if not show_axis_titles:
        axis_style["title"] = dict(text="")

    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_layout(
        font=dict(color=fontcolor),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
    )
    return fig


def _set_plotly_3d_theme(
    fig,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    scene_aspectmode="data",
    scene_aspectratio=None,
):
    """Apply a consistent light or dark style to a Plotly 3D scene."""
    (
        gridcolor,
        axiscolor,
        scene_bgcolor,
        plot_bgcolor,
        paper_bgcolor,
        fontcolor,
    ) = _resolve_plotly_theme(
        theme=theme,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        scene_bgcolor=scene_bgcolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
    )
    axis_ticks = "" if axis_ticks is None else axis_ticks

    axis_style = dict(
        visible=show_axes,
        showgrid=show_grid,
        showticklabels=showticklabels,
        ticks=axis_ticks,
        showline=show_axisline,
        zeroline=show_zeroline,
        showspikes=showspikes,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        linecolor=axiscolor,
        tickcolor=axiscolor,
        color=axiscolor,
        showbackground=True,
        backgroundcolor=scene_bgcolor,
    )

    if not show_axis_titles:
        axis_style["title"] = dict(text="")

    scene = dict(
        bgcolor=scene_bgcolor,
        aspectmode=scene_aspectmode,
        xaxis=axis_style.copy(),
        yaxis=axis_style.copy(),
        zaxis=axis_style.copy(),
    )
    if scene_aspectratio is not None:
        scene["aspectratio"] = scene_aspectratio

    fig.update_scenes(**scene)
    fig.update_layout(
        font=dict(color=fontcolor),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
    )
    return fig


def splitscatter2d(
    adata,
    groupby,
    splitby,
    basis="X_umap",
    use_raw=False,
    cmap=None,
    show_background=False,
    save=None,
    size=None,
    size_max=20,
    scale=1,
    show=True,
    template="none",
    scene_aspectmode="data",
    y_reverse=False,
    x_reverse=False,
    img_x_reverse=False,
    img_y_reverse=False,
    same_scale=True,
    scene_aspectratio=dict(x=1, y=1),
    return_fig=False,
    ncols=2,
    soft=False,
    figscale=900,
    werror=20,
    width=None,
    height=None,
    clips=None,
    vmin=None,
    vmax=None,
    showlegend=True,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    itemsizing="constant",
    itemwidth=None,
    show_image=False,
    sample=None,
    opacity=None,
    render_mode="auto",
    image_opacity=None,
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
    img_basis=None,
    library_id=None,
    legdict={},
    dargs={},
    **kargs,
):
    Fd = Fatch(
        adata=adata,
        basis=basis,
        groupby=groupby,
        splitby=splitby,
        use_raw=use_raw,
        cmap=cmap,
        sample=sample,
        **dargs,
    )
    data_df = Fd.data.copy()
    Order = data_df[splitby].cat.categories.tolist()
    nrows, ncols = colrows(len(Order), ncols=ncols, soft=soft)
    dimdict = dict(zip(list("xy"), Fd.dimsname))

    clips = None if clips is None else list(clips)
    width = ncols * figscale + werror if width is None else width
    height = nrows * figscale if height is None else height

    category_orders = None
    color_discrete_sequence = None
    color_continuous_scale = None

    gtype = Fd.gdtype[groupby]
    if gtype == "discrete":
        category_orders = data_df[groupby].cat.categories.tolist()
        color_discrete_sequence = Fd.colors[groupby]
    elif gtype == "continuous":
        color_continuous_scale = Fd.colors[groupby]
        cmin = data_df[groupby].min()

        if vmin is not None:
            data_df = data_df.loc[data_df[groupby] >= vmin, :]
        if vmax is not None:
            data_df[groupby] = data_df[groupby].clip(None, vmax)
        if clips is not None:
            if np.ndim(clips) == 1:
                data_df[groupby] = np.clip(
                    data_df[groupby],
                    clips[0],
                    clips[1],
                )
            elif np.ndim(clips) > 1:
                iclip = clips.pop(0)
                data_df[groupby] = np.clip(
                    data_df[groupby],
                    iclip[0],
                    iclip[1],
                )

        cmin = data_df[groupby].min()
        cmax = data_df[groupby].max()
        dimdict["range_color"] = [float(cmin), float(cmax)]

    data_df.sort_values(
        by=groupby,
        ascending=False,
        inplace=True,
    )
    data_df["index"] = data_df.index

    facet_orders = {splitby: Order}
    if category_orders is not None:
        facet_orders[groupby] = category_orders

    fig = px.scatter(
        data_df,
        color=groupby,
        facet_col=splitby,
        facet_col_wrap=ncols,
        facet_col_spacing=facet_col_spacing,
        facet_row_spacing=facet_row_spacing,
        size=size,
        size_max=size_max,
        render_mode=render_mode,
        width=width,
        height=height,
        color_discrete_sequence=color_discrete_sequence,
        color_continuous_scale=color_continuous_scale,
        hover_name="index",
        hover_data=["index"],
        category_orders=facet_orders,
        **dimdict,
        **kargs,
    )

    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")

    if show_image:
        images = Fd.fetchimages(
            images=images,
            img_key=img_key,
            basis=img_basis or basis,
            library_id=Order,
        )
        Fd.add_layout_images(
            fig,
            images,
            ncols,
            image_opacity=image_opacity,
            xanchor="right" if x_reverse else "left",
            yanchor="top" if y_reverse else "bottom",
            x_reverse=img_x_reverse,
            y_reverse=img_y_reverse,
        )

    itemwidth = itemwidth or 30
    fig.update_layout(
        legend=dict(
            itemsizing=itemsizing,
            itemwidth=itemwidth,
            **legdict,
        ),
        coloraxis_colorbar=dict(
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
            xaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
        ),
        plot_bgcolor="#FFFFFF",
    )

    _set_plotly_2d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
    )

    if size is not None:
        fig.update_traces(
            marker=dict(
                opacity=opacity,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )
    else:
        fig.update_traces(
            marker=dict(
                opacity=opacity,
                size=scale,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )

    if same_scale:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if show:
        fig.show()
    if save:
        fig.write_html(save)
    if return_fig:
        return fig


def scatter2ds(
    adata,
    groupbys=None,
    basis="X_umap",
    use_raw=False,
    cmap=None,
    save=None,
    outpre=None,
    size=None,
    size_max=20,
    scale=1,
    show=True,
    template="none",
    scene_aspectmode="data",
    render_mode="auto",
    y_reverse=False,
    x_reverse=False,
    img_x_reverse=False,
    img_y_reverse=False,
    same_scale=True,
    scene_aspectratio=dict(x=1, y=1),
    return_fig=False,
    ncols=2,
    soft=False,
    figscale=900,
    werror=20,
    width=None,
    height=None,
    clips=None,
    vmin=None,
    vmax=None,
    showlegend=True,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    itemsizing="constant",
    itemwidth=None,
    show_image=False,
    sample=None,
    opacity=None,
    image_opacity=None,
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
    img_basis=None,
    library_id=None,
    legdict={},
    dargs={},
    **kargs,
):
    Fd = Fatch(
        adata=adata,
        basis=basis,
        groupby=groupbys,
        use_raw=use_raw,
        cmap=cmap,
        sample=sample,
        **dargs,
    )

    nrows, ncols = colrows(Fd.ngroup, ncols=ncols, soft=soft)
    dimdict = dict(zip(list("xy"), Fd.dimsname))
    data_df = Fd.data.copy()

    clips = None if clips is None else list(clips)
    width = ncols * figscale + werror if width is None else width
    height = nrows * figscale if height is None else height

    category_orders = []
    color_discrete_sequence = []
    dis_groups = []
    con_groups = []
    continuous_cmap = None

    for col in Fd.groupby:
        gtype = Fd.gdtype[col]
        if gtype == "discrete":
            category_orders.extend(
                data_df[col].cat.categories.tolist()
            )
            color_discrete_sequence.extend(Fd.colors[col])
            dis_groups.append(col)
        elif gtype == "continuous":
            continuous_cmap = Fd.colors[col]
            con_groups.append(col)

    assert not (len(dis_groups) > 0 and len(con_groups) > 0)

    SData = pd.melt(
        data_df,
        id_vars=Fd.dimsname,
        value_vars=dis_groups + con_groups,
        var_name="groups",
        value_name="Type",
    )

    if len(con_groups) > 0:
        if vmin is not None:
            SData = SData.loc[SData["Type"] >= vmin, :]
        if vmax is not None:
            SData["Type"] = SData["Type"].clip(None, vmax)
        if clips is not None:
            if np.ndim(clips) == 1:
                SData["Type"] = np.clip(
                    SData["Type"],
                    clips[0],
                    clips[1],
                )
            elif np.ndim(clips) > 1:
                iclip = clips.pop(0)
                SData["Type"] = np.clip(
                    SData["Type"],
                    iclip[0],
                    iclip[1],
                )

    SData.sort_values(by="Type", ascending=False, inplace=True)
    SData["index"] = SData.index

    fig = px.scatter(
        SData,
        color="Type",
        facet_col="groups",
        facet_col_wrap=ncols,
        facet_col_spacing=facet_col_spacing,
        facet_row_spacing=facet_row_spacing,
        size=size,
        size_max=size_max,
        render_mode=render_mode,
        width=width,
        height=height,
        color_discrete_sequence=color_discrete_sequence,
        color_continuous_scale=continuous_cmap,
        hover_name="index",
        hover_data=["index"],
        category_orders={"Type": category_orders},
        **dimdict,
        **kargs,
    )

    itemwidth = itemwidth or (30 + len(dis_groups))
    fig.update_layout(
        legend=dict(
            itemsizing=itemsizing,
            itemwidth=itemwidth,
            **legdict,
        ),
        coloraxis_colorbar=dict(
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
            xaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
        ),
        plot_bgcolor="#FFFFFF",
    )

    _set_plotly_2d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
    )

    if size is not None:
        fig.update_traces(
            marker=dict(
                opacity=opacity,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )
    else:
        fig.update_traces(
            marker=dict(
                size=scale,
                opacity=opacity,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )

    if same_scale:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")

    if show_image:
        images = Fd.fetchimages(
            images=images,
            img_key=img_key,
            basis=img_basis or basis,
            library_id=library_id,
        )
        Fd.add_layout_image(
            fig,
            images[0],
            image_opacity=image_opacity,
            xanchor="right" if x_reverse else "left",
            yanchor="top" if y_reverse else "bottom",
            x_reverse=img_x_reverse,
            y_reverse=img_y_reverse,
        )

    if show:
        fig.show()
    if outpre or save:
        save = (
            outpre + "." + ".".join(dis_groups) + ".2d.html"
            if outpre
            else save
        )
        fig.write_html(save)
    if return_fig:
        return fig


def scatter3d(
    adata,
    groupby,
    basis="X_umap",
    use_raw=False,
    cmap=None,
    select=None,
    nacolor="lightgrey",
    show=True,
    xyz=[0, 1, 2],
    size=None,
    size_max=20,
    scale=1,
    width=None,
    height=None,
    scene_aspectmode="data",
    keep_all=False,
    order=True,
    ascending=False,
    return_fig=False,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    y_reverse=False,
    x_reverse=False,
    z_reverse=False,
    scene_aspectratio=dict(x=1, y=1, z=1),
    itemsizing="constant",
    template="none",
    save=None,
    vmin=None,
    vmax=None,
    clip=None,
    sample=None,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    thicknessmode="pixels",
    cbar_width=25,
    lenmode="pixels",
    cbar_height=300,
    cbar_yanchor="middle",
    cbar_ypos=0.5,
    cbar_ypad=10,
    ticks="outside",
    legdict={},
    dargs={},
    **kargs,
):
    Fd = Fatch(
        adata=adata,
        basis=basis,
        groupby=[groupby],
        use_raw=use_raw,
        cmap=cmap,
        sample=sample,
        **dargs,
    )

    if Fd.dims < 3:
        raise ValueError("The dims must be larger than 2!")

    dimdict = dict(zip(list("xyz"), np.array(Fd.dimsname)[xyz]))
    ctype = Fd.gdtype[groupby]
    color = Fd.colors[groupby]
    idata = Fd.data.copy()
    dimdict.update(pxsetcolor(color, ctype=ctype, cmap=cmap))

    if ctype == "discrete":
        category_order = idata[groupby].cat.categories.tolist()

        if not keep_all:
            idata[groupby] = idata[groupby].cat.remove_unused_categories()
            keep_order = idata[groupby].cat.categories.tolist()
            colors = dimdict["color_discrete_sequence"]

            if isinstance(colors, list):
                colors = [
                    c for c, o in zip(colors, category_order)
                    if o in keep_order
                ]
                dimdict["color_discrete_sequence"] = colors
            category_order = keep_order

        category_orders = {groupby: category_order}

        if select is not None:
            select = [select] if isinstance(select, str) else select
            if isinstance(dimdict["color_discrete_sequence"], list):
                colors = [
                    cs if co in select else nacolor
                    for co, cs in zip(
                        category_orders[groupby],
                        dimdict["color_discrete_sequence"],
                    )
                ]
                dimdict["color_discrete_sequence"] = colors

        dimdict["category_orders"] = category_orders

    elif ctype == "continuous":
        cmin = idata[groupby].min()

        if vmin is not None:
            idata = idata[idata[groupby] >= vmin]
        if vmax is not None:
            idata[groupby] = idata[groupby].clip(None, vmax)
        if clip is not None:
            idata[groupby] = np.clip(
                idata[groupby],
                clip[0],
                clip[1],
            )
        if order == "guess" or order is True:
            idata.sort_values(
                by=groupby,
                ascending=ascending,
                inplace=True,
            )

        cmax = idata[groupby].max()
        dimdict["range_color"] = [float(cmin), float(cmax)]

    fig = px.scatter_3d(
        idata,
        color=groupby,
        size=size,
        size_max=size_max,
        width=width,
        height=height,
        hover_name="index",
        hover_data=["index"],
        **dimdict,
        **kargs,
    )

    fig.update_layout(
        legend=dict(itemsizing=itemsizing, **legdict),
        coloraxis_colorbar=dict(
            thicknessmode=thicknessmode,
            thickness=cbar_width,
            lenmode=lenmode,
            len=cbar_height,
            yanchor=cbar_yanchor,
            y=cbar_ypos,
            ypad=cbar_ypad,
            ticks=ticks,
        ),
        template=template,
    )

    _set_plotly_3d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        showspikes=showspikes,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        scene_bgcolor=scene_bgcolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
        scene_aspectmode=scene_aspectmode,
        scene_aspectratio=scene_aspectratio,
    )

    markerdict = dict(
        marker={
            "line": {
                "width": 0,
                "color": "DarkSlateGrey",
            }
        },
        selector={"mode": "markers"},
    )
    if size is None:
        markerdict["marker"]["size"] = scale
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


def scatter3ds(
    adata,
    groupbys,
    outpre=None,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    **kargs,
):
    for groupby in groupbys:
        save = (
            None
            if outpre is None
            else "%s.%s.3d.html" % (
                outpre,
                groupby.replace(" ", "."),
            )
        )
        scatter3d(
            adata,
            groupby,
            save=save,
            theme=theme,
            show_axes=show_axes,
            show_grid=show_grid,
            showticklabels=showticklabels,
            axis_ticks=axis_ticks,
            show_axis_titles=show_axis_titles,
            show_axisline=show_axisline,
            show_zeroline=show_zeroline,
            showspikes=showspikes,
            gridwidth=gridwidth,
            gridcolor=gridcolor,
            axiscolor=axiscolor,
            scene_bgcolor=scene_bgcolor,
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            fontcolor=fontcolor,
            **kargs,
        )


def scatters(
    adata,
    groupbys,
    basis="X_umap",
    use_raw=False,
    cmap=None,
    matches=None,
    line_color="gray",
    line_width=1,
    line_alpha=1,
    color="b",
    line_weight=None,
    line_cmap=None,
    save=None,
    outpre=None,
    show=True,
    ncols=2,
    figscale=900,
    werror=30,
    width=None,
    height=None,
    aspectmode="data",
    shared_xaxes=True,
    shared_yaxes=True,
    aspectratio=dict(x=1, y=1, z=1),
    y_reverse=False,
    x_reverse=False,
    xyz=[0, 1, 2],
    clips=None,
    image_opacity=None,
    vmin=None,
    vmax=None,
    cmid=None,
    error=10,
    scale=1,
    legendwscape=0.1,
    lengedxloc=1.05,
    keep_all=False,
    ascending=False,
    return_fig=False,
    legend_tracegroupgap=25,
    legend_font_size=14,
    thickness=20,
    cblen=0.5,
    cby=0.5,
    ticks="outside",
    tickmode="auto",
    template="none",
    itemwidth=None,
    order="guess",
    soft=False,
    clickmode="event+select",
    same_scale=True,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    legend_groupclick="toggleitem",
    legend_itemclick="toggle",
    legend_itemdoubleclick="toggleothers",
    show_image=False,
    imageaspos=False,
    sample=None,
    margin=None,
    showlegend=True,
    subplot_dict={},
    dargs={},
    **kargs,
):
    Fd = Fatch(
        adata=adata,
        basis=basis,
        groupby=groupbys,
        use_raw=use_raw,
        cmap=cmap,
        sample=sample,
        **dargs,
    )

    groups = Fd.groupby
    clips = None if clips is None else list(clips)
    idata = Fd.data.copy()
    idata["index"] = idata.index

    ncell = len(groups)
    nrows, ncols = colrows(ncell, ncols=ncols, soft=soft)
    width = ncols * figscale + werror if width is None else width
    height = nrows * figscale if height is None else height

    if Fd.dims == 2:
        GoS = go.Scattergl
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            print_grid=False,
            subplot_titles=groups,
            **subplot_dict,
        )
    elif Fd.dims == 3:
        GoS = go.Scatter3d
        specs = np.array(
            [{"type": "scene"}] * (nrows * ncols)
        ).reshape(nrows, ncols)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=groups,
            specs=specs.tolist(),
        )
    else:
        raise ValueError("Only 2D and 3D coordinates are supported.")

    legendps = (
        lengedxloc + legendwscape
        if "discrete" in list(Fd.gdtype.values())
        else lengedxloc
    )

    for n, group in enumerate(groups):
        irow, icol = n // ncols + 1, n % ncols + 1
        ctype = Fd.gdtype[group]
        icolor = Fd.colors[group]
        colors = pxsetcolor(icolor, ctype=ctype)

        if ctype == "discrete":
            iorder = idata[group].cat.categories.tolist()

            if not keep_all:
                idata[group] = idata[group].cat.remove_unused_categories()
                keep_order = idata[group].cat.categories.tolist()
                if isinstance(icolor, list):
                    icolor = [
                        c for c, o in zip(icolor, iorder)
                        if o in keep_order
                    ]
                iorder = keep_order

            cordict = dict(zip(iorder, icolor))
            for category in iorder:
                iidata = idata.loc[idata[group] == category, :]
                dimdict = {
                    axis: iidata[name]
                    for axis, name in zip(
                        list("xyz"),
                        Fd.dimsname,
                    )
                }
                dimdict.update(
                    {
                        "name": category,
                        "legendgrouptitle": {
                            "text": group,
                            "font": {"size": 14},
                        },
                        "legendgroup": str(n + 1),
                        "marker": dict(
                            color=cordict[category],
                            size=scale,
                            line=dict(
                                width=0,
                                color="DarkSlateGrey",
                            ),
                        ),
                    }
                )
                fig.append_trace(
                    GoS(
                        mode="markers",
                        showlegend=showlegend,
                        **dimdict,
                        **kargs,
                    ),
                    row=irow,
                    col=icol,
                )

        elif ctype == "continuous":
            if clips is not None:
                if np.ndim(clips) == 1:
                    idata[group] = np.clip(
                        idata[group],
                        clips[0],
                        clips[1],
                    )
                elif np.ndim(clips) > 1:
                    iclip = clips.pop(0)
                    idata[group] = np.clip(
                        idata[group],
                        iclip[0],
                        iclip[1],
                    )

            ivmin = idata[group].min() if vmin is None else vmin
            ivmax = idata[group].max() if vmax is None else vmax

            if vmin is not None:
                idata = idata[idata[group] >= vmin]
            if vmax is not None:
                idata[group] = idata[group].clip(None, vmax)
            if order == "guess" or order is True:
                idata.sort_values(
                    by=group,
                    ascending=ascending,
                    inplace=True,
                )

            dimdict = {
                axis: idata[name]
                for axis, name in zip(
                    list("xyz"),
                    Fd.dimsname,
                )
            }
            dimdict.update(
                {
                    "name": group,
                    "legendgroup": str(n + 1),
                }
            )
            colorscale = (
                colors["color_continuous_scale"]
                if cmap is None
                else cmap
            )
            colorbar = dict(
                thickness=thickness,
                title=group,
                len=cblen,
                x=legendps,
                y=cby,
                tickmode=tickmode,
                ticks=ticks,
                outlinewidth=0,
            )
            dimdict["marker"] = dict(
                colorscale=colorscale,
                showscale=showlegend,
                color=idata[group],
                size=scale,
                cmin=ivmin,
                cmax=ivmax,
                cmid=cmid,
                line=dict(width=0, color="DarkSlateGrey"),
                colorbar=colorbar,
            )
            fig.append_trace(
                GoS(
                    mode="markers",
                    showlegend=False,
                    marker_coloraxis=None,
                    **dimdict,
                    **kargs,
                ),
                row=irow,
                col=icol,
            )
            legendps += legendwscape

        if same_scale:
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                row=irow,
                col=icol,
            )
        fig.update_scenes(
            aspectmode=aspectmode,
            aspectratio=aspectratio,
            row=irow,
            col=icol,
        )

    if matches is not None:
        assert matches.shape[1] >= 6
        if line_weight is not None:
            line_weight = np.asarray(line_weight)
            line_weight = line_weight / line_weight.max()
            line_widths = line_weight * line_width
            line_colors = cmaptolist(
                line_cmap,
                spaces=line_weight,
            )
        else:
            line_widths = np.ones(matches.shape[0]) * line_width
            line_colors = [line_color] * matches.shape[0]

        XYZ = np.array(
            [
                matches[:, [0, 3]],
                matches[:, [1, 4]],
                matches[:, [2, 5]],
            ]
        )[Fd.axes]

        for i in range(matches.shape[0]):
            line_data = dict(
                x=XYZ[0][i],
                y=XYZ[1][i],
                mode="lines",
                showlegend=False,
                line={
                    "color": line_colors[i],
                    "width": line_widths[i],
                },
                opacity=line_alpha,
            )
            if Fd.dims == 3:
                line_data["z"] = XYZ[2][i]
            fig.append_trace(
                GoS(**line_data),
                row=irow,
                col=icol,
            )

    fig.update_layout(
        height=height,
        width=width,
        legend=dict(
            itemsizing="constant",
            itemwidth=itemwidth or 30 + len(groups),
        ),
        legend_tracegroupgap=legend_tracegroupgap,
        legend_groupclick=legend_groupclick,
        legend_itemclick=legend_itemclick,
        legend_itemdoubleclick=legend_itemdoubleclick,
        legend_font_size=legend_font_size,
        clickmode=clickmode,
        template=template,
        scene=dict(
            xaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            aspectmode=aspectmode,
            aspectratio=aspectratio,
        ),
        margin=margin,
        plot_bgcolor="#FFFFFF",
    )

    if Fd.dims == 2:
        _set_plotly_2d_theme(
            fig,
            theme=theme,
            show_axes=show_axes,
            show_grid=show_grid,
            showticklabels=showticklabels,
            axis_ticks=axis_ticks,
            show_axis_titles=show_axis_titles,
            show_axisline=show_axisline,
            show_zeroline=show_zeroline,
            gridwidth=gridwidth,
            gridcolor=gridcolor,
            axiscolor=axiscolor,
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            fontcolor=fontcolor,
        )
    else:
        _set_plotly_3d_theme(
            fig,
            theme=theme,
            show_axes=show_axes,
            show_grid=show_grid,
            showticklabels=showticklabels,
            axis_ticks=axis_ticks,
            show_axis_titles=show_axis_titles,
            show_axisline=show_axisline,
            show_zeroline=show_zeroline,
            showspikes=showspikes,
            gridwidth=gridwidth,
            gridcolor=gridcolor,
            axiscolor=axiscolor,
            scene_bgcolor=scene_bgcolor,
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            fontcolor=fontcolor,
            scene_aspectmode=aspectmode,
            scene_aspectratio=aspectratio,
        )

    if show_image and hasattr(Fd, "image") and Fd.image is not None:
        if imageaspos:
            Fd.add_image(
                fig,
                Fd.image,
                image_opacity=image_opacity,
            )
        else:
            Fd.add_layout_image(
                fig,
                Fd.image,
                image_opacity=image_opacity,
                x_reverse=x_reverse,
                y_reverse=y_reverse,
            )

    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")

    if show:
        fig.show()
    if outpre or save:
        out = (
            "%s.%s.%sd.html"
            % (outpre, ".".join(groups), Fd.dims)
            if outpre
            else save
        )
        fig.write_html(out)
    if return_fig:
        return fig


def pview0(
    *data,
    X=None,
    Y=None,
    Z=None,
    group=None,
    group_diff=False,
    samples=None,
    random_state=None,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    **kargs,
):
    ncells = len(data)
    groupidx = np.repeat(
        np.arange(ncells),
        [idata.shape[0] for idata in data],
    )
    start, sampleidx = 0, []

    if samples is not None:
        if isinstance(samples, (float, int)):
            samples = [samples for _ in range(ncells)]
        elif isinstance(samples, list):
            samples = samples[:ncells]
        else:
            raise ValueError(
                "samples should be a number or a list of numbers"
            )

        rng = np.random.default_rng(random_state)
        for i in range(ncells):
            isize = data[i].shape[0]
            sample = samples[i]

            if sample is None or sample == 1:
                idx = np.arange(isize)
            elif sample > 1:
                if sample < 10:
                    print("warning: sample size is smaller than 10")
                idx = rng.choice(
                    isize,
                    min(int(sample), isize),
                    replace=False,
                )
            elif sample < 1:
                idx = rng.choice(
                    isize,
                    int(isize * sample),
                    replace=False,
                )
            else:
                idx = np.array([], dtype=int)

            sampleidx.append(idx + start)
            start += isize
        sampleidx = np.concatenate(sampleidx)

    if all(isinstance(idata, pd.DataFrame) for idata in data):
        datas = pd.concat(data, axis=0)
        if group is None:
            datas["group"] = groupidx.astype(str)
        else:
            datas["group"] = datas[group]
            if group_diff:
                datas["groupidx"] = groupidx
                datas["group"] = (
                    datas["group"].astype(str)
                    + "_"
                    + datas["groupidx"].astype(str)
                )
        if X is None and Y is None and Z is None:
            X, Y, Z = datas.columns[:3].tolist()
    else:
        datas = pd.DataFrame(
            np.vstack(data),
            columns=list("xyz"),
        )
        datas["group"] = groupidx.astype(str)
        X, Y, Z = list("xyz")

    if len(sampleidx) > 0:
        datas = datas.iloc[sampleidx, :]

    return qscatter(
        datas,
        X=X,
        Y=Y,
        Z=Z,
        group="group",
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        showspikes=showspikes,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        scene_bgcolor=scene_bgcolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
        **kargs,
    )


def pview(
    *data,
    X=None,
    Y=None,
    Z=None,
    group=None,
    group_diff=False,
    samples=None,
    random_state=None,
    order=True,
    keep_all=False,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    **kargs,
):
    ncells = len(data)
    groupidx = np.repeat(
        np.arange(ncells),
        [idata.shape[0] for idata in data],
    )

    start, sampleidx = 0, []
    if samples is not None:
        if isinstance(samples, (float, int)):
            samples = [samples for _ in range(ncells)]
        elif isinstance(samples, list):
            samples = samples[:ncells]
        else:
            raise ValueError(
                "samples should be a number or a list of numbers"
            )

        rng = np.random.default_rng(random_state)
        for i in range(ncells):
            isize = data[i].shape[0]
            sample = samples[i]

            if sample is None or sample == 1:
                idx = np.arange(isize)
            elif sample > 1:
                if sample < 10:
                    print("warning: sample size is smaller than 10")
                idx = rng.choice(
                    isize,
                    min(int(sample), isize),
                    replace=False,
                )
            elif sample < 1:
                idx = rng.choice(
                    isize,
                    int(isize * sample),
                    replace=False,
                )
            else:
                idx = np.array([], dtype=int)

            sampleidx.append(idx + start)
            start += isize
        sampleidx = np.concatenate(sampleidx)

    if all(isinstance(idata, pd.DataFrame) for idata in data):
        datas = pd.concat(data, axis=0)

        if group is None:
            datas["group"] = groupidx.astype(str)
        else:
            if all(
                isinstance(idata[group].dtype, pd.CategoricalDtype)
                for idata in data
            ):
                categories = data[0][group].cat.categories.tolist()
                datas["group"] = pd.Categorical(
                    datas[group],
                    categories=categories,
                    ordered=True,
                )
            else:
                datas["group"] = datas[group]

            if group_diff:
                datas["groupidx"] = groupidx
                datas["group"] = (
                    datas["group"].astype(str)
                    + "_"
                    + datas["groupidx"].astype(str)
                )

        if X is None and Y is None and Z is None:
            X, Y, Z = datas.columns[:3].tolist()
    else:
        datas = pd.DataFrame(
            np.vstack(data),
            columns=list("xyz"),
        )
        datas["group"] = groupidx.astype(str)
        X, Y, Z = list("xyz")

    if len(sampleidx) > 0:
        datas = datas.iloc[sampleidx, :].copy()

    return qscatter(
        datas,
        X=X,
        Y=Y,
        Z=Z,
        group="group",
        order=order,
        keep_all=keep_all,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        showspikes=showspikes,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        scene_bgcolor=scene_bgcolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
        **kargs,
    )


def qscatter(
    Data,
    X=None,
    Y=None,
    Z=None,
    group=None,
    save=None,
    show=True,
    scale=1,
    height=None,
    width=None,
    xwidth=1000,
    ywidth=1000,
    zwidth=1000,
    scene_aspectmode="data",
    scene_aspectratio=dict(x=1, y=1, z=1),
    clip=None,
    sample=None,
    random_state=200504,
    order=True,
    ascending=False,
    keep_all=False,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    colormap=None,
    template="none",
    vmin=None,
    vmax=None,
    **kargs,
):
    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data, columns=list("xyz"))
    else:
        Data = Data.copy()

    if X is None and Y is None and Z is None:
        X, Y, Z = Data.columns[:3].tolist()

    Data["index"] = Data.index

    if isinstance(
        group,
        (
            pd.Series,
            pd.Categorical,
            np.ndarray,
            list,
        ),
    ):
        if (
            isinstance(group, pd.Series)
            and isinstance(group.dtype, pd.CategoricalDtype)
        ):
            Data["group"] = pd.Categorical(
                group.to_numpy(),
                categories=group.cat.categories,
                ordered=True,
            )
        else:
            Data["group"] = np.asarray(group)
        group = "group"

    if (
        group is not None
        and isinstance(Data[group].dtype, pd.CategoricalDtype)
    ):
        category_order = Data[group].cat.categories.tolist()
    else:
        category_order = None

    if sample is not None:
        if isinstance(sample, int) and sample > 1:
            Data = Data.sample(
                n=min(sample, len(Data)),
                replace=False,
                random_state=random_state,
            )
        elif isinstance(sample, float) and sample <= 1:
            Data = Data.sample(
                frac=sample,
                replace=False,
                random_state=random_state,
            )

    dimdict = dict(zip(list("xyz"), (X, Y, Z)))

    if group is not None:
        ctype = vartype(Data[group])
        dimdict["color"] = group

        if ctype == "discrete":
            if isinstance(order, (list, tuple, np.ndarray, pd.Index)):
                category_order = list(order)
            elif (
                (order is True or order == "guess")
                and category_order is not None
            ):
                category_order = category_order.copy()
            else:
                category_order = (
                    Data[group]
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )

            if not keep_all:
                observed = Data[group].dropna().unique()
                category_order = [
                    category
                    for category in category_order
                    if category in observed
                ]

            Data[group] = pd.Categorical(
                Data[group],
                categories=category_order,
                ordered=True,
            )

            if colormap is None:
                colormap = color_palette(len(category_order))

            dimdict.update(
                {
                    "category_orders": {
                        group: category_order,
                    },
                    "color_discrete_sequence": colormap,
                }
            )

        elif ctype == "continuous":
            cmin = Data[group].min()

            if vmin is not None:
                Data = Data[Data[group] >= vmin]
            if vmax is not None:
                Data[group] = Data[group].clip(None, vmax)
            if clip is not None:
                Data[group] = np.clip(
                    Data[group],
                    clip[0],
                    clip[1],
                )
            if order is True or order == "guess":
                Data.sort_values(
                    by=group,
                    ascending=ascending,
                    inplace=True,
                )
            cmax = Data[group].max()
            if "range_color" not in kargs:
                dimdict["range_color"] = [float(cmin), float(cmax)]
            colormap = cmap1px if colormap is None else colormap
            dimdict["color_continuous_scale"] = colormap

    fig = px.scatter_3d(
        Data,
        hover_name="index",
        hover_data=["index"],
        **dimdict,
        **kargs,
    )

    fig.update_layout(
        height=height,
        width=width,
        legend=dict(itemsizing="constant"),
        template=template,
    )

    _set_plotly_3d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        showspikes=showspikes,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        scene_bgcolor=scene_bgcolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
        scene_aspectmode=scene_aspectmode,
        scene_aspectratio=scene_aspectratio,
    )

    fig.update_traces(
        marker=dict(
            size=scale,
            line=dict(width=0, color="DarkSlateGrey"),
        ),
        selector=dict(mode="markers"),
    )

    if save:
        fig.write_html(save)
    if show:
        fig.show()
    else:
        return fig


def pview2d(
    *data,
    X=None,
    Y=None,
    group=None,
    group_diff=False,
    colorlist=[],
    nrows=None,
    ncols=None,
    figscale=900,
    werror=20,
    width=None,
    height=None,
    facet_col_spacing=None,
    facet_row_spacing=None,
    samples=None,
    seed=None,
    render_mode="auto",
    save=None,
    outpre=None,
    size=None,
    size_max=20,
    scale=1,
    show=True,
    template="none",
    scene_aspectmode="data",
    y_reverse=False,
    x_reverse=False,
    same_scale=True,
    scene_aspectratio=dict(x=1, y=1),
    return_fig=False,
    showlegend=True,
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    itemsizing="constant",
    itemwidth=None,
    sample=None,
    opacity=None,
    thicknessmode="pixels",
    cbar_width=25,
    lenmode="pixels",
    cbar_height=300,
    cbar_yanchor="middle",
    cbar_ypos=0.5,
    cbar_ypad=10,
    ticks="outside",
    matches=None,
    line_color="gray",
    line_width=1,
    line_alpha=1,
    color="b",
    line_weight=None,
    line_cmap=None,
    legdict={},
    **kargs,
):
    ncells = len(data)
    nrows, ncols = colrows(
        ncells,
        ncols=ncols,
        nrows=nrows,
    )
    width = ncols * figscale + werror if width is None else width
    height = nrows * figscale if height is None else height

    groupidx = np.repeat(
        np.arange(ncells),
        [idata.shape[0] for idata in data],
    )
    start, sampleidx = 0, []

    if samples is not None:
        if isinstance(samples, (float, int)):
            samples = [samples for _ in range(ncells)]
        elif isinstance(samples, list):
            samples = samples[:ncells]
        else:
            raise ValueError(
                "samples should be a number or a list of numbers"
            )

        rng = np.random.default_rng(seed)
        for i in range(ncells):
            isize = data[i].shape[0]
            sample_i = samples[i]

            if sample_i is None or sample_i == 1:
                idx = np.arange(isize)
            elif sample_i > 1:
                if sample_i < 10:
                    print("warning: sample size is smaller than 10")
                idx = rng.choice(
                    isize,
                    min(int(sample_i), isize),
                    replace=False,
                )
            elif sample_i < 1:
                idx = rng.choice(
                    isize,
                    int(isize * sample_i),
                    replace=False,
                )
            else:
                idx = np.array([], dtype=int)

            sampleidx.append(idx + start)
            start += isize
        sampleidx = np.concatenate(sampleidx)

    if all(isinstance(idata, pd.DataFrame) for idata in data):
        datas = pd.concat(data, axis=0)
        if group is None:
            datas["group"] = groupidx.astype(str)
        else:
            datas["group"] = datas[group]
            if group_diff:
                datas["groupidx"] = groupidx
                datas["group"] = (
                    datas["group"].astype(str)
                    + "_"
                    + datas["groupidx"].astype(str)
                )

        if X is None and Y is None:
            dimdict = dict(
                zip(list("xy"), datas.columns[:2].tolist())
            )
        else:
            dimdict = dict(zip(list("xy"), [X, Y]))
        datas["cellid"] = datas.index
    else:
        datas = pd.DataFrame(
            np.concatenate(data, axis=0),
            columns=list("xy"),
        )
        datas["group"] = groupidx.astype(str)
        datas["cellid"] = np.hstack(
            [np.arange(idata.shape[0]) for idata in data]
        )
        dimdict = dict(zip(list("xy"), list("xy")))

    datas["split"] = groupidx.astype(str)
    if len(sampleidx) > 0:
        datas = datas.iloc[sampleidx, :]

    fig = px.scatter(
        datas,
        color="split",
        facet_col="split",
        facet_col_wrap=ncols,
        facet_col_spacing=facet_col_spacing,
        facet_row_spacing=facet_row_spacing,
        size=size,
        size_max=size_max,
        render_mode=render_mode,
        width=width,
        height=height,
        hover_name="cellid",
        hover_data=["cellid"],
        **dimdict,
        **kargs,
    )

    itemwidth = itemwidth or (30 + ncells)
    fig.update_layout(
        legend=dict(
            itemsizing=itemsizing,
            itemwidth=itemwidth,
            **legdict,
        ),
        coloraxis_colorbar=dict(
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
            xaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
        ),
        plot_bgcolor="#FFFFFF",
    )

    _set_plotly_2d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
    )

    if size is not None:
        fig.update_traces(
            marker=dict(
                opacity=opacity,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )
    else:
        fig.update_traces(
            marker=dict(
                size=scale,
                opacity=opacity,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )

    if same_scale:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if y_reverse:
        fig.update_yaxes(autorange="reversed")
    if x_reverse:
        fig.update_xaxes(autorange="reversed")

    if show:
        fig.show()
    if save:
        fig.write_html(save)
    if return_fig:
        return fig


def qscatter0(
    Data,
    X=None,
    Y=None,
    Z=None,
    group=None,
    save=None,
    show=True,
    scale=1,
    xwidth=800,
    ywidth=800,
    zwidth=800,
    scene_aspectmode="data",
    scene_aspectratio=dict(x=1, y=1, z=1),
    clip=None,
    sample=None,
    random_state=200504,
    order="guess",
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    showspikes=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    scene_bgcolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    colormap=None,
    template="none",
    **kargs,
):
    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data, columns=list("xyz"))
    if X is None and Y is None and Z is None:
        X, Y, Z = Data.columns[:3].tolist()

    Data["index"] = Data.index

    if isinstance(
        group,
        (pd.Series, pd.Categorical, np.ndarray, list),
    ):
        if (
            isinstance(group, pd.Series)
            and isinstance(group.dtype, pd.CategoricalDtype)
        ):
            categories = group.cat.categories
            Data["group"] = pd.Categorical(
                np.asarray(group),
                categories=categories,
            )
        else:
            Data["group"] = np.asarray(group)
        group = "group"

    if sample is not None:
        if isinstance(sample, int) and sample > 1:
            Data = Data.sample(
                n=min(sample, len(Data)),
                replace=False,
                random_state=random_state,
            )
        if isinstance(sample, float) and sample <= 1:
            Data = Data.sample(
                frac=sample,
                replace=False,
                random_state=random_state,
            )

    dimdict = dict(zip(list("xyz"), (X, Y, Z)))
    if group is not None:
        ctype = vartype(Data[group])
        dimdict["color"] = group

        if ctype == "discrete":
            if isinstance(Data[group].dtype, pd.CategoricalDtype):
                iorder = Data[group].cat.categories.tolist()
            else:
                iorder = Data[group].unique().tolist()

            if colormap is None:
                colormap = color_palette(len(iorder))
            dimdict.update(
                {
                    "category_orders": {group: iorder},
                    "color_discrete_sequence": colormap,
                }
            )
        elif ctype == "continuous":
            if clip is not None:
                Data[group] = np.clip(
                    Data[group],
                    clip[0],
                    clip[1],
                )
            if order == "guess" or order is True:
                Data = Data.sort_values(
                    by=group,
                    ascending=True,
                )
            colormap = "Viridis" if colormap is None else colormap
            dimdict["color_continuous_scale"] = colormap

    fig = px.scatter_3d(
        Data,
        hover_name="index",
        hover_data=["index"],
        **dimdict,
        **kargs,
    )
    fig.update_layout(
        legend=dict(itemsizing="constant"),
        scene_aspectmode=scene_aspectmode,
        scene_aspectratio=scene_aspectratio,
        template=template,
        scene=dict(
            xaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
        ),
        plot_bgcolor="#FFFFFF",
    )

    _set_plotly_3d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        showspikes=showspikes,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        scene_bgcolor=scene_bgcolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
        scene_aspectmode=scene_aspectmode,
        scene_aspectratio=scene_aspectratio,
    )
    fig.update_traces(
        marker=dict(
            size=scale,
            line=dict(width=0, color="DarkSlateGrey"),
        ),
        selector=dict(mode="markers"),
    )
    fig.update_traces(text=[{"index": Data.index}])

    if save:
        fig.write_html(save)
    if show:
        fig.show()
    else:
        return fig


def qscatter2d(
    Data,
    X=None,
    Y=None,
    group=None,
    save=None,
    show=True,
    scale=1,
    same_scale=True,
    size=None,
    opacity=None,
    sample=None,
    figscale=1000,
    werror=100,
    width=None,
    height=None,
    scene_aspectmode="data",
    reverse_y=False,
    reverse_x=False,
    showlegend=True,
    scene_aspectratio=dict(x=1, y=1, z=1),
    clip=None,
    random_state=200504,
    order="guess",
    theme="white",
    show_axes=True,
    show_grid=True,
    showticklabels=False,
    axis_ticks="",
    show_axis_titles=True,
    show_axisline=True,
    show_zeroline=False,
    gridwidth=1,
    gridcolor=None,
    axiscolor=None,
    plot_bgcolor=None,
    paper_bgcolor=None,
    fontcolor=None,
    colordict={},
    render_mode="auto",
    colormap=None,
    template="none",
    itemsizing="constant",
    itemwidth=None,
    legdict={},
    **kargs,
):
    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data, columns=list("xy"))
    else:
        Data = Data.copy()

    if X is None and Y is None:
        X, Y = Data.columns[:2].tolist()

    Data["index"] = Data.index

    if isinstance(
        group,
        (pd.Series, pd.Categorical, np.ndarray, list),
    ):
        if (
            isinstance(group, pd.Series)
            and isinstance(group.dtype, pd.CategoricalDtype)
        ):
            categories = group.cat.categories
            Data["group"] = pd.Categorical(
                np.asarray(group),
                categories=categories,
            )
        else:
            Data["group"] = np.asarray(group)
        group = "group"

    if sample is not None:
        if isinstance(sample, int) and sample > 1:
            Data = Data.sample(
                n=min(sample, len(Data)),
                replace=False,
                random_state=random_state,
            )
        if isinstance(sample, float) and sample <= 1:
            Data = Data.sample(
                frac=sample,
                replace=False,
                random_state=random_state,
            )

    dimdict = dict(zip(list("xy"), (X, Y)))
    if group is not None:
        ctype = vartype(Data[group])
        dimdict["color"] = group

        if ctype == "discrete":
            if isinstance(Data[group].dtype, pd.CategoricalDtype):
                iorder = Data[group].cat.categories.tolist()
            else:
                iorder = Data[group].unique().tolist()

            colormap = colordict.get(
                group,
                color_palette(len(iorder)),
            )
            dimdict.update(
                {
                    "category_orders": {group: iorder},
                    "color_discrete_sequence": colormap,
                }
            )
        elif ctype == "continuous":
            if clip is not None:
                Data[group] = np.clip(
                    Data[group],
                    clip[0],
                    clip[1],
                )
            if order == "guess" or order is True:
                Data = Data.sort_values(
                    by=group,
                    ascending=True,
                )
            colormap = "Viridis" if colormap is None else colormap
            dimdict["color_continuous_scale"] = colormap

    width = figscale + werror if width is None else width
    height = figscale if height is None else height

    fig = px.scatter(
        Data,
        hover_name="index",
        hover_data=["index"],
        width=width,
        height=height,
        render_mode=render_mode,
        **dimdict,
        **kargs,
    )

    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    if reverse_x:
        fig.update_xaxes(autorange="reversed")

    itemwidth = itemwidth or 30
    fig.update_layout(
        legend=dict(
            itemsizing=itemsizing,
            itemwidth=itemwidth,
            **legdict,
        ),
        showlegend=showlegend,
        scene_aspectmode=scene_aspectmode,
        template=template,
        scene_aspectratio=scene_aspectratio,
        scene=dict(
            xaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                visible=show_grid,
                showticklabels=showticklabels,
            ),
        ),
        plot_bgcolor="#FFFFFF",
    )

    _set_plotly_2d_theme(
        fig,
        theme=theme,
        show_axes=show_axes,
        show_grid=show_grid,
        showticklabels=showticklabels,
        axis_ticks=axis_ticks,
        show_axis_titles=show_axis_titles,
        show_axisline=show_axisline,
        show_zeroline=show_zeroline,
        gridwidth=gridwidth,
        gridcolor=gridcolor,
        axiscolor=axiscolor,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        fontcolor=fontcolor,
    )

    fig.update_traces(text=[{"index": Data.index}])
    if size is not None:
        fig.update_traces(
            marker=dict(
                opacity=opacity,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )
    else:
        fig.update_traces(
            marker=dict(
                opacity=opacity,
                size=scale,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )

    if same_scale:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if save:
        fig.write_html(save)
    if show:
        fig.show()
    else:
        return fig
