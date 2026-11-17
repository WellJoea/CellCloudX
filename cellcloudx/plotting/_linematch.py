"""Generic index-based 2D and PyVista matching plots."""

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Colormap, is_color_like, ListedColormap, Normalize, to_rgba_array

from ._imageview import drawMatches as lineMatches2D
from ._colors import color_palette, to_hex
from ._match_state import _numpy
from ._pyvistaplts import surfaces3d, surfaces_df, save_mesh
from ..utilis._arrays import list_iter, vartype




def _match_palette(order, palette):
    """Resolve categorical palettes consistently for points and matching lines."""
    if isinstance(palette, dict):
        return palette
    if palette is None:
        shades = color_palette(len(order)) if len(order) else []
    elif is_color_like(palette):
        shades = [palette] * len(order)
    elif isinstance(palette, (str, Colormap)):
        cm = colormaps[palette] if isinstance(palette, str) else palette
        shades = cm(np.linspace(0, 1, len(order)))
    else:
        shades = [palette[i % len(palette)] for i in range(len(order))]
    return dict(zip(order, shades))


def _match_lines(pl, start, end, edge_ids, widths, weight, line_color,
                 weight_width, line_width, line_alpha, line_radius,
                 line_style, tube_sides, arrowstyle, arrow_size, rng,
                 smooth_shading, line_lighting, name, lkargs):
    """Build independent line cells, optional tubes, and arrow glyphs in batches."""
    import pyvista as pv

    n = len(start)
    if n == 0:
        return
    rgba = line_color
    if weight is not None and weight_width:
        maximum = weight.max()
        widths = widths * (weight / maximum if maximum > 0 else np.zeros(n))
    if rgba is None:
        rgba = np.c_[rng.random((n, 3)), np.ones(n)]
    else:
        rgba = to_rgba_array(rgba)
        rgba = np.broadcast_to(rgba, (n, 4)).copy()
    rgba[:, 3] *= 1 if line_alpha is None else line_alpha
    valid = (widths > 0) & (np.linalg.norm(end - start, axis=1) > 0)
    if not valid.any():
        return
    start, end, widths, rgba, edge_ids = [
        value[valid] for value in (start, end, widths, rgba, edge_ids)]
    n = len(start)
    mesh = pv.line_segments_from_points(np.stack([start, end], axis=1).reshape(-1, 3))
    mesh.cell_data['edge_id'] = edge_ids
    mesh.cell_data['line_rgba'] = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    if weight is not None:
        mesh.cell_data['weight'] = weight[valid]

    # A VTK actor has one pixel width. Tubes preserve arbitrary per-edge widths.
    if line_style == 'tube' or not np.all(widths == widths[0]):
        mesh.point_data['radius'] = np.repeat(line_radius * widths, 2)
        mesh = mesh.tube(scalars='radius', absolute=True, n_sides=tube_sides)
    elif line_style != 'line':
        raise ValueError("line_style must be 'line' or 'tube'.")
    options = dict(line_width=float(line_width * widths[0]),
                   render_lines_as_tubes=True, lighting=line_lighting,
                   smooth_shading=smooth_shading, reset_camera=False, render=False)
    options.update(lkargs or {})
    options.update(name=name, scalars='line_rgba', rgb=True, preference='cell',
                   show_scalar_bar=False)
    pl.add_mesh(mesh, **options)

    if arrowstyle in ('-', None):
        return
    directions = {'->': (1,), '-|>': (1,), '<-': (-1,), '<|-': (-1,),
                  '<->': (1, -1), '<|-|>': (1, -1)}
    if arrowstyle not in directions:
        raise ValueError("arrowstyle must be '-', '->', '<-', or '<->'.")
    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    for sign in directions[arrowstyle]:
        tip = end if sign == 1 else start
        heads = pv.PolyData(tip - sign * arrow_size * vector)
        heads['direction'] = sign * vector
        heads['length'] = arrow_size * length
        heads['line_rgba'] = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        arrows = heads.glyph(orient='direction', scale='length', factor=1,
                            geom=pv.Arrow(tip_length=0.35, tip_radius=0.18,
                                          shaft_radius=0.03))
        options.update(name=name + '_arrow_' + str(sign), preference='point')
        pl.add_mesh(arrows, **options)


def lineMatches3D(pairlist, bgs=None, pairidx=None, matches=None,
                  groupbys=None, groupby=None, points_colors=None, cmap=None,
                  color=('#146fb1', '#ff750c', '#168816'), size=None, alpha=None,
                  point_psize=3, point_opacity=1, point_styles='points',
                  bg_color='gray', bg_size=1, bg_alpha=0.15, bg_groupbys=None,
                  line_widths=None, line_weight=None, line_color='gray',
                  line_width=1, line_alpha=0.3, line_cmap=None, line_clim=None,
                  line_sample=None, line_top=None, show_line=True, seed=None,
                  line_style='line', line_radius=None, tube_sides=8,
                  arrowstyle='-', arrow_size=0.04,
                  axes=(0, 1, 2), layout=None, gap=0.2, offsets=None,
                  project=False, invert_xaxis=False, invert_yaxis=False,
                  invert_zaxis=False, aspect_equal=True, box_aspect=None,
                  xyzscales=None, titles=None, title=None, title_fontsize=15,
                  axis_off=True, ticks_off=False, grid=False, axislabel='Dim',
                  show_axes=False, show_bounds=False, show_legend=False,
                  show_scalar_bar=False, clim=None, na_color='gray',
                  scalar_bar_args=None, legend_args=None, bounds_args=None,
                  meshlist=None, colors='#AAAAAA', opacity=0.2,
                  show_edges=True, edge_colors='#AAAAAA', mesh_style='surface',
                  smooth_shading=True, point_lighting=True, mesh_lighting=False,
                  line_lighting=False, render_points_as_spheres=True,
                  theme_style=None, theme=None, background_color='black',
                  font_color=None, font_size=15, font_family='arial',
                  window_size=(1600, 800), image_scale=1, line_smoothing=False,
                  anti_aliasing=True, aa_type='fxaa', multi_samples=None,
                  all_renderers=True, cpos='xy', view_up=None,
                  view_isometric=False, azimuth=None, elevation=None,
                  roll=None, zoom=None, parallel_projection=True,
                  notebook=None, off_screen=None, jupyter_backend=None,
                  start_xvfb=False, Light=None, pl=None, return_pl=False,
                  save=None, save_type=None, show=True,
                  pkargs=None, lkargs=None, mkargs=None, fkargs=None, skargs=None,
                  color_by=None, weight_width=True, mesh_line_width=3, edge_opacity=None):
    """Plot point clouds and index-based matches, reusing the plotting module.

    pairlist contains (N,2)/(N,3) point arrays. matches has the same format as
    lineMatches2D: [(ix0,iy0), (ix1,iy1), ...], one pair of equal-length integer
    index arrays per pairidx entry. A (2,E) array is equivalent to (ix,iy).
    None matches points by row. pairidx defaults to adjacent clouds.

    line_weight/line_widths contain one edge array per pairidx entry; line_color
    may be one solid color, one color per pair, or arrays of edge colors per
    pair. Numeric weights control relative width w/max(w) if weight_width=True;
    categorical weights never change widths. Use weight_width=False for widths
    controlled only by line_width and line_widths.

    color_by=None/'weight'/'score' colors by line_weight, or uses line_color
    when no weights are supplied. 'category' treats line_weight as categories,
    including numeric category IDs. 'source'/'target' uses the corresponding
    groupbys labels at each edge endpoint. 'count' colors by the number of edges
    with the same (source label, target label) in each full matching pair BEFORE
    sampling; groupbys is required. 'color' uses line_color independently of
    line_weight. Width weights remain independent of color_by.

    line_cmap accepts a colormap name/object, category-color dict, or categorical
    palette. None uses viridis for numeric values, the endpoint point palette
    for source/target labels, or a shared palette for other categories.
    line_clim applies only to numeric colors; None uses the full unsampled
    range shared by all matching pairs. Missing color values use na_color.
    show_legend includes categorical line colors. There is no dependency on
    any matching-result or relation-table format.

    line_sample: None/1 = all, 0 = none, fraction = random proportion, >1 =
    random count per pair. line_top takes the FIRST n edges instead of sampling.
    Edge attributes follow the same edge order before sampling.

    groupbys/bg_groupbys contain labels or numeric values per cloud.
    points_colors accepts a shared category-color dict, palette, colormap, or
    nested per-cloud palettes. point size/opacity/style/lighting may be scalar,
    per cloud, or nested per category. Missing values use na_color. All points
    remain visible independently of matching-line sampling.

    layout='x'/'y'/'z' separates clouds along the positive axis. Prefix '-' to
    reverse the placement direction, e.g. layout='-z' stacks clouds downward
    with cpos='xz'. Cloud order, orientation and matching indices are retained.
    gap is relative to the largest cloud extent on that axis. offsets=(L,3),
    or a per-cloud step vector (3,), overrides layout. Points, backgrounds and
    per-cloud meshlist entries move together.
    axes=(0,1) or project=True gives a true XY projection; cpos='xy' changes only
    the initial view. Inputs are never modified.

    meshlist[i] is one mesh, a list of meshes, or None for cloud i. colors,
    opacity, show_edges, edge_colors, mesh_style, mesh_lighting, mesh_line_width
    and edge_opacity accept a scalar, per-cloud list, or nested per-mesh lists
    matching meshlist. None entries retain their position in these lists.
    mesh_line_width controls mesh edges/wireframes; line_width controls matches.
    smooth_shading is shared globally. mkargs is one common extra-options dict.

    Appearance, mesh rendering, camera and export reuse surfaces3d/surfaces_df/
    save_mesh. line_style='tube' and variable widths use actual tube meshes;
    line_radius is their base radius in coordinate units. arrowstyle accepts
    '-', '->', '<-', '<->'. arrow_size is relative to segment length.

    pkargs/lkargs/mkargs/fkargs forward rendering options; skargs forwards
    save_mesh options. return_pl=True returns a live plotter before show/save;
    otherwise show=False saves and closes, following save_mesh. With an existing
    pl, rendering uses its active subplot and retains its theme/window.
    """
    import pyvista as pv

    n = len(pairlist)
    assert n > 1, 'pairlist must contain at least two point clouds.'
    pairidx = list(zip(range(n-1), range(1, n))) if pairidx is None else pairidx
    if color_by not in (None, 'weight', 'score', 'category', 'source', 'target', 'count', 'color'):
        raise ValueError('Unknown color_by: ' + str(color_by))
    if color_by in ('source', 'target', 'count') and groupbys is None:
        raise ValueError('groupbys is required for color_by=' + repr(color_by))
    if color_by in ('weight', 'score', 'category') and line_weight is None:
        raise ValueError('line_weight is required for color_by=' + repr(color_by))
    if matches is not None:
        assert len(matches) == len(pairidx), 'matches and pairidx must have the same length.'
    rng = np.random.default_rng(seed)
    axes = list(axes)
    signs = np.array([-1 if invert_xaxis else 1, -1 if invert_yaxis else 1,
                      -1 if invert_zaxis else 1])

    def coordinates(value):
        value = _numpy(value).astype(float, copy=True)
        if value.shape[1] == 2:
            value = np.c_[value, np.zeros(len(value))]
        value = value[:, axes]
        if len(axes) == 2:
            value = np.c_[value, np.zeros(len(value))]
        if project:
            value[:, 2] = 0
        return value * signs

    points = [coordinates(value) for value in pairlist]
    backgrounds, combined_bg = [None] * n, None
    if bgs is not None:
        if isinstance(bgs, (list, tuple)):
            assert len(bgs) == n, 'bgs and pairlist must have the same length.'
            backgrounds = [None if x is None else coordinates(x) for x in bgs]
        else:
            if layout is not None or offsets is not None:
                raise ValueError('Use one bgs array per cloud with layout/offsets.')
            combined_bg = coordinates(bgs)
    meshes = [[] for _ in range(n)]
    if meshlist is not None:
        assert len(meshlist) == n, 'meshlist and pairlist must have the same length.'
    for i, entry in enumerate(meshlist or []):
        for mesh in entry if isinstance(entry, (list, tuple)) else [entry]:
            if mesh is None:
                meshes[i].append(None)
                continue
            mesh = mesh.copy()
            mesh.points = coordinates(mesh.points)
            for data in (mesh.point_data, mesh.cell_data):
                key = data.active_normals_name
                if key is not None:
                    if len(axes) == 3 and not project:
                        data[key] = data[key][:, axes] * signs
                    else:
                        del data[key]
            meshes[i].append(mesh)
    extents = []
    for i in range(n):
        parts = [points[i]] + [mesh.points for mesh in meshes[i] if mesh is not None]
        if backgrounds[i] is not None:
            parts.append(backgrounds[i])
        parts = [value for value in parts if len(value)]
        joined = np.vstack(parts) if parts else np.zeros((1, 3))
        extents.append(np.array([joined.min(0), joined.max(0)]))
    shifts = np.zeros((n, 3))
    if offsets is not None:
        offsets = _numpy(offsets)
        shifts = np.arange(n)[:, None] * offsets if offsets.ndim == 1 else offsets
    elif layout is not None:
        reverse = layout.startswith('-')
        axis = {'x': 0, 'y': 1, 'z': 2}[layout.lstrip('-')]
        end, start = (0, 1) if reverse else (1, 0)
        distance = max(np.ptp(extent, axis=0)[axis] for extent in extents)
        distance = distance or max(np.ptp(np.vstack(extents), axis=0).max(), 1.)
        for i in range(1, n):
            shifts[i, axis] = (extents[i-1][end, axis] + shifts[i-1, axis]
                              - extents[i][start, axis] + (-gap if reverse else gap) * distance)
    for i in range(n):
        points[i] += shifts[i]
        extents[i] += shifts[i]
        if backgrounds[i] is not None:
            backgrounds[i] += shifts[i]
        for mesh in meshes[i]:
            if mesh is not None:
                mesh.points += shifts[i]

    # Prepare shared categorical colors; the existing renderer handles drawing.
    palettes = points_colors if points_colors is not None else cmap
    nested = (isinstance(palettes, (list, tuple)) and len(palettes) == n
              and any(isinstance(v, (dict, Colormap)) or
                      (isinstance(v, (str, list, tuple, np.ndarray)) and not is_color_like(v))
                      for v in palettes))
    groups = ([] if groupbys is None else list(groupbys))
    groups += ([] if bg_groupbys is None else list(bg_groupbys)) + [groupby]
    order = []
    for values in groups:
        if values is None:
            continue
        values = pd.Series(_numpy(values) if hasattr(values, 'detach') else values)
        if vartype(values) == 'discrete':
            cats = values.cat.categories if isinstance(values.dtype, pd.CategoricalDtype) else pd.unique(values.dropna())
            order.extend(c for c in cats if c not in order)
    shared = palettes if isinstance(palettes, dict) else None
    if order and not nested and shared is None:
        shared = _match_palette(order, palettes)
    psize, palpha, pstyle, plight = [list_iter(v, last_as_default=True) for v in
        (point_psize if size is None else size, point_opacity if alpha is None else alpha,
         point_styles, point_lighting)]
    solid, bgsolid = [list_iter([v] if is_color_like(v) else v, last_as_default=True)
                      for v in (color, bg_color)]
    bgsize, bgalpha = [list_iter(v, last_as_default=True) for v in (bg_size, bg_alpha)]
    frames, columns, cmaps, sizes, alphas, styles, lights, legends = [], [], [], [], [], [], [], {}

    def add_cloud(data, values, pc, shade, style, name='points'):
        if len(data) == 0:
            return
        solid_points = values is None
        values = (pd.Series(_numpy(values) if hasattr(values, 'detach') else values).reset_index(drop=True)
                  if values is not None else pd.Series(pd.Categorical([name] * len(data))))
        discrete = vartype(values) == 'discrete'
        cats = (values.cat.categories if isinstance(values.dtype, pd.CategoricalDtype)
                else pd.unique(values.dropna())) if discrete else None
        if discrete:
            missing = values.isna().to_numpy()
            if missing.any():
                add_cloud(data[missing], None, None, na_color, style, 'NA')
                data, values = data[~missing], values[~missing].reset_index(drop=True)
                if len(data) == 0:
                    return
            values = pd.Series(pd.Categorical(values, categories=cats))
            pc = _match_palette(cats, [shade] if solid_points else pc)
            pc = [pc.get(c, na_color) for c in cats]
            pc = [to_hex(c, keep_alpha=True) for c in pc]
            names = list(map(str, cats))
            if len(set(names)) != len(names):
                names = [str(k) + ': ' + label for k, label in enumerate(names)]
            legends.update(zip(names, pc))
        else:
            valid = np.isfinite(values.to_numpy(dtype=float))
            if not valid.all():
                add_cloud(data[~valid], None, None, na_color, style, 'NA')
                data, values = data[valid], values[valid].reset_index(drop=True)
                if len(data) == 0:
                    return
            pc = colormaps['viridis'] if pc is None else pc
            pc = colormaps[pc] if isinstance(pc, str) else pc
            if not isinstance(pc, Colormap):
                pc = ListedColormap(list(pc.values()) if isinstance(pc, dict) else pc)
        split = discrete and any(isinstance(v, (list, tuple, np.ndarray)) for v in style)
        for k in range(len(cats)) if split else [None]:
            mask = values.eq(cats[k]).to_numpy() if split else np.ones(len(data), dtype=bool)
            if not mask.any():
                continue
            column = 'match_group_' + str(len(frames))
            frame = pd.DataFrame(data[mask], columns=list('xyz'))
            frame[column] = (pd.Categorical.from_codes(values[mask].cat.codes,
                             categories=names) if discrete
                             else values[mask].reset_index(drop=True))
            frames.append(frame)
            columns.append(column)
            cmaps.append(pc)
            settings = [list_iter(v, last_as_default=True)[k] for v in style] if split else style
            for destination, value in zip((sizes, alphas, styles, lights), settings):
                destination.append(value)

    for i in range(n):
        for role, (data, values, shade, style) in enumerate([
            (backgrounds[i], None if bg_groupbys is None else bg_groupbys[i], bgsolid[i],
             (bgsize[i], bgalpha[i], pstyle[i], plight[i])),
            (points[i], None if groupbys is None else groupbys[i], solid[i],
             (psize[i], palpha[i], pstyle[i], plight[i]))]):
            if data is None:
                continue
            pc = palettes[i] if nested else (shared if values is not None and vartype(values) == 'discrete' else palettes)
            name = (str(titles[i]) if titles is not None else 'points_' + str(i))
            add_cloud(data, values, pc if values is not None else None, shade, style,
                      name + ' background' if role == 0 else name)
    if combined_bg is not None:
        add_cloud(combined_bg, groupby, shared if groupby is not None and vartype(groupby) == 'discrete' else palettes,
                  bgsolid[0], (bgsize[0], bgalpha[0], pstyle[0], plight[0]))

    # surfaces_df owns plotter/theme creation; surfaces3d draws all actors once.
    if pl is not None:
        theme = pl.theme
    if theme_style == 'white':
        background_color, font_color = 'white', font_color or 'black'
    else:
        background_color = 'black' if theme_style in (None, 'dark') else background_color
        font_color = font_color or (theme.font.color if theme is not None else 'white')
    pl = surfaces_df(pl=pl, theme=theme, background_color=background_color,
                     font_color=font_color, font_family=font_family, font_size=font_size,
                     title_size=title_fontsize, window_size=window_size, image_scale=image_scale,
                     notebook=notebook, off_screen=off_screen, start_xvfb=start_xvfb,
                     jupyter_backend=jupyter_backend, line_smoothing=line_smoothing,
                     anti_aliasing=None, Light=Light, fkargs=fkargs or {}, return_pl=True)
    mesh_options = [list_iter([v] if k in (0, 3) and is_color_like(v) else v, last_as_default=True)
                    for k, v in enumerate((colors, opacity, show_edges, edge_colors, mesh_style,
                                           mesh_lighting, mesh_line_width, edge_opacity))]
    flat_meshes, mesh_values = [], [[] for _ in mesh_options]
    for i, entries in enumerate(meshes):
        for j, mesh in enumerate(entries):
            if mesh is None:
                continue
            flat_meshes.append(mesh)
            for k, (target, values) in enumerate(zip(mesh_values, mesh_options)):
                value = values[i]
                value = [value] if k in (0, 3) and is_color_like(value) else value
                target.append(list_iter(value, last_as_default=True)[j])
    if flat_meshes or frames:
        with np.errstate(divide='ignore', invalid='ignore'):
            surfaces3d(pl=pl, meshlist=flat_meshes or None, pointslist=frames,
                       groupbys=columns, points_colors=cmaps, point_psize=sizes,
                       point_opacity=alphas, point_styles=styles, point_lighting=lights,
                       colors=mesh_values[0], opacity=mesh_values[1], show_edges=mesh_values[2],
                       edge_colors=mesh_values[3], mesh_style=mesh_values[4], mesh_lighting=mesh_values[5],
                       line_width=mesh_values[6], edge_opacity=mesh_values[7],
                       smooth_shading=smooth_shading, render_points_as_spheres=render_points_as_spheres,
                       font_color=font_color, legend_font_color=font_color,
                       pvmin=None if clim is None else clim[0], pvmax=None if clim is None else clim[1],
                       na_color=na_color, args_cbar=scalar_bar_args or {}, show_scalar_bar=show_scalar_bar,
                       dropmin=False, show_actor=False, anti_aliasing=None, return_pl=True,
                       pkargs=dict({'render': False, 'reset_camera': False}, **(pkargs or {})),
                       mkargs=dict({'render': False, 'reset_camera': False}, **(mkargs or {})))

    bounds = np.vstack(extents)
    if combined_bg is not None and len(combined_bg):
        bounds = np.vstack([bounds, combined_bg.min(0), combined_bg.max(0)])
    extent = np.ptp(bounds, axis=0)
    radius = (max(np.linalg.norm(extent), 1.) * line_width / max(pl.window_size) / 2
              if line_radius is None else line_radius)
    if show_line and line_width > 0:
        # Resolve full edge values before sampling so categories/counts/ranges agree.
        edge_data, line_order, numeric = [], [], []
        for p, (i, j) in enumerate(pairidx):
            ix, iy = (np.tile(np.arange(min(len(points[i]), len(points[j]))), (2, 1))
                      if matches is None else matches[p])
            ix, iy = _numpy(ix), _numpy(iy)
            values, weight = None, None
            if line_weight is not None:
                raw = line_weight[p]
                values = pd.Series(_numpy(raw) if hasattr(raw, 'detach') else raw).reset_index(drop=True)
                if vartype(values) == 'continuous' and color_by != 'category':
                    weight = values.to_numpy(dtype=float)
            if color_by in ('source', 'target', 'count'):
                source = pd.Series(_numpy(groupbys[i]) if hasattr(groupbys[i], 'detach') else groupbys[i]).iloc[ix].reset_index(drop=True)
                target = pd.Series(_numpy(groupbys[j]) if hasattr(groupbys[j], 'detach') else groupbys[j]).iloc[iy].reset_index(drop=True)
                if color_by == 'count':
                    frame = pd.DataFrame({'source': source, 'target': target})
                    values = frame.groupby(['source', 'target'], observed=True, dropna=False)['source'].transform('size')
                else:
                    values = pd.Series(pd.Categorical(source if color_by == 'source' else target))
            elif color_by == 'category':
                values = pd.Series(pd.Categorical(values))
            elif color_by == 'color':
                values = None
            if values is not None:
                if vartype(values) == 'discrete':
                    cats = values.cat.categories if isinstance(values.dtype, pd.CategoricalDtype) else pd.unique(values.dropna())
                    line_order.extend(c for c in cats if c not in line_order)
                else:
                    numeric.append(values.to_numpy(dtype=float))
            edge_data.append((ix, iy, weight, values))
        line_palette = _match_palette(line_order, line_cmap)
        if line_cmap is None and shared is not None:
            if color_by in ('source', 'target'):
                line_palette = shared
            else:
                line_palette.update({c: shared[c] for c in line_order if c in shared})
        numeric = np.concatenate(numeric) if numeric else np.empty(0)
        numeric = numeric[np.isfinite(numeric)]
        limits = line_clim if line_clim is not None else ((numeric.min(), numeric.max()) if len(numeric) else (0, 1))
        for p, ((i, j), (ix, iy, weight, values)) in enumerate(zip(pairidx, edge_data)):
            keep = np.arange(len(ix))
            if line_top is not None:
                keep = keep[:line_top]
            elif line_sample is not None and line_sample != 1:
                count = int(line_sample * len(ix)) if line_sample < 1 else min(int(line_sample), len(ix))
                keep = rng.choice(keep, size=count, replace=False)
            widths = np.ones(len(ix)) if line_widths is None else _numpy(line_widths[p])
            shade = line_color if line_color is None or is_color_like(line_color) else line_color[p]
            if values is not None and vartype(values) == 'discrete':
                pc = line_palette
                if line_cmap is None and color_by in ('source', 'target'):
                    cloud = i if color_by == 'source' else j
                    if nested:
                        labels = pd.Series(_numpy(groupbys[cloud]) if hasattr(groupbys[cloud], 'detach') else groupbys[cloud])
                        cats = labels.cat.categories if isinstance(labels.dtype, pd.CategoricalDtype) else pd.unique(labels.dropna())
                        pc = _match_palette(cats, palettes[cloud])
                legends.update({str(c): to_hex(v, keep_alpha=True) for c, v in pc.items()})
                shade = to_rgba_array([na_color if pd.isna(v) else pc.get(v, na_color)
                                       for v in values.iloc[keep]])
            elif values is not None:
                v = values.iloc[keep].to_numpy(dtype=float)
                shade = np.broadcast_to(to_rgba_array(na_color), (len(v), 4)).copy()
                valid = np.isfinite(v)
                cm = 'viridis' if line_cmap is None else line_cmap
                cm = colormaps[cm] if isinstance(cm, str) else cm
                shade[valid] = cm(Normalize(*limits)(v[valid]))
            elif shade is not None and not is_color_like(shade):
                shade = _numpy(shade)[keep]
            weight = None if weight is None else weight[keep]
            _match_lines(pl, points[i][ix[keep]], points[j][iy[keep]], keep, widths[keep],
                         weight, shade, weight_width, line_width, line_alpha, radius,
                         line_style, tube_sides, arrowstyle, arrow_size, rng, smooth_shading,
                         line_lighting, 'match_lines_' + str(p), lkargs)
    if titles is not None:
        for i, extent_i in enumerate(extents):
            position = extent_i.mean(0)
            position[1] = extent_i[1, 1] + .04 * max(np.ptp(extent_i, axis=0).max(), 1.)
            pl.add_point_labels(position[None], [str(titles[i])], font_size=title_fontsize,
                                text_color=font_color, font_family=font_family, shape=None,
                                show_points=False, always_visible=True, name='match_title_' + str(i))
    if show_legend and legends:
        args = dict(labels=list(legends.items()), bcolor=None, face='circle', font_family=font_family)
        args.update(legend_args or {})
        legend = pl.add_legend(**args)
        legend.GetEntryTextProperty().SetColor(pv.Color(font_color).float_rgb)
    safe_extent = np.where(extent > 0, extent, 1.)
    scales = (np.asarray(box_aspect) * safe_extent.max() / safe_extent if box_aspect is not None
              else safe_extent.max() / safe_extent if not aspect_equal else np.ones(3))
    if xyzscales is not None:
        scales *= xyzscales
    if parallel_projection:
        pl.enable_parallel_projection()
    else:
        pl.disable_parallel_projection()
    pl.reset_camera()
    bounds_options = dict(grid='back' if grid else None, color=font_color,
                          show_xlabels=not ticks_off, show_ylabels=not ticks_off,
                          show_zlabels=not ticks_off, font_size=font_size,
                          xtitle=axislabel+'_1', ytitle=axislabel+'_2', ztitle=axislabel+'_3')
    bounds_options.update(bounds_args or {})
    surfaces_df(pl=pl, theme=pl.theme, cpos='iso' if view_isometric else cpos,
                view_up=view_up, azimuth=azimuth, elevation=elevation, roll=roll, zoom=zoom,
                xyzscales=scales, title=title, main_title_font=font_family,
                main_title_color=font_color, main_title_size=title_fontsize,
                show_axes=show_axes and not axis_off, show_bounds=(show_bounds or grid) and not axis_off,
                axes_labels=dict(xlabel=axislabel+'_1', ylabel=axislabel+'_2', zlabel=axislabel+'_3'),
                bounds_args=bounds_options, anti_aliasing=anti_aliasing, aa_type=aa_type,
                multi_samples=multi_samples, all_renderers=all_renderers, return_pl=True)
    if return_pl:
        return pl
    args = dict(scale=image_scale, jupyter_backend=jupyter_backend)
    args.update(skargs or {})
    return save_mesh(pl, save_type=save_type)(save, show=show, **args)