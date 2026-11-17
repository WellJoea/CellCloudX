import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, FuncNorm, to_hex, to_rgba
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle

def _palette(order, palette):
    if palette is None:
        return list(plt.get_cmap('tab20')(np.linspace(0, 1, len(order))))
    return [to_rgba(palette[label]) for label in order]

def _numpy(values):
    if hasattr(values, 'detach'):
        return values.detach().cpu().numpy()
    return np.asarray(values)

def _save_match(fig, save, dpi, save_kargs):
    """Save by extension; Plotly raster DPI uses the existing 100 px/in layout."""
    if save is None:
        return
    if isinstance(fig, plt.Figure):
        fig.savefig(save, dpi=dpi, **save_kargs)
        return

    from pathlib import Path
    suffix = Path(save).suffix.lower()
    if suffix in ('.html', '.htm'):
        fig.write_html(str(save), **save_kargs)
    elif suffix == '.json':
        fig.write_json(str(save), **save_kargs)
    else:
        options = dict(save_kargs)
        fmt = (options.get('format') or suffix.lstrip('.') or 'png').lower()
        options['format'] = fmt
        if dpi is not None and fmt in ('png', 'jpg', 'jpeg', 'webp'):
            options.setdefault('scale', dpi / 100)
        fig.write_image(str(save), **options)

def matching_table(result, x, y, order, score_col = 'score' ):
    """Retained-pair counts/mean scores and original/unique matched cell totals."""
    keep = result.get('keep', slice(None))
    ix, iy = _numpy(result['Inds'][:, keep])
    score = _numpy(result[score_col][keep])
    if  score_col == 'psp':
        score = score/score.max()

    n = len(order)
    cx = pd.Categorical(x, categories=order).codes.astype(np.intp)
    cy = pd.Categorical(y, categories=order).codes.astype(np.intp)
    flat = cx[ix] * n + cy[iy]
    count = np.bincount(flat, minlength=n*n).reshape(n, n)
    score_sum = np.bincount(flat, weights=score, minlength=n*n).reshape(n, n)
    mean = np.full((n, n), np.nan)
    np.divide(score_sum, count, out=mean, where=count > 0)
    coverage = []
    for codes, indices in [(cx, ix), (cy, iy)]:
        total = np.bincount(codes, minlength=n)
        matched = np.bincount(codes[np.unique(indices)], minlength=n)
        coverage.append(pd.DataFrame({'total': total, 'matched': matched}, index=order))

    match_rate = (cx[ix]  == cy[iy] ).sum()/len(ix)
    match_score=  (match_rate * (np.unique(ix).shape[0]/cx.shape[0]) * (np.unique(iy).shape[0]/cy.shape[0])) ** (1.0/3.0)
    return {
        'order': list(order),
        'count': pd.DataFrame(count, index=order, columns=order),
        'score': pd.DataFrame(mean, index=order, columns=order),
        'coverage_x': coverage[0],
        'coverage_y': coverage[1],
        'edges': pd.DataFrame({'x_index': ix, 'y_index': iy, 'score': score}),
        'match_rate': match_rate,
        'match_score':match_score
    }

def matching_table1(result, x, y, order, score_col='score'):
    keep = result.get('keep', slice(None))
    ix = _numpy(result['Inds'][0, keep])
    iy = _numpy(result['Inds'][1, keep])
    score = _numpy(result[score_col][keep])
    if  score_col == 'psp':
        score = score/score.max()
    order = list(order)
    cx = pd.Categorical(x, categories=order)
    cy = pd.Categorical(y, categories=order)

    edges = pd.DataFrame({
        'x_index': ix,
        'y_index': iy,
        'score':   score,
        'x_type':  cx[ix],
        'y_type':  cy[iy],
    })

    count = (
        pd.crosstab(edges['x_type'], edges['y_type'])
          .reindex(index=order, columns=order, fill_value=0)
    )

    mean = (
        edges.groupby(['x_type', 'y_type'], observed=True)['score']
             .mean()
             .unstack('y_type')
             .reindex(index=order, columns=order)
    )

    def _coverage(cat, idx_col):
        total   = pd.Series(cat).value_counts().reindex(order, fill_value=0)
        matched = (edges.drop_duplicates(idx_col)['x_type' if idx_col == 'x_index' else 'y_type']
                        .value_counts().reindex(order, fill_value=0))
        return pd.DataFrame({'total': total, 'matched': matched})

    coverage_x = _coverage(cx, 'x_index')
    coverage_y = _coverage(cy, 'y_index')
    match_rate = (cx[ix].values == cy[iy].values).sum()/len(ix)
    match_score=  (match_rate * (np.unique(ix).shape[0]/cx.shape[0]) * (np.unique(iy).shape[0]/cy.shape[0])) ** (1.0/3.0)
    return {
        'order':      order,
        'count':      count,
        'score':      mean,
        'coverage_x': coverage_x,
        'coverage_y': coverage_y,
        'edges':      edges[['x_index', 'y_index', 'score']],
        'match_rate': match_rate,
        'match_score':match_score
    }


def match_heatmap(rel, *, names=('source', 'target'), palette=None,
                       metric='count', cmap='YlOrRd', annotate=False,
                       annotate_score=False, marginal_kind='bar', vmin=0,
                       vmax=None, min_pairs=1, fontsize=9,
                       figsize=(14, 12), log1p=False, marginal_log1p=False,
                       gap=.04, cbar_gap=.06, marginal_size=.85, show=True,
                       save=None, dpi=None, **save_kargs):
    """Count/score heatmap with matched + unmatched stacked cell counts.

    log1p changes count colors; marginal_log1p changes the cell-count axes.
    Limits, ticks, annotations and returned values remain in original units.
    Gaps and marginal_size are in inches. Dots mark matched/original counts.
    save=None disables saving; otherwise the extension selects the format.
    dpi and **save_kargs are forwarded to Figure.savefig before showing.
    """
    order = rel['order']
    count = rel['count']
    supported = count >= min_pairs
    values = rel[metric].where(supported, 0 if metric == 'count' else np.nan)
    colors = _palette(order, palette)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('white')
    if vmax is None:
        vmax = 1 if metric == 'score' else max(count.to_numpy().max(), 1)
    norm = Normalize(vmin, vmax)
    if log1p and metric == 'count':
        norm = FuncNorm((np.log1p, np.expm1), vmin=vmin, vmax=vmax)

    n = len(order)
    cell = .38 if annotate_score else .32
    label_space = max(len(str(c)) for c in order) * fontsize * .56 / 72 + .4
    width = max(figsize[0], label_space + marginal_size + gap + n * cell + cbar_gap + .7)
    height = max(figsize[1], label_space + marginal_size + gap + n * cell + .55)
    heat_width = width - label_space - marginal_size - gap - cbar_gap - .7
    heat_height = height - label_space - marginal_size - gap - .55
    fig = plt.figure(figsize=(width, height), facecolor='white')
    grid = fig.add_gridspec(
        3, 5, width_ratios=[marginal_size, gap, heat_width, cbar_gap, .15],
        height_ratios=[marginal_size, gap, heat_height],
        left=label_space / width, right=1 - .55 / width,
        bottom=label_space / height, top=1 - .55 / height,
        wspace=0, hspace=0)
    ax = fig.add_subplot(grid[2, 2])
    source = fig.add_subplot(grid[2, 0])
    target = fig.add_subplot(grid[0, 2])
    cax = fig.add_subplot(grid[2, 4])
    image = ax.imshow(values, cmap=cm, norm=norm, aspect='auto')
    unit = 'Matched pairs' if metric == 'count' else 'Mean match score'
    colorbar = fig.colorbar(image, cax=cax, label=unit)
    if log1p and metric == 'count':
        ticks = np.expm1(np.linspace(np.log1p(vmin), np.log1p(vmax), 5))
        colorbar.set_ticks(np.unique(np.rint(ticks)))
    positions = np.arange(n)
    ax.set_xticks(positions, order, rotation=90, fontsize=fontsize)
    ax.set_yticks([])
    ax.set_xlabel(names[1])

    total_x = rel['coverage_x']['total'].to_numpy()
    total_y = rel['coverage_y']['total'].to_numpy()
    matched_x = rel['coverage_x']['matched'].to_numpy()
    matched_y = rel['coverage_y']['matched'].to_numpy()
    if marginal_kind == 'bar':
        source.barh(positions, matched_x, color=colors)
        source.barh(positions, total_x - matched_x, left=matched_x, color=colors, alpha=.22)
        target.bar(positions, matched_y, color=colors)
        target.bar(positions, total_y - matched_y, bottom=matched_y, color=colors, alpha=.22)
        pale_label = 'Unmatched cells'
    else:
        source.scatter(total_x, positions, c=colors, alpha=.22, s=25)
        source.scatter(matched_x, positions, c=colors, s=25)
        target.scatter(positions, total_y, c=colors, alpha=.22, s=25)
        target.scatter(positions, matched_y, c=colors, s=25)
        pale_label = 'Original cells'
    if marginal_log1p:
        source.set_xscale('function', functions=(np.log1p, np.expm1))
        target.set_yscale('function', functions=(np.log1p, np.expm1))
        ticks_x = np.unique(np.rint(np.expm1(np.linspace(0, np.log1p(total_x.max()), 4))))
        source.set_xticks(ticks_x)
        ticks_y = np.expm1(np.linspace(0, np.log1p(total_y.max()), 4))
    else:
        ticks_y = np.linspace(0, total_y.max(), 3)
    target.set_yticks(np.unique(np.rint(ticks_y))[1:])
    cell_unit = 'Cells (log1p)' if marginal_log1p else 'Cells'
    source.set(
        xlim=(total_x.max(), 0), ylim=(n - .5, -.5), xlabel=cell_unit,
        ylabel=names[0], yticks=positions, yticklabels=order)
    target.set(xlim=(-.5, n - .5), ylim=(0, total_y.max()), xticks=[], ylabel=cell_unit)
    source.set_xlabel(cell_unit, fontsize=fontsize-1)
    target.set_ylabel(cell_unit, fontsize=fontsize-1)
    source.tick_params(labelsize=fontsize)
    target.tick_params(labelsize=fontsize-1)
    target.yaxis.tick_right()
    target.yaxis.set_label_position('right')
    handles = [
        Patch(facecolor='.5', label='Matched cells'),
        Patch(facecolor='.5', alpha=.22, label=pale_label)]
    fig.legend(
        handles=handles, loc='lower right', frameon=False, fontsize=fontsize,
        bbox_to_anchor=(source.get_position().x1, target.get_position().y0),
        borderaxespad=0, handlelength=1.1, handletextpad=.4, labelspacing=.2)
    ax.marginal_axes = {'source': source, 'target': target}

    if annotate or annotate_score:
        for i, j in zip(*np.nonzero(supported.to_numpy())):
            value = values.iat[i, j]
            lines = []
            if annotate:
                lines.append(f'{value:g}' if metric == 'count' else f'{value:.2f}')
            if annotate_score and metric == 'count':
                score_text = f'{rel["score"].iat[i, j]:.2f}'.lstrip('0')
                lines.append(f'S={score_text}')
            rgb = np.asarray(cm(norm(value))[:3])
            brightness = rgb @ [.2126, .7152, .0722]
            ax.text(
                j, i, '\n'.join(lines), ha='center', va='center',
                fontsize=min(fontsize, 7 if annotate_score else 8),
                color='white' if brightness < .48 else '.15')
    fig.suptitle(f'{names[0]} → {names[1]} | {unit}', fontsize=fontsize + 2)
    _save_match(fig, save, dpi, save_kargs)
    if show:
        plt.show()
    return fig, ax, values


def match_sankey(relations, *, names=None,
                      kind='river', palette=None, cmap='viridis',
                      color_by='score', min_pairs=1, keep_max=False, vmin=None, vmax=None,
                      fontsize=9, figsize=(18, 12), alpha=.8, show=True,
                      save=None, dpi=None, auto_size=False,
                      node_gap=None, min_slot=None, node_pad=None,
                      node_layout='aligned', width_power=1.,
                      show_title=True, show_text=True, **save_kargs):
    """Plot adjacent matching relations across three time points.

    Node sizes use all positive counts; min_pairs only hides links.
    Categories without matches at a stage do not create nodes or zero ports.
    Compact spacing includes only categories present at that stage.
    keep_max retains the largest positive link per source below the threshold.
    Link colors represent original count, score, or source/target categories.
    Automatic color limits use finite values of displayed links.
    figsize is the exact canvas size in inches unless auto_size=True restores
    the original automatic enlargement. Small canvases bound layout margins
    without changing fontsize; reduce fontsize when many categories are shown.
    Avoid bbox_inches='tight' when the saved dimensions must match figsize.
    River node_gap and min_slot are per-category fractions of plot height;
    None uses the original .16/n gap and .44/n minimum slot. Set min_slot=0
    to remove minimum-height padding around small categories. By default, stages
    share the largest category envelope, leaving space around smaller bars.
    Plotly node_pad is the native node spacing in pixels; None uses fontsize*1.5.
    River node_layout='aligned' shares category centers across stages;
    'compact' packs each stage separately, retaining one common width scale.
    width_power transforms each positive edge count for display only: 1 keeps
    linear widths, 0.5 uses square roots to make small links more visible.
    Display node totals and ports use the same transformed edge weights.
    Raw matrices, count colors, filtering, hover counts and statistics are
    unchanged. Nonlinear widths are noted in the footer when show_text=True.
    show_title controls the overall title; show_text controls the bottom
    explanatory note. Category labels, stage names and hover text are retained.
    save=None disables saving; otherwise the extension selects the format.
    River plots forward dpi and **save_kargs to Figure.savefig.
    Plotly exports HTML/HTM, JSON, or static PNG/JPEG/WebP/SVG/PDF files.
    Raster dpi sets scale=dpi/100 unless scale is supplied explicitly; it
    controls pixel resolution, not DPI metadata. HTML/JSON/vector exports
    ignore dpi. Other save options go to the selected native export method.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize, to_hex, to_rgba
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    if node_layout not in ('aligned', 'compact'):
        raise ValueError("node_layout must be 'aligned' or 'compact'.")
    if width_power <= 0:
        raise ValueError('width_power must be positive.')
    order = relations[0]['order']
    if names is None:
        names = list( map(str, range(len(relations) + 1)) )
    
    n = len(order)
    matrices = []
    scores = []
    for rel in relations:
        count = rel['count'].reindex(index=order, columns=order)
        matrices.append(count.where(count > 0, 0))
        scores.append(
            rel['score'].reindex(index=order, columns=order).to_numpy()
        )

    incoming = np.zeros((3, n))
    outgoing = np.zeros((3, n))
    for k, matrix in enumerate(matrices):
        outgoing[k] = matrix.sum(1)
        incoming[k + 1] = matrix.sum(0)
    amounts = np.maximum(incoming, outgoing)
    active = amounts > 0
    display_matrices = [matrix.to_numpy(dtype=float) ** width_power for matrix in matrices]
    display_incoming = np.zeros_like(incoming)
    display_outgoing = np.zeros_like(outgoing)
    for k, matrix in enumerate(display_matrices):
        display_outgoing[k] = matrix.sum(1)
        display_incoming[k + 1] = matrix.sum(0)
    display_amounts = np.maximum(display_incoming, display_outgoing)

    if palette is None:
        colors = plt.get_cmap('tab20')(np.linspace(0, 1, n))
    else:
        colors = [to_rgba(palette[label]) for label in order]

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('.65')
    shown = []
    for matrix in matrices:
        a = matrix.to_numpy()
        keep = (a > 0) & (a >= min_pairs)
        if keep_max:
            i = np.arange(n)
            j = a.argmax(axis=1)
            keep[i, j] |= a[i, j] > 0
        shown.append(keep)
    score_values = np.concatenate([
        s[keep & np.isfinite(s)]
        for s, keep in zip(scores, shown)
    ])
    count_values = np.concatenate([
        matrix.to_numpy()[keep] for matrix, keep in zip(matrices, shown)
    ])
    color_values = count_values if color_by == 'count' else score_values
    color_label = 'Matched pairs' if color_by == 'count' else 'Mean score'
    if vmin is None:
        vmin = color_values.min() if color_values.size else 0.
    if vmax is None:
        vmax = color_values.max() if color_values.size else vmin

    norm = Normalize(vmin, vmax, clip=True)
    show_scale = color_by in ('score', 'count') and color_values.size > 0 and vmax > vmin
    color_samples = cm(np.linspace(0, 1, 256))
    color_samples[:, 3] = alpha
    legend_cmap = ListedColormap(color_samples)

    def css(rgba):
        return 'rgba(%d,%d,%d,%g)' % (
            *[round(c * 255) for c in rgba[:3]], alpha
        )

    # Reserve label space within the requested canvas.
    width, height = figsize
    label_w = max(len(str(c)) for c in order) * fontsize * .56 / 72 + .35
    top, bottom = .8, .65
    if auto_size:
        width = max(width, 2 * label_w + 8)
        height = max(height, n * fontsize * 1.9 / 72 + 1.7)
        right = label_w + (1.2 if color_by in ('score', 'count') else .3)
    else:
        reserve = min(1.2 if show_scale else .3, .2 * width)
        label_w = min(label_w, (.65 * width - reserve) / 2)
        right = label_w + reserve
        scale = min(1., .35 * height / (top + bottom))
        top, bottom = top * scale, bottom * scale
    plot_width = width - label_w - right

    gap = .16 / n if node_gap is None else node_gap
    minimum = .44 / n if min_slot is None else min_slot
    envelope = display_amounts if node_layout == 'compact' else display_amounts.max(0)
    slot_active = envelope > 0
    slot_count = slot_active.sum(axis=-1, keepdims=True)
    gap_count = np.maximum(slot_count - 1, 0)
    available = 1 - gap * gap_count
    if gap < 0 or minimum < 0 or np.any(available <= 0) or np.any(minimum * slot_count > available):
        raise ValueError('node_gap and min_slot must be nonnegative and fit within plot height.')
    low = 0.
    high = np.min(available / np.maximum(envelope.sum(axis=-1, keepdims=True), 1.))
    for _ in range(50):
        mid = (low + high) / 2
        if np.any(np.where(slot_active, np.maximum(envelope * mid, minimum), 0).sum(axis=-1, keepdims=True) > available):
            high = mid
        else:
            low = mid

    slots = np.where(slot_active, np.maximum(envelope * low, minimum), 0)
    gaps = gap * slot_active
    offset = (1 - slots.sum(axis=-1, keepdims=True) - gap * gap_count) / 2
    centers = offset + np.cumsum(slots + gaps, axis=-1) - slots / 2 - gaps
    centers = np.where(slot_active, centers, .5)
    if node_layout == 'aligned':
        centers = np.tile(centers, (3, 1))
    heights = display_amounts * low
    tops = centers - heights / 2
    xs = [0., .5, 1.]
    bar_w = .0125

    title = ' → '.join(names) + ' | Matched pairs'
    color_note = 'Link color: mean score; gray = missing score'
    if color_by == 'count':
        color_note = 'Link color: original matched-pair count'
    if color_by in ('source', 'target'):
        color_note = f'Link color: {color_by} celltype of each interval'
    if color_by in ('score', 'count') and color_values.size and vmin == vmax:
        color_note = f'{color_label} = {vmin:.6g}'
    color_note += f' | Bars: all matches; links: count >= {min_pairs:g}'
    if keep_max:
        color_note += ' + largest positive link per source'
    if width_power != 1:
        color_note += f' | Widths: count^{width_power:g} (display only)'

    polygons = []
    fills = []
    rows = []
    t = np.linspace(0, 1, 40)
    smooth = t * t * (3 - 2 * t)

    for k, matrix in enumerate(matrices):
        a = matrix.to_numpy()
        display = display_matrices[k]
        if kind == 'river':
            start = centers[k, :, None] - display_outgoing[k, :, None] * low / 2
            start = start + (display.cumsum(1) - display) * low
            end = centers[k + 1, None, :] - display_incoming[k + 1, None, :] * low / 2
            end = end + (display.cumsum(0) - display) * low

        for i, j in zip(*np.nonzero(a)):
            rgba = colors[j] if color_by == 'target' else colors[i]
            if color_by in ('score', 'count'):
                score = a[i, j] if color_by == 'count' else scores[k][i, j]
                rgba = (
                    cm(norm(score))
                    if np.isfinite(score) else (.65, .65, .65, 1)
                )

            visible = bool(shown[k][i, j])
            color = css(rgba) if visible else 'rgba(0,0,0,0)'
            rows.append((
                k, k * n + i, (k + 1) * n + j,
                a[i, j], scores[k][i, j], color, visible
            ))

            if kind == 'river' and visible:
                x = xs[k] + bar_w / 2 + (.5 - bar_w) * t
                y = start[i, j] + (end[i, j] - start[i, j]) * smooth
                polygons.append(np.column_stack([
                    np.r_[x, x[::-1]],
                    np.r_[y, (y + display[i, j] * low)[::-1]],
                ]))
                fills.append((*rgba[:3], alpha))

    all_links = pd.DataFrame(
        rows,
        columns=['stage', 'source', 'target', 'count', 'score', 'color', 'shown']
    )
    links = all_links.loc[all_links['shown'].astype(bool)].copy()
    node_ids = np.flatnonzero(active.ravel())

    if kind == 'river':
        fig, ax = plt.subplots(figsize=(width, height), facecolor='white')
        fig.subplots_adjust(
            left=label_w / width, right=1 - right / width,
            top=1 - top / height, bottom=bottom / height
        )
        ax.set(xlim=(-.025, 1.025), ylim=(1.015, -.015))
        ax.axis('off')
        ax.add_collection(
            PolyCollection(polygons, facecolors=fills, edgecolors='none')
        )

        for k, name in enumerate(names):
            for j, label in enumerate(order):
                if not active[k, j]:
                    continue
                ports = (display_incoming[k, j], display_outgoing[k, j])
                if k in (0, 2):
                    ports = (display_amounts[k, j], display_amounts[k, j])

                for side, value in enumerate(ports):
                    if value <= 0:
                        continue
                    x = xs[k] + (side - 1) * bar_w / 2
                    y = centers[k, j] - value * low / 2
                    ax.add_patch(Rectangle(
                        (x, y), bar_w / 2, value * low,
                        color=colors[j], linewidth=0
                    ))

                if k in (0, 2) and amounts[k, j] > 0:
                    ax.text(
                        -.025 if k == 0 else 1.025,
                        centers[k, j], str(label),
                        ha='right' if k == 0 else 'left',
                        va='center', fontsize=fontsize, clip_on=False
                    )

            ax.text(
                xs[k], -.03, name, ha='center',
                va='bottom', fontsize=fontsize + 1
            )

        if show_title:
            fig.suptitle(title, fontsize=fontsize + 2, y=.99, wrap=not auto_size)
        if show_text:
            fig.text(label_w / width, .02, color_note, fontsize=fontsize - 1,
                     wrap=not auto_size, va='bottom' if width_power != 1 else 'baseline')

        if show_scale:
            cbar_x = (width - .5 if auto_size else width - right + label_w + .1)
            cax = fig.add_axes([cbar_x / width, .14, .15 / width, .7])
            colorbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=legend_cmap),
                cax=cax, label=color_label,
                ticks=np.linspace(vmin, vmax, 5)
            )
            if not auto_size:
                colorbar.ax.tick_params(labelsize=fontsize)
                colorbar.ax.yaxis.label.set_size(fontsize)

    else:
        import plotly.graph_objects as go

        # Retain all link values to preserve native Sankey node sizes.
        node_map = {node: index for index, node in enumerate(node_ids)}
        node_data = [
            [
                names[node // n],
                incoming.ravel()[node],
                outgoing.ravel()[node],
                str(order[node % n])
            ]
            for node in node_ids
        ]
        node_centers = np.full((3, n), np.nan)
        for k in range(3):
            present = active[k]
            weights = display_amounts[k, present]
            node_centers[k, present] = (
                np.cumsum(weights) - weights / 2
            ) / weights.sum()

        stage_x = [.001, .5, .999]

        # Draw visible links above transparent links.
        all_links = all_links.sort_values('shown', kind='stable')

        fig = go.Figure(go.Sankey(
            arrangement='fixed', valueformat=',.0f',
            node=dict(
                label=[''] * len(node_ids),
                color=[to_hex(colors[node % n]) for node in node_ids],
                x=[stage_x[node // n] for node in node_ids],
                y=node_centers.ravel()[node_ids].tolist(),
                align='center', pad=fontsize * 1.5 if node_pad is None else node_pad,
                thickness=14,
                line=dict(width=0), customdata=node_data,
                hovertemplate=(
                    '%{customdata[0]} | %{customdata[3]}'
                    '<br>Incoming: %{customdata[1]:,.0f}'
                    '<br>Outgoing: %{customdata[2]:,.0f}<extra></extra>'
                )
            ),
            link=dict(
                source=[node_map[node] for node in all_links['source']],
                target=[node_map[node] for node in all_links['target']],
                value=(all_links['count'].tolist() if width_power == 1 else [
                    display_matrices[row.stage][row.source % n, row.target % n]
                    for row in all_links.itertuples()
                ]),
                color=all_links['color'].tolist(),
                hovercolor=all_links['color'].tolist(),
                line=dict(width=0),
                label=[
                    f'{names[k]} → {names[k + 1]}'
                    for k in all_links['stage']
                ],
                customdata=[
                    (
                        f'{names[row.stage]} → {names[row.stage + 1]}'
                        f'<br>{order[row.source % n]} → {order[row.target % n]}'
                        f'<br>Pairs: {row.count:,.0f}'
                        f'<br>Mean score: {row.score:.4g}'
                    )
                    if row.shown else ''
                    for row in all_links.itertuples()
                ],
                hovertemplate='%{customdata}<extra></extra>'
            )
        ))
        ax = None
        fig.update_layout(
            width=round(width * 100), height=round(height * 100),
            title=title if show_title else None, font_size=round(fontsize * 4 / 3),
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(
                l=round(label_w * 100), r=round(right * 100),
                t=round(top * 100), b=round(bottom * 100), autoexpand=False
            ),
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )

        for node in node_ids:
            k, j = divmod(node, n)
            if k in (0, 2):
                fig.add_annotation(
                    x=-.015 if k == 0 else 1.015,
                    y=1 - node_centers[k, j],
                    xref='paper', yref='paper',
                    text=str(order[j]), showarrow=False,
                    yanchor='middle',
                    xanchor='right' if k == 0 else 'left'
                )

        for x, name in zip(stage_x, names):
            fig.add_annotation(
                x=x, y=1.04, xref='paper', yref='paper',
                text=name, showarrow=False
            )

        if show_text:
            fig.add_annotation(
                x=0, y=-.065, xref='paper', yref='paper',
                text=color_note, showarrow=False,
                xanchor='left', font_size=fontsize
            )

        if show_scale:
            stops = [
                [float(q), css(cm(q))]
                for q in np.linspace(0, 1, 33)
            ]
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                showlegend=False, hoverinfo='skip',
                marker=dict(
                    color=[vmin], cmin=vmin, cmax=vmax,
                    colorscale=stops, showscale=True,
                    colorbar=dict(
                        title=color_label, thickness=14,
                        x=1 + (label_w + .1) / plot_width,
                        xanchor='left',
                        tickvals=np.linspace(vmin, vmax, 5)
                    )
                )
            ))

    data = dict(
        order=order, matrices=matrices,
        incoming=incoming, outgoing=outgoing,
        centers=centers, heights=heights, tops=tops, scale=low,
        links=links, all_links=all_links, node_ids=node_ids,
        score_range=((score_values.min(), score_values.max()) if color_by == 'count'
                     else (vmin, vmax)) if score_values.size else None,
        color_range=(vmin, vmax) if color_by in ('score', 'count') and color_values.size else None,
        display_matrices=display_matrices,
        display_incoming=display_incoming, display_outgoing=display_outgoing,
        width_power=width_power, node_layout=node_layout
    )
    if kind == 'sankey':
        data['node_centers'] = node_centers

    _save_match(fig, save, dpi, save_kargs)
    if show:
        if kind == 'river':
            plt.show()
        else:
            fig.show()

    return fig, ax, data
