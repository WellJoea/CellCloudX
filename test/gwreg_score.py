import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
import scipy.sparse as ssp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
import scipy.sparse as ssp
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def barplot3d(
    data,
    x='Time',
    y='knn',
    z='4_normal',
    hue='group',
    x_order=None,
    y_order=None,
    hue_order=None,
    palette=None,
    aggfunc='mean',

    # 画布与三轴比例
    figsize=(13, 8),
    box_aspect=(1.1, 1.1, 1.1,),

    # 视角
    elev=25,
    azim=-55,
    roll=0,

    # 柱体大小
    bar_width=0.32,
    bar_depth=0.68,

    # 间距
    group_gap=0.05,
    x_gap=0.30,
    y_gap=0.25,

    # 柱体样式
    baseline=0,
    alpha=0.88,
    edgecolor=',,',
    linewidth=0.6,
    shade=True,
    bar_kwargs=None,

    # 坐标轴
    xlabel=None,
    ylabel=None,
    zlabel=None,
    labelpad=(12, 10, 8),
    xtick_rotation=30,
    zlim=None,

    # 图例
    legend=True,
    legend_kwargs=None,

    # 输出
    save=None,
    dpi=1200,
    show=True,
):
    """
    绘制三维分类柱状图。

    Parameters
    ----------
    bar_width : float
        每根柱子沿 x 轴方向的宽度。
    bar_depth : float
        每根柱子沿 y 轴方向的深度。
    group_gap : float
        同一个 x 类别内，不同 hue 柱子之间的间距。
    x_gap : float
        相邻 x 类别柱组之间的间距。
    y_gap : float
        相邻 y 类别之间的间距。
    figsize : tuple
        Matplotlib 画布的宽和高。
    box_aspect : tuple
        三维坐标框的 x、y、z 长宽高比例。
    """

    def get_order(values, order=None):
        if order is not None:
            return list(order)

        observed = values.dropna()

        if isinstance(values.dtype, pd.CategoricalDtype):
            observed_set = set(observed)
            return [
                value
                for value in values.cat.categories
                if value in observed_set
            ]

        order = observed.drop_duplicates().tolist()

        if pd.api.types.is_numeric_dtype(values):
            order = sorted(order)

        return order

    required = [x, y, z, hue]
    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(f'Missing columns: {missing}')

    if bar_width <= 0 or bar_depth <= 0:
        raise ValueError(
            'bar_width and bar_depth must be positive.'
        )

    if min(group_gap, x_gap, y_gap) < 0:
        raise ValueError(
            'group_gap, x_gap and y_gap must be non-negative.'
        )

    plot_data = data[required].dropna().copy()

    x_order = get_order(plot_data[x], x_order)
    y_order = get_order(plot_data[y], y_order)
    hue_order = get_order(plot_data[hue], hue_order)

    if not x_order or not y_order or not hue_order:
        raise ValueError('No data remain for plotting.')

    plot_df = (
        plot_data
        .groupby(
            [x, y, hue],
            observed=True,
            sort=False
        )[z]
        .agg(aggfunc)
        .rename('_height')
        .reset_index()
        .dropna(subset=['_height'])
    )

    xmap = {
        value: index
        for index, value in enumerate(x_order)
    }
    ymap = {
        value: index
        for index, value in enumerate(y_order)
    }
    hmap = {
        value: index
        for index, value in enumerate(hue_order)
    }

    plot_df['_xi'] = plot_df[x].map(xmap)
    plot_df['_yi'] = plot_df[y].map(ymap)
    plot_df['_hi'] = plot_df[hue].map(hmap)

    plot_df = plot_df.dropna(
        subset=['_xi', '_yi', '_hi']
    )

    if plot_df.empty:
        raise ValueError(
            'No rows match x_order, y_order and hue_order.'
        )

    if palette is None:
        tab10 = plt.get_cmap('tab10')
        color_map = {
            value: tab10(index % tab10.N)
            for index, value in enumerate(hue_order)
        }

    elif isinstance(palette, dict):
        missing_colors = [
            value for value in hue_order
            if value not in palette
        ]

        if missing_colors:
            raise KeyError(
                f'Missing palette colors for: {missing_colors}'
            )

        color_map = {
            value: palette[value]
            for value in hue_order
        }

    else:
        palette = list(palette)

        if len(palette) == 0:
            raise ValueError('palette cannot be empty.')

        color_map = {
            value: palette[index % len(palette)]
            for index, value in enumerate(hue_order)
        }

    # 同一个 x 类别内，全部 hue 柱体占据的总宽度
    group_span = (
        len(hue_order) * bar_width
        + (len(hue_order) - 1) * group_gap
    )

    x_step = group_span + x_gap
    y_step = bar_depth + y_gap

    xi = plot_df['_xi'].to_numpy(dtype=float)
    yi = plot_df['_yi'].to_numpy(dtype=float)
    hi = plot_df['_hi'].to_numpy(dtype=float)
    height = plot_df['_height'].to_numpy(dtype=float)

    bx = (
        xi * x_step
        + hi * (bar_width + group_gap)
    )
    by = yi * y_step
    bz = np.full_like(height, float(baseline))

    colors = [
        color_map[value]
        for value in plot_df[hue]
    ]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    options = dict(
        color=colors,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        shade=shade,
        antialiased=True
    )
    options.update(bar_kwargs or {})

    ax.bar3d(
        bx,
        by,
        bz,
        np.full_like(height, bar_width),
        np.full_like(height, bar_depth),
        height - baseline,
        **options
    )

    # 刻度放在每组柱体中心
    xticks = (
        np.arange(len(x_order)) * x_step
        + group_span / 2
    )
    yticks = (
        np.arange(len(y_order)) * y_step
        + bar_depth / 2
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        x_order,
        rotation=xtick_rotation,
        ha='right'
    )

    ax.set_yticks(yticks)
    ax.set_yticklabels(y_order)

    ax.set_xlim(
        -x_gap / 2,
        (len(x_order) - 1) * x_step
        + group_span
        + x_gap / 2
    )

    ax.set_ylim(
        -y_gap / 2,
        (len(y_order) - 1) * y_step
        + bar_depth
        + y_gap / 2
    )

    if zlim is None:
        zmin = min(
            float(baseline),
            float(np.nanmin(height))
        )
        zmax = max(
            float(baseline),
            float(np.nanmax(height))
        )

        zrange = zmax - zmin
        margin = 0.08 * (zrange if zrange > 0 else 1)

        ax.set_zlim(zmin, zmax + margin)

    else:
        ax.set_zlim(*zlim)

    ax.set_xlabel(
        x if xlabel is None else xlabel,
        labelpad=labelpad[0]
    )
    ax.set_ylabel(
        y if ylabel is None else ylabel,
        labelpad=labelpad[1]
    )
    ax.set_zlabel(
        z if zlabel is None else zlabel,
        labelpad=labelpad[2]
    )

    ax.view_init(
        elev=elev,
        azim=azim,
        roll=roll
    )

    ax.set_box_aspect(box_aspect)

    if legend:
        handles = [
            Patch(
                facecolor=color_map[value],
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                label=value
            )
            for value in hue_order
        ]

        options = dict(
            title=hue,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            frameon=False
        )
        options.update(legend_kwargs or {})

        ax.legend(
            handles=handles,
            **options
        )

    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.grid(False)

    fig.subplots_adjust(
        left=0.02,
        right=0.82 if legend else 0.98,
        bottom=0.08,
        top=0.98
    )

    if save is not None:
        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig, ax, plot_df


def plot_paired_trend(
    df,
    y_group="3_d",
    x_group="basis",
    pair_col=None,
    basis_order=None,
    time=None,
    figsize=(7, 6),

    # trim 参数
    trim_abs=None,
    trim_ylim=None,
    trim_quantile=None,

    # log 平移参数
    use_shift = True,
    min_shifted=1.0,

    # 颜色与线条参数
    cmap_name="viridis",
    line_colors=None,
    point_colors=None,
    linewidth=0.9,
    line_alpha=0.80,
    scatter_size=14,
    scatter_alpha=0.90,
    scatter_linewidth=0.5,
    set_title= True,
    set_label= True,

    # mean ± SEM 参数
    show_mean_sem=True,
    mean_color="black",
    mean_linewidth=1.0,
    mean_markersize=5,
    capsize=4,

    # 右侧文字参数
    show_right_label=True,
    label_top_n=None,
    text_dx=0.10,
    text_fontsize=8,
    text_angle=0,
    text_alpha=0.95,
    text_color_by_strength=True,

    # 坐标轴参数
    use_log=True,
    show_zero_line=True,
    xlabel="Basis",
    ylabel=None,
    title=None,
    rotation=0,

    # 图例与保存
    show_colorbar=True,
    legend=True,
    save=None,
    ax=None, fig=None, show=True,
):
    """
    Paired trend plot with optional trimming, integer upward shifting, log y-axis,
    viridis-colored paired lines, and right-side index labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    y_group : str
        Column name for raw paired difference, e.g. "3_d".
    x_group : str
        Column name for method / basis.
    pair_col : str or None
        Column name for paired sample ID. If None, dataframe index is used.
    basis_order : list or None
        Order of x-axis categories.
    trim_abs : float or None
        Keep rows with abs(y_group) <= trim_abs.
    trim_ylim : tuple or None
        Keep rows with trim_ylim[0] <= y_group <= trim_ylim[1].
    trim_quantile : tuple or None
        Keep rows within quantile range, e.g. (0.01, 0.99).
    text_angle : float
        Rotation angle of right-side text labels.
    """

    # =========================
    # 0. Copy data
    # =========================
    plot_df = df.copy()

    if pair_col is None:
        plot_df["pair_id"] = plot_df.index.astype(str)
    else:
        plot_df["pair_id"] = plot_df[pair_col].astype(str)

    y_shift_group = f"{y_group}_shift"

    plot_df[y_group] = plot_df[y_group].astype(float)

    # =========================
    # 1. basis 顺序
    # =========================
    if basis_order is None:
        if hasattr(plot_df[x_group], "cat"):
            basis_order = plot_df[x_group].cat.categories.tolist()
        else:
            basis_order = plot_df[x_group].drop_duplicates().tolist()

    x_map = {b: i for i, b in enumerate(basis_order)}

    # =========================
    # 2. trim 离散点
    # =========================
    raw_n = plot_df.shape[0]

    if trim_ylim is not None:
        y_low, y_high = trim_ylim
    elif trim_abs is not None:
        y_low, y_high = -float(trim_abs), float(trim_abs)
    elif trim_quantile is not None:
        q_low, q_high = trim_quantile
        y_low, y_high = plot_df[y_group].quantile([q_low, q_high]).to_numpy()
    else:
        y_low, y_high = -np.inf, np.inf

    plot_df = plot_df[
        plot_df[y_group].between(y_low, y_high)
    ].copy()

    trim_n = plot_df.shape[0]

    print(f"Raw rows = {raw_n}")
    print(f"Rows after trim = {trim_n}")
    print(f"Trim range = [{y_low:.4f}, {y_high:.4f}]")

    if trim_n == 0:
        raise ValueError("No data left after trimming. Please loosen trim_abs / trim_ylim / trim_quantile.")

    # =========================
    # 3. 整体向上平移，使 log 纵坐标可用
    # =========================
    y_raw = plot_df[y_group].astype(float)
    y_min = np.nanmin(y_raw)

    if use_shift:
        y_shift = max(0, int(np.ceil(min_shifted - y_min)))
        print(f"Original min after trim = {y_min:.4f}")
        print(f"Integer shift = {y_shift}")
    else:
        y_shift = 0
    plot_df[y_shift_group] = plot_df[y_group] + y_shift


    # =========================
    # 4. 计算每条趋势线的变化强度
    # =========================
    trend_strength = (
        plot_df
        .dropna(subset=[x_group, y_group])
        .groupby("pair_id")[y_group]
        .agg(lambda x: np.nanmax(x) - np.nanmin(x))
    )

    vmin = trend_strength.min()
    vmax = trend_strength.max()

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # =========================
    # 5. 绘图
    # =========================
    sns.set_theme(style="ticks")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # =========================
    # 5.1 个体配对趋势线
    # =========================
    for pid, sub in plot_df.groupby("pair_id"):
        sub = sub.dropna(subset=[x_group, y_shift_group]).copy()
        sub = sub[sub[x_group].isin(basis_order)]
        sub["x_pos"] = sub[x_group].map(x_map)
        sub = sub.sort_values("x_pos")

        if sub.shape[0] >= 2:
            strength = trend_strength.loc[pid]
            color = cmap(norm(strength))
            if line_colors is None:
                lcolor = color
            else:
                lcolor =sub[line_colors].values[-1]

            if point_colors is None:
                pcolor = color
            else:
                pcolor = sub[point_colors].values
        
            ax.plot(
                sub["x_pos"],
                sub[y_shift_group],
                color=lcolor,
                linewidth=linewidth,
                alpha=line_alpha,
                zorder=1
            )

            ax.scatter(
                sub["x_pos"],
                sub[y_shift_group],
                color='none',
                s=scatter_size,
                edgecolors=pcolor ,
                linewidth=scatter_linewidth,
                alpha=scatter_alpha,
                zorder=2
            )

    # =========================
    # 5.2 均值 ± SEM
    # =========================
    if show_mean_sem:
        summary = (
            plot_df
            .groupby(x_group, observed=False)[y_shift_group]
            .agg(["mean", "sem"])
            .reindex(basis_order)
        )

        x = np.arange(len(basis_order))

        mean_y = summary["mean"].to_numpy()
        sem_y = summary["sem"].fillna(0).to_numpy()

        if use_log:

            # log scale 下不能 <=0
            lower_y = np.maximum(
                mean_y - sem_y,
                1e-8
            )

            upper_y = mean_y + sem_y

            yerr = np.vstack([
                mean_y - lower_y,
                upper_y - mean_y
            ])

        else:
            yerr = sem_y
    

        ax.errorbar(
            x,
            mean_y,
            yerr=yerr,
            marker="o",
            markersize=mean_markersize,
            linewidth=mean_linewidth,
            capsize=capsize,
            color=mean_color,
            zorder=5,
            # label="Mean ± SEM"
        )

    # =========================
    # 5.3 原始 0 线
    # =========================
    if show_zero_line:
        ax.axhline(
            y_shift,
            color="black",
            linewidth=1,
            linestyle="--",
            alpha=1.0,
            zorder=3,
            # label="Original 0 line"
        )

    # =========================
    # 5.4 右侧 index / pair_id 标签
    # =========================
    label_df_all = None
    if show_right_label:
        right_basis = basis_order[-1]
        right_x = x_map[right_basis]

        label_df = plot_df[
            plot_df[x_group].eq(right_basis)
        ].dropna(subset=[y_shift_group]).copy()

        label_df = label_df.drop_duplicates(subset=["pair_id"])

        label_df = label_df.sort_values(y_shift_group)
        label_df_all = label_df.copy()
        if label_top_n is not None:
            top_ids = (
                trend_strength
                .sort_values(ascending=False)
                .head(label_top_n)
                .index
            )
            label_df = label_df[label_df["pair_id"].isin(top_ids)]

        # label_df = label_df.sort_values(y_shift_group)

        for _, row in label_df.iterrows():
            pid = row["pair_id"]
            y_value = row[y_shift_group]

            if text_color_by_strength and pid in trend_strength.index:
                text_color = cmap(norm(trend_strength.loc[pid]))
            else:
                text_color = "black"

            ax.text(
                right_x + text_dx,
                y_value,
                str(pid),
                fontsize=text_fontsize,
                rotation=text_angle,
                rotation_mode="anchor",
                va="center",
                ha="left",
                color=text_color,
                alpha=text_alpha,
                clip_on=False,
                zorder=10
            )

        # 给右侧文字留空间
        ax.set_xlim(-0.5, len(basis_order) - 0.5 + 1.0)
    else:
        ax.set_xlim(-0.5, len(basis_order) - 0.5)

    # =========================
    # 6. 坐标轴
    # =========================
    if use_log:
        ax.set_yscale("log")

    x = np.arange(len(basis_order))
    ax.set_xticks(x)
    ax.set_xticklabels(
        basis_order,
        rotation=rotation,
        fontdict={'size': 8},
        ha="right")

    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = f"Shifted score difference: {y_group} + {y_shift}"
    if set_label:
        ax.set_ylabel(ylabel)

    if title is None:
        if time is None:
            title = f"Paired trend across {x_group}"
        else:
            title = f"Paired trend across {x_group} at {time}"
    if set_title:
        ax.set_title(title)

    ax.grid(False)

    # 四周边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # =========================
    # 7. 颜色条
    # =========================
    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        if set_label:
            cbar.set_label("Trend strength: max - min")

    if legend:
        ax.legend(frameon=False)
    
    if show:
        plt.tight_layout()
        if save is not None:
            fig.savefig(f"{save}.paired_trend_log.pdf", transparent=True, format="pdf")
            fig.savefig(f"{save}.paired_trend_log.png", dpi=900, bbox_inches="tight")
            plot_df.to_csv(f"{save}.paired_trend_log.data.csv", index=True)

        plt.show()
        return fig, ax, plot_df, trend_strength, label_df_all



from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def _adjust_pvalues(pvalues, method="holm"):
    """Adjust finite p-values while preserving their original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]
    m = p.size

    if m == 0:
        return adjusted

    method = "none" if method is None else str(method).lower()
    if method in {"none", "raw"}:
        q = p
    elif method in {"bonferroni", "bonf"}:
        q = np.minimum(p * m, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        ranked = p[order]
        ranked = np.maximum.accumulate((m - np.arange(m)) * ranked)
        q = np.empty(m, dtype=float)
        q[order] = np.minimum(ranked, 1.0)
    elif method in {"fdr_bh", "bh"}:
        order = np.argsort(p)
        ranked = p[order]
        ranked = ranked * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty(m, dtype=float)
        q[order] = np.minimum(ranked, 1.0)
    else:
        raise ValueError(
            "stat_correction must be 'holm', 'bonferroni', "
            "'fdr_bh' or None."
        )

    adjusted[valid] = q
    return adjusted


def _pvalue_text(pvalue, mode="star"):
    if not np.isfinite(pvalue):
        return "NA"

    if pvalue < 1e-4:
        star = "****"
    elif pvalue < 1e-3:
        star = "***"
    elif pvalue < 1e-2:
        star = "**"
    elif pvalue < 0.05:
        star = "*"
    else:
        star = "ns"

    ptext = "p<1e-4" if pvalue < 1e-4 else f"p={pvalue:.3g}"
    if mode == "star":
        return star
    if mode == "pvalue":
        return ptext
    if mode == "both":
        return f"{star} ({ptext})"
    raise ValueError("stat_format must be 'star', 'pvalue' or 'both'.")


def _paired_test(x, y, method="wilcoxon", alternative="two-sided"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if x.size < 2:
        return np.nan, np.nan, int(x.size)

    difference = x - y
    if np.allclose(difference, 0):
        return 0.0, 1.0, int(x.size)

    method = str(method).lower()
    if method in {"ttest", "t-test", "paired_t"}:
        result = stats.ttest_rel(
            x,
            y,
            alternative=alternative,
            nan_policy="omit",
        )
    elif method in {"wilcoxon", "wilcox"}:
        result = stats.wilcoxon(
            x,
            y,
            alternative=alternative,
            zero_method="wilcox",
            method="auto",
        )
    else:
        raise ValueError("stat_test must be 'ttest' or 'wilcoxon'.")

    return float(result.statistic), float(result.pvalue), int(x.size)


def _repeated_measures_anova(values):
    """Balanced one-way repeated-measures ANOVA for complete pairs."""
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    if n < 2 or k < 2:
        return np.nan, np.nan

    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    ss_group = n * np.sum((values.mean(axis=0) - grand) ** 2)
    ss_subject = k * np.sum((values.mean(axis=1) - grand) ** 2)
    ss_error = max(ss_total - ss_group - ss_subject, 0.0)

    df_group = k - 1
    df_error = (n - 1) * (k - 1)
    ms_group = ss_group / df_group
    ms_error = ss_error / df_error

    if ms_error <= np.finfo(float).eps:
        if ms_group <= np.finfo(float).eps:
            return 0.0, 1.0
        return np.inf, 0.0

    statistic = ms_group / ms_error
    pvalue = stats.f.sf(statistic, df_group, df_error)
    return float(statistic), float(pvalue)


def _resolve_stat_pairs(groups, stat_pairs):
    if len(groups) < 2:
        return []
    if stat_pairs is None or stat_pairs == "all":
        return list(combinations(groups, 2))
    if stat_pairs == "adjacent":
        return list(zip(groups[:-1], groups[1:]))

    pairs = [tuple(pair) for pair in stat_pairs]
    for first, second in pairs:
        if first not in groups or second not in groups or first == second:
            raise ValueError(f"Invalid stat pair: {(first, second)}")
    return pairs


def _paired_statistics(
    plot_df,
    y_group,
    x_group,
    basis_order,
    stat_test,
    stat_pairs,
    stat_correction,
    stat_alternative,
    stat_global,
):
    values = plot_df[["pair_id", x_group, y_group]].dropna().copy()
    duplicated = values.duplicated(["pair_id", x_group], keep=False)
    if duplicated.any():
        raise ValueError(
            "Each pair_id must have at most one value in each group "
            "for paired statistical testing."
        )

    wide = values.pivot(index="pair_id", columns=x_group, values=y_group)
    groups = [group for group in basis_order if group in wide.columns]
    rows = []

    method = str(stat_test).lower()
    if stat_global and len(groups) > 2:
        complete = wide[groups].dropna()
        matrix = complete.to_numpy(dtype=float)

        if matrix.shape[0] >= 2:
            if method in {"wilcoxon", "wilcox"}:
                if np.allclose(matrix - matrix[:, [0]], 0):
                    statistic, pvalue = 0.0, 1.0
                else:
                    result = stats.friedmanchisquare(*matrix.T)
                    statistic = float(result.statistic)
                    pvalue = float(result.pvalue)
                test_name = "Friedman"
            elif method in {"ttest", "t-test", "paired_t"}:
                statistic, pvalue = _repeated_measures_anova(matrix)
                test_name = "RM-ANOVA"
            else:
                raise ValueError("stat_test must be 'ttest' or 'wilcoxon'.")

            rows.append({
                "kind": "global",
                "group1": None,
                "group2": None,
                "test": test_name,
                "statistic": statistic,
                "pvalue": pvalue,
                "pvalue_adj": pvalue,
                "n": int(matrix.shape[0]),
            })

    pair_rows = []
    for first, second in _resolve_stat_pairs(groups, stat_pairs):
        paired = wide[[first, second]].dropna()
        statistic, pvalue, n = _paired_test(
            paired[first],
            paired[second],
            method=stat_test,
            alternative=stat_alternative,
        )
        pair_rows.append({
            "kind": "pairwise",
            "group1": first,
            "group2": second,
            "test": "paired t-test" if method in {
                "ttest", "t-test", "paired_t"
            } else "Wilcoxon",
            "statistic": statistic,
            "pvalue": pvalue,
            "n": n,
        })

    if pair_rows:
        adjusted = _adjust_pvalues(
            [row["pvalue"] for row in pair_rows],
            method=stat_correction,
        )
        for row, pvalue_adj in zip(pair_rows, adjusted):
            row["pvalue_adj"] = pvalue_adj
        rows.extend(pair_rows)

    return pd.DataFrame(rows, columns=[
        "kind", "group1", "group2", "test", "statistic",
        "pvalue", "pvalue_adj", "n",
    ])


def _annotate_statistics(
    ax,
    stat_df,
    x_map,
    use_log,
    stat_format,
    stat_show_ns,
    stat_show_n,
    stat_color,
    stat_fontsize,
    stat_linewidth,
):
    if stat_df.empty:
        return

    pair_rows = stat_df[stat_df["kind"] == "pairwise"].copy()
    pair_rows = pair_rows[np.isfinite(pair_rows["pvalue_adj"])]
    if not stat_show_ns:
        pair_rows = pair_rows[pair_rows["pvalue_adj"] < 0.05]

    if not pair_rows.empty:
        pair_rows["distance"] = pair_rows.apply(
            lambda row: abs(x_map[row["group2"]] - x_map[row["group1"]]),
            axis=1,
        )
        pair_rows = pair_rows.sort_values("distance")

    global_rows = stat_df[stat_df["kind"] == "global"]
    lower, upper = ax.get_ylim()

    if use_log:
        if lower <= 0 or upper <= 0:
            raise ValueError("Log-scale significance annotation requires positive limits.")
        low_t, high_t = np.log10(lower), np.log10(upper)
        span = max(high_t - low_t, 0.5)
        start = high_t + 0.06 * span
        step = 0.13 * span
        height = 0.035 * span
        transform = lambda value: 10 ** value
    else:
        span = max(upper - lower, abs(upper) * 0.1, 1e-12)
        start = upper + 0.06 * span
        step = 0.13 * span
        height = 0.035 * span
        transform = lambda value: value

    level = 0
    for _, row in pair_rows.iterrows():
        first = x_map[row["group1"]]
        second = x_map[row["group2"]]
        if first > second:
            first, second = second, first

        y0 = transform(start + level * step)
        y1 = transform(start + level * step + height)
        label = _pvalue_text(row["pvalue_adj"], stat_format)
        if stat_show_n:
            label = f"{label}\n(n={int(row['n'])})"

        ax.plot(
            [first, first, second, second],
            [y0, y1, y1, y0],
            color=stat_color,
            linewidth=stat_linewidth,
            clip_on=False,
            zorder=20,
        )
        ax.text(
            (first + second) / 2,
            y1,
            label,
            color=stat_color,
            fontsize=stat_fontsize,
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=21,
        )
        level += 1

    if not global_rows.empty:
        row = global_rows.iloc[0]
        y = transform(start + level * step + height)
        label = f"{row['test']}: {_pvalue_text(row['pvalue'], 'pvalue')}"
        if stat_show_n:
            label += f" (n={int(row['n'])})"
        ax.text(
            np.mean(list(x_map.values())),
            y,
            label,
            color=stat_color,
            fontsize=stat_fontsize,
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=21,
        )
        level += 1

    if level:
        new_upper = transform(start + level * step + 0.08 * span)
        ax.set_ylim(lower, new_upper)


def plot_paired_trend_log(
    df,
    y_group="3_d",
    x_group="basis",
    pair_col=None,
    basis_order=None,
    time=None,
    figsize=(7, 6),

    # trim 参数
    trim_abs=None,
    trim_ylim=None,
    trim_quantile=None,

    # log 平移参数
    use_shift = True,
    min_shifted=1.0,

    # 颜色与线条参数
    cmap_name="viridis",
    linewidth=0.9,
    line_alpha=0.80,
    scatter_size=14,
    scatter_alpha=0.90,
    set_title= True,
    set_label= True,

    # mean ± SEM 参数
    show_mean_sem=True,
    mean_color="black",
    mean_linewidth=1.0,
    mean_markersize=5,
    capsize=4,

    # 右侧文字参数
    show_right_label=True,
    label_top_n=None,
    text_dx=0.10,
    text_fontsize=8,
    text_angle=0,
    text_alpha=0.95,
    text_color_by_strength=True,

    # 坐标轴参数
    use_log=True,
    show_zero_line=True,
    xlabel="Basis",
    ylabel=None,
    title=None,
    rotation=0,

    # 配对统计参数
    show_stat=True,
    stat_test="wilcoxon",
    stat_pairs=None,
    stat_correction="holm",
    stat_alternative="two-sided",
    stat_global=True,
    stat_format="star",
    stat_show_ns=True,
    stat_show_n=False,
    stat_color="black",
    stat_fontsize=8,
    stat_linewidth=0.8,

    # 图例与保存
    show_colorbar=True,
    legend=True,
    save=None,
    ax=None, fig=None, show=True,
    return_stats=False,
):
    """
    Paired trend plot with optional trimming, integer upward shifting, log y-axis,
    viridis-colored paired lines, and right-side index labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    y_group : str
        Column name for raw paired difference, e.g. "3_d".
    x_group : str
        Column name for method / basis.
    pair_col : str or None
        Column name for paired sample ID. If None, dataframe index is used.
    basis_order : list or None
        Order of x-axis categories.
    trim_abs : float or None
        Keep rows with abs(y_group) <= trim_abs.
    trim_ylim : tuple or None
        Keep rows with trim_ylim[0] <= y_group <= trim_ylim[1].
    trim_quantile : tuple or None
        Keep rows within quantile range, e.g. (0.01, 0.99).
    text_angle : float
        Rotation angle of right-side text labels.
    show_stat : bool
        Whether to calculate and display paired statistical significance.
    stat_test : {'wilcoxon', 'ttest'}
        Pairwise test. With more than two groups, it also selects Friedman
        (wilcoxon) or repeated-measures ANOVA (ttest) as the global test.
    stat_pairs : None, 'all', 'adjacent' or sequence of tuple
        Group pairs to compare. None compares all pairs.
    stat_correction : {'holm', 'bonferroni', 'fdr_bh', None}
        Multiple-testing correction applied to pairwise p-values.
    stat_alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis. Its direction follows (group1 - group2).
    return_stats : bool
        If True, append the statistics DataFrame to the returned tuple.
    """

    # =========================
    # 0. Copy data
    # =========================
    plot_df = df.copy()

    if pair_col is None:
        plot_df["pair_id"] = plot_df.index.astype(str)
    else:
        plot_df["pair_id"] = plot_df[pair_col].astype(str)

    y_shift_group = f"{y_group}_shift"

    plot_df[y_group] = plot_df[y_group].astype(float)

    # =========================
    # 1. basis 顺序
    # =========================
    if basis_order is None:
        if hasattr(plot_df[x_group], "cat"):
            basis_order = plot_df[x_group].cat.categories.tolist()
        else:
            basis_order = plot_df[x_group].drop_duplicates().tolist()

    x_map = {b: i for i, b in enumerate(basis_order)}

    # =========================
    # 2. trim 离散点
    # =========================
    raw_n = plot_df.shape[0]

    if trim_ylim is not None:
        y_low, y_high = trim_ylim
    elif trim_abs is not None:
        y_low, y_high = -float(trim_abs), float(trim_abs)
    elif trim_quantile is not None:
        q_low, q_high = trim_quantile
        y_low, y_high = plot_df[y_group].quantile([q_low, q_high]).to_numpy()
    else:
        y_low, y_high = -np.inf, np.inf

    plot_df = plot_df[
        plot_df[y_group].between(y_low, y_high)
    ].copy()

    trim_n = plot_df.shape[0]

    print(f"Raw rows = {raw_n}")
    print(f"Rows after trim = {trim_n}")
    print(f"Trim range = [{y_low:.4f}, {y_high:.4f}]")

    if trim_n == 0:
        raise ValueError("No data left after trimming. Please loosen trim_abs / trim_ylim / trim_quantile.")

    # =========================
    # 3. 整体向上平移，使 log 纵坐标可用
    # =========================
    y_raw = plot_df[y_group].astype(float)
    y_min = np.nanmin(y_raw)

    if use_shift:
        y_shift = max(0, int(np.ceil(min_shifted - y_min)))
        print(f"Original min after trim = {y_min:.4f}")
        print(f"Integer shift = {y_shift}")
    else:
        y_shift = 0
    plot_df[y_shift_group] = plot_df[y_group] + y_shift

    if use_log and np.any(plot_df[y_shift_group] <= 0):
        raise ValueError(
            "Log scale requires positive plotted values. Enable use_shift "
            "or use use_log=False."
        )

    if show_stat:
        stat_df = _paired_statistics(
            plot_df=plot_df,
            y_group=y_group,
            x_group=x_group,
            basis_order=basis_order,
            stat_test=stat_test,
            stat_pairs=stat_pairs,
            stat_correction=stat_correction,
            stat_alternative=stat_alternative,
            stat_global=stat_global,
        )
    else:
        stat_df = pd.DataFrame(columns=[
            "kind", "group1", "group2", "test", "statistic",
            "pvalue", "pvalue_adj", "n",
        ])


    # =========================
    # 4. 计算每条趋势线的变化强度
    # =========================
    trend_strength = (
        plot_df
        .dropna(subset=[x_group, y_group])
        .groupby("pair_id")[y_group]
        .agg(lambda x: np.nanmax(x) - np.nanmin(x))
    )

    vmin = trend_strength.min()
    vmax = trend_strength.max()

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # =========================
    # 5. 绘图
    # =========================
    sns.set_theme(style="ticks")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif fig is None:
        fig = ax.figure

    # =========================
    # 5.1 个体配对趋势线
    # =========================
    for pid, sub in plot_df.groupby("pair_id"):
        sub = sub.dropna(subset=[x_group, y_shift_group]).copy()
        sub = sub[sub[x_group].isin(basis_order)]
        sub["x_pos"] = sub[x_group].map(x_map)
        sub = sub.sort_values("x_pos")

        if sub.shape[0] >= 2:
            strength = trend_strength.loc[pid]
            color = cmap(norm(strength))

            ax.plot(
                sub["x_pos"],
                sub[y_shift_group],
                color=color,
                linewidth=linewidth,
                alpha=line_alpha,
                zorder=1
            )

            ax.scatter(
                sub["x_pos"],
                sub[y_shift_group],
                color=color,
                s=scatter_size,
                edgecolors='none',
                alpha=scatter_alpha,
                zorder=2
            )

    # =========================
    # 5.2 均值 ± SEM
    # =========================
    if show_mean_sem:
        summary = (
            plot_df
            .groupby(x_group, observed=False)[y_shift_group]
            .agg(["mean", "sem"])
            .reindex(basis_order)
        )

        x = np.arange(len(basis_order))

        mean_y = summary["mean"].to_numpy()
        sem_y = summary["sem"].fillna(0).to_numpy()

        if use_log:

            # log scale 下不能 <=0
            lower_y = np.maximum(
                mean_y - sem_y,
                1e-8
            )

            upper_y = mean_y + sem_y

            yerr = np.vstack([
                mean_y - lower_y,
                upper_y - mean_y
            ])

        else:
            yerr = sem_y
    

        ax.errorbar(
            x,
            mean_y,
            yerr=yerr,
            marker="o",
            markersize=mean_markersize,
            linewidth=mean_linewidth,
            capsize=capsize,
            color=mean_color,
            zorder=5,
            # label="Mean ± SEM"
        )

    # =========================
    # 5.3 原始 0 线
    # =========================
    if show_zero_line:
        ax.axhline(
            y_shift,
            color="black",
            linewidth=1,
            linestyle="--",
            alpha=1.0,
            zorder=3,
            # label="Original 0 line"
        )

    # =========================
    # 5.4 右侧 index / pair_id 标签
    # =========================
    label_df_all = None
    if show_right_label:
        right_basis = basis_order[-1]
        right_x = x_map[right_basis]

        label_df = plot_df[
            plot_df[x_group].eq(right_basis)
        ].dropna(subset=[y_shift_group]).copy()

        label_df = label_df.drop_duplicates(subset=["pair_id"])

        label_df = label_df.sort_values(y_shift_group)
        label_df_all = label_df.copy()
        if label_top_n is not None:
            top_ids = (
                trend_strength
                .sort_values(ascending=False)
                .head(label_top_n)
                .index
            )
            label_df = label_df[label_df["pair_id"].isin(top_ids)]

        # label_df = label_df.sort_values(y_shift_group)

        for _, row in label_df.iterrows():
            pid = row["pair_id"]
            y_value = row[y_shift_group]

            if text_color_by_strength and pid in trend_strength.index:
                text_color = cmap(norm(trend_strength.loc[pid]))
            else:
                text_color = "black"

            ax.text(
                right_x + text_dx,
                y_value,
                str(pid),
                fontsize=text_fontsize,
                rotation=text_angle,
                rotation_mode="anchor",
                va="center",
                ha="left",
                color=text_color,
                alpha=text_alpha,
                clip_on=False,
                zorder=10
            )

        # 给右侧文字留空间
        ax.set_xlim(-0.5, len(basis_order) - 0.5 + 1.0)
    else:
        ax.set_xlim(-0.5, len(basis_order) - 0.5)

    # =========================
    # 6. 坐标轴
    # =========================
    if use_log:
        ax.set_yscale("log")

    x = np.arange(len(basis_order))
    ax.set_xticks(x)
    ax.set_xticklabels(
        basis_order,
        rotation=rotation,
        fontdict={'size': 8},
        ha="right")

    ax.set_xlabel(xlabel)

    if show_stat:
        _annotate_statistics(
            ax=ax,
            stat_df=stat_df,
            x_map=x_map,
            use_log=use_log,
            stat_format=stat_format,
            stat_show_ns=stat_show_ns,
            stat_show_n=stat_show_n,
            stat_color=stat_color,
            stat_fontsize=stat_fontsize,
            stat_linewidth=stat_linewidth,
        )

    if ylabel is None:
        ylabel = f"Shifted score difference: {y_group} + {y_shift}"
    if set_label:
        ax.set_ylabel(ylabel)

    if title is None:
        if time is None:
            title = f"Paired trend across {x_group}"
        else:
            title = f"Paired trend across {x_group} at {time}"
    if set_title:
        ax.set_title(title)

    ax.grid(False)

    # 四周边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # =========================
    # 7. 颜色条
    # =========================
    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        if set_label:
            cbar.set_label("Trend strength: max - min")

    if legend:
        ax.legend(frameon=False)

    plot_df.attrs["statistics"] = stat_df
    fig.tight_layout()

    if save is not None:
        fig.savefig(
            f"{save}.paired_trend_log.pdf",
            transparent=True,
            format="pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            f"{save}.paired_trend_log.png",
            dpi=900,
            bbox_inches="tight",
        )
        plot_df.to_csv(f"{save}.paired_trend_log.data.csv", index=True)
        stat_df.to_csv(
            f"{save}.paired_trend_log.statistics.csv",
            index=False,
        )

    if show:
        plt.show()

    result = (fig, ax, plot_df, trend_strength, label_df_all)
    return (*result, stat_df) if return_stats else result




def plot_paired_trend_log(
    df,
    y_group="3_d",
    x_group="basis",
    pair_col=None,
    basis_order=None,
    time=None,
    figsize=(7, 6),

    # trim 参数
    trim_abs=None,
    trim_ylim=None,
    trim_quantile=None,

    # log 平移参数
    use_shift = True,
    min_shifted=1.0,

    # 颜色与线条参数
    cmap_name="viridis",
    linewidth=0.9,
    line_alpha=0.80,
    scatter_size=14,
    scatter_alpha=0.90,
    set_title= True,
    set_label= True,

    # mean ± SEM 参数
    show_mean_sem=True,
    mean_color="black",
    mean_linewidth=1.0,
    mean_markersize=5,
    capsize=4,

    # 右侧文字参数
    show_right_label=True,
    label_top_n=None,
    text_dx=0.10,
    text_fontsize=8,
    text_angle=0,
    text_alpha=0.95,
    text_color_by_strength=True,

    # 坐标轴参数
    use_log=True,
    show_zero_line=True,
    xlabel="Basis",
    ylabel=None,
    title=None,
    rotation=0,

    # 图例与保存
    show_colorbar=True,
    legend=True,
    save=None,
    ax=None, fig=None, show=True,
):
    """
    Paired trend plot with optional trimming, integer upward shifting, log y-axis,
    viridis-colored paired lines, and right-side index labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    y_group : str
        Column name for raw paired difference, e.g. "3_d".
    x_group : str
        Column name for method / basis.
    pair_col : str or None
        Column name for paired sample ID. If None, dataframe index is used.
    basis_order : list or None
        Order of x-axis categories.
    trim_abs : float or None
        Keep rows with abs(y_group) <= trim_abs.
    trim_ylim : tuple or None
        Keep rows with trim_ylim[0] <= y_group <= trim_ylim[1].
    trim_quantile : tuple or None
        Keep rows within quantile range, e.g. (0.01, 0.99).
    text_angle : float
        Rotation angle of right-side text labels.
    """

    # =========================
    # 0. Copy data
    # =========================
    plot_df = df.copy()

    if pair_col is None:
        plot_df["pair_id"] = plot_df.index.astype(str)
    else:
        plot_df["pair_id"] = plot_df[pair_col].astype(str)

    y_shift_group = f"{y_group}_shift"

    plot_df[y_group] = plot_df[y_group].astype(float)

    # =========================
    # 1. basis 顺序
    # =========================
    if basis_order is None:
        if hasattr(plot_df[x_group], "cat"):
            basis_order = plot_df[x_group].cat.categories.tolist()
        else:
            basis_order = plot_df[x_group].drop_duplicates().tolist()

    x_map = {b: i for i, b in enumerate(basis_order)}

    # =========================
    # 2. trim 离散点
    # =========================
    raw_n = plot_df.shape[0]

    if trim_ylim is not None:
        y_low, y_high = trim_ylim
    elif trim_abs is not None:
        y_low, y_high = -float(trim_abs), float(trim_abs)
    elif trim_quantile is not None:
        q_low, q_high = trim_quantile
        y_low, y_high = plot_df[y_group].quantile([q_low, q_high]).to_numpy()
    else:
        y_low, y_high = -np.inf, np.inf

    plot_df = plot_df[
        plot_df[y_group].between(y_low, y_high)
    ].copy()

    trim_n = plot_df.shape[0]

    print(f"Raw rows = {raw_n}")
    print(f"Rows after trim = {trim_n}")
    print(f"Trim range = [{y_low:.4f}, {y_high:.4f}]")

    if trim_n == 0:
        raise ValueError("No data left after trimming. Please loosen trim_abs / trim_ylim / trim_quantile.")

    # =========================
    # 3. 整体向上平移，使 log 纵坐标可用
    # =========================
    y_raw = plot_df[y_group].astype(float)
    y_min = np.nanmin(y_raw)

    if use_shift:
        y_shift = max(0, int(np.ceil(min_shifted - y_min)))
        print(f"Original min after trim = {y_min:.4f}")
        print(f"Integer shift = {y_shift}")
    else:
        y_shift = 0
    plot_df[y_shift_group] = plot_df[y_group] + y_shift


    # =========================
    # 4. 计算每条趋势线的变化强度
    # =========================
    trend_strength = (
        plot_df
        .dropna(subset=[x_group, y_group])
        .groupby("pair_id")[y_group]
        .agg(lambda x: np.nanmax(x) - np.nanmin(x))
    )

    vmin = trend_strength.min()
    vmax = trend_strength.max()

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # =========================
    # 5. 绘图
    # =========================
    sns.set_theme(style="ticks")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # =========================
    # 5.1 个体配对趋势线
    # =========================
    for pid, sub in plot_df.groupby("pair_id"):
        sub = sub.dropna(subset=[x_group, y_shift_group]).copy()
        sub = sub[sub[x_group].isin(basis_order)]
        sub["x_pos"] = sub[x_group].map(x_map)
        sub = sub.sort_values("x_pos")

        if sub.shape[0] >= 2:
            strength = trend_strength.loc[pid]
            color = cmap(norm(strength))

            ax.plot(
                sub["x_pos"],
                sub[y_shift_group],
                color=color,
                linewidth=linewidth,
                alpha=line_alpha,
                zorder=1
            )

            ax.scatter(
                sub["x_pos"],
                sub[y_shift_group],
                color=color,
                s=scatter_size,
                edgecolors='none',
                alpha=scatter_alpha,
                zorder=2
            )

    # =========================
    # 5.2 均值 ± SEM
    # =========================
    if show_mean_sem:
        summary = (
            plot_df
            .groupby(x_group, observed=False)[y_shift_group]
            .agg(["mean", "sem"])
            .reindex(basis_order)
        )

        x = np.arange(len(basis_order))

        mean_y = summary["mean"].to_numpy()
        sem_y = summary["sem"].fillna(0).to_numpy()

        if use_log:

            # log scale 下不能 <=0
            lower_y = np.maximum(
                mean_y - sem_y,
                1e-8
            )

            upper_y = mean_y + sem_y

            yerr = np.vstack([
                mean_y - lower_y,
                upper_y - mean_y
            ])

        else:
            yerr = sem_y
    

        ax.errorbar(
            x,
            mean_y,
            yerr=yerr,
            marker="o",
            markersize=mean_markersize,
            linewidth=mean_linewidth,
            capsize=capsize,
            color=mean_color,
            zorder=5,
            # label="Mean ± SEM"
        )

    # =========================
    # 5.3 原始 0 线
    # =========================
    if show_zero_line:
        ax.axhline(
            y_shift,
            color="black",
            linewidth=1,
            linestyle="--",
            alpha=1.0,
            zorder=3,
            # label="Original 0 line"
        )

    # =========================
    # 5.4 右侧 index / pair_id 标签
    # =========================
    label_df_all = None
    if show_right_label:
        right_basis = basis_order[-1]
        right_x = x_map[right_basis]

        label_df = plot_df[
            plot_df[x_group].eq(right_basis)
        ].dropna(subset=[y_shift_group]).copy()

        label_df = label_df.drop_duplicates(subset=["pair_id"])

        label_df = label_df.sort_values(y_shift_group)
        label_df_all = label_df.copy()
        if label_top_n is not None:
            top_ids = (
                trend_strength
                .sort_values(ascending=False)
                .head(label_top_n)
                .index
            )
            label_df = label_df[label_df["pair_id"].isin(top_ids)]

        # label_df = label_df.sort_values(y_shift_group)

        for _, row in label_df.iterrows():
            pid = row["pair_id"]
            y_value = row[y_shift_group]

            if text_color_by_strength and pid in trend_strength.index:
                text_color = cmap(norm(trend_strength.loc[pid]))
            else:
                text_color = "black"

            ax.text(
                right_x + text_dx,
                y_value,
                str(pid),
                fontsize=text_fontsize,
                rotation=text_angle,
                rotation_mode="anchor",
                va="center",
                ha="left",
                color=text_color,
                alpha=text_alpha,
                clip_on=False,
                zorder=10
            )

        # 给右侧文字留空间
        ax.set_xlim(-0.5, len(basis_order) - 0.5 + 1.0)
    else:
        ax.set_xlim(-0.5, len(basis_order) - 0.5)

    # =========================
    # 6. 坐标轴
    # =========================
    if use_log:
        ax.set_yscale("log")

    x = np.arange(len(basis_order))
    ax.set_xticks(x)
    ax.set_xticklabels(
        basis_order,
        rotation=rotation,
        fontdict={'size': 8},
        ha="right")

    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = f"Shifted score difference: {y_group} + {y_shift}"
    if set_label:
        ax.set_ylabel(ylabel)

    if title is None:
        if time is None:
            title = f"Paired trend across {x_group}"
        else:
            title = f"Paired trend across {x_group} at {time}"
    if set_title:
        ax.set_title(title)

    ax.grid(False)

    # 四周边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # =========================
    # 7. 颜色条
    # =========================
    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        if set_label:
            cbar.set_label("Trend strength: max - min")

    if legend:
        ax.legend(frameon=False)
    
    if show:
        plt.tight_layout()
        if save is not None:
            fig.savefig(f"{save}.paired_trend_log.pdf", transparent=True, format="pdf")
            fig.savefig(f"{save}.paired_trend_log.png", dpi=900, bbox_inches="tight")
            plot_df.to_csv(f"{save}.paired_trend_log.data.csv", index=True)

        plt.show()
        return fig, ax, plot_df, trend_strength, label_df_all


# fig, ax, plot_df_trimmed, trend_strength = plot_paired_trend_log(
#     iidf,
#     y_group="3_d",
#     x_group="basis",
#     time=time,
    # trim_abs=20,          # 保留 -20 到 20
    # trim_ylim=(-10, 30),
#     trim_quantile=(0.02, 0.98),
    # trim_ylim=(-10, 30),
    # trim_quantile=(0.01, 0.99),
#     label_top_n=15,
#     text_angle=25,
#     linewidth=0.8,
#     line_alpha=0.75,
#     scatter_size=16,
#     figsize=(7.5, 6),
# )


def slice_knn(Xs, knn=8, ks=1, radius=None,kd_method='annoy', device=None, CI=0.99, disable=True):
    offsets = np.cumsum([0] + [len(X) for X in Xs])
    N_total = int(offsets[-1])

    all_src, all_dst, all_d = [], [], []
    L = len(Xs)
    pbar = tqdm(range(L-1), total=L-1, desc=f'knn {L}', disable=disable)

    for s in pbar:
        for j in range(s+1, min(s+ks+1, L)):
            Xa, Xb = Xs[s], Xs[j]
            Na, Nb = len(Xa), len(Xb)

            src1, dst1, dist1 = coord_edges(Xa, Xb, knn = knn, radius=radius, device=device, method=kd_method)
            src2, dst2, dist2 = coord_edges(Xb, Xa, knn = knn, radius=radius, device=device, method=kd_method)

            all_src.append(offsets[s] + src1)
            all_src.append(offsets[j] + src2)
            all_dst.append(offsets[j] + dst1)
            all_dst.append(offsets[s] + dst2)
            all_d.append(dist1)
            all_d.append(dist2)

    src = np.concatenate(all_src) 
    dst = np.concatenate(all_dst)
    dist = np.concatenate(all_d)

    K  = int(len(dist)*CI)
    kidx = np.argpartition(dist, K,)[:K]
    src = src[kidx]
    dst = dst[kidx]
    
    w   = np.ones(len(src))
    W = sparse.csr_matrix((w, (dst, src)), shape=(N_total, N_total))
    return W

def GMD(Xs, Fs, labels=None, W=None, alpha=0.5, knn=8, 
        normalx= True, normalf=True,
        temp=[1,1], agg='mean', use_xy=True,
        radius=None,kd_method='annoy',  CI=0.97):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score


    if W is None:
        W = slice_knn(Xs, knn=knn, radius=radius, kd_method=kd_method, CI=CI)
    src, dst = W.nonzero()

    sx = np.concatenate(Xs, axis=0)
    sf = np.concatenate(Fs, axis=0)

    if use_xy:
        sx = sx[:, :2]
    if normalx:
        sx = scaler(sx)


    if normalf:
        sf = normalize(sf)
    
    spdist = np.linalg.norm(sx[src] - sx[dst], axis=1) ** 2
    ftdist = np.linalg.norm(sf[src] - sf[dst], axis=1) ** 2
    # ftdist1 = (sf[src] * sf[dst]).sum(1)

    sigmap = np.median(spdist[spdist>0]) * temp[0]
    sigmaf = np.median(ftdist[ftdist>0]) * temp[1]

    sigmap = 1 * temp[0]
    sigmaf = 1 * temp[1]


    score1 = spdist/sigmap + ftdist/sigmaf
    score2 = np.exp(- spdist/sigmap - ftdist/sigmaf ) ** 0.5
    score3 = (np.exp(- spdist/sigmap) + np.exp(- ftdist/sigmaf))/2
    score4 =  (spdist * ftdist) **0.5
    score5 =  ((spdist**alpha )* (ftdist**(1-alpha))) **0.5

    df = pd.DataFrame(np.c_[score1, score2, score3, score4, score5],
                         columns=['score1', 'score2', 'score3', 'score4', 'score5'])

    if agg == 'mean':
        mdf = df.mean(0)
        mdf['score4'] = df['score4'].sum(0) /(sx.shape[0]**0.5)
        mdf['score5'] = df['score5'].sum(0) /(sx.shape[0]**0.5)
    elif agg == 'sum':
        mdf = df.sum(0)

    if not (labels is None):
        types = labels.columns.tolist()
        methods = [['ARI','AMI', 'NMI'], [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score]]
        methods = [['ARI'], [adjusted_rand_score]]
        scores = []
        for itl in types:
            for inm, imd in zip(*methods):
                iscore = imd(labels[itl].values[src], labels[itl].values[dst])
                scores.append([f'{inm}({itl})', iscore])
        scores = pd.DataFrame(scores, columns=['types', 'score'])
        mdf = pd.concat([mdf, scores.set_index('types').iloc[:,0]])
        for itl in types:
            iscore = labels[itl].values[src] == labels[itl].values[dst]
            isoore = iscore.sum() / (~iscore).sum()
            mdf[f'rs({itl})'] = isoore
    return mdf
    
def laplacian_smoothness_scores(Xs, Fs=None, W=None, knn=8, radius=None,kd_method='annoy',
                                bf = 10, 
                                normalx=True, normalf=True, 
                                sigx2=1,sigf2=1, lambdax=1):
    if W is None:
        W = slice_knn(Xs, knn=knn, radius=radius, kd_method=kd_method)
    src, dst = W.nonzero()

    Xa = np.vstack(Xs)
    if normalx:
        Xa = scaler(Xa) * bf
    dx2 = np.linalg.norm(Xa[src] - Xa[dst], axis=1) **2
    V1 = np.exp(-dx2 / sigx2)
    V2 = np.exp(-dx2)
  
    if Fs is not None:
        Fa = np.vstack(Fs)
        if normalf:
            Fa = normalize(Fa)
        df2 = np.linalg.norm(Fa[src] - Fa[dst], axis=1) ** 2
        V1 = np.exp(-dx2 / (sigx2*lambdax)  - df2 / sigf2)
        V2 = np.exp(- (dx2*df2) **0.5 )
        V3 = np.exp(-df2)
        V4 = ((Fa[src]*Fa[dst]).sum(1) + 1)/2
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(5, 3))
    axs[0,0].hist(dx2, bins=100)
    axs[0,0].set_title('dx2')
    if Fs is not None:
        axs[0,1].hist(df2, bins=100)
        axs[0,1].set_title('df2')
    axs[1,0].hist(V1, bins=100)
    axs[1,0].set_title('V1')
    axs[1,1].hist(V2, bins=100)
    axs[1,1].set_title('V2')
    plt.show()

    W1 = sparse.csr_matrix((V1, (src, dst)), shape=W.shape)
    L1 = sparse.diags(np.array(W1.sum(1)).ravel()) - W1

    W2  = sparse.csr_matrix((V2, (src, dst)), shape=W.shape)
    L2 = sparse.diags(np.array(W2.sum(1)).ravel()) - W2

    W3 = sparse.csr_matrix((V3, (src, dst)), shape=W.shape)
    L3 = sparse.diags(np.array(W3.sum(1)).ravel()) - W3

    W4 = sparse.csr_matrix((V4, (src, dst)), shape=W.shape)
    L4 = sparse.diags(np.array(W4.sum(1)).ravel()) - W4

    Ex1 = (V1 * dx2).sum()/Xa.shape[0]
    Ex2 = (V2 * dx2).sum()/Xa.shape[0]
    Ex3 = np.trace(Xa.T @ (L1 @ Xa))/Xa.shape[0]
    Ex4 = np.trace(Xa.T @ (L2 @ Xa))/Xa.shape[0]
    Ex5 = np.trace(Xa.T @ (L3 @ Xa))/Xa.shape[0]
    Ex6 = np.trace(Xa.T @ (L4 @ Xa))/Xa.shape[0]
    Ex7 = Ex1/V1.sum()
    Ex8 = Ex2/V2.sum()

    if Fs is not None:
        Ef1 = (V1 * df2).sum()/Xa.shape[0]
        Ef2 = (V2 * df2).sum()/Xa.shape[0]
        Ef3 = np.trace(Fa.T @ (L1 @ Fa))/Xa.shape[0]
        Ef4 = np.trace(Fa.T @ (L2 @ Fa))/Xa.shape[0]
        Ef5 = np.trace(Fa.T @ (L3 @ Fa))/Xa.shape[0]    
        Ef6 = np.trace(Fa.T @ (L4 @ Fa))/Xa.shape[0]
        Ef7 = Ef1/V1.sum()
        Ef8 = Ef2/V2.sum()

    R = np.array([[Ex1, Ex2, Ex3, Ex4, Ex5, Ex6, Ex7, Ex8], [Ef1, Ef2, Ef3, Ef4, Ef5, Ef6, Ef7, Ef8]])
    print(R)
    return R

def gene_smoothness_scores(Xs, G, Fs= None, W=None, knn=8, ks=1, radius=None,kd_method='annoy',
                                 CI=0.98,
                                sigx2=1,sigf2=1, lambdax=1):
    if W is None:
        W = slice_knn(Xs, knn=knn, ks=ks,radius=radius, kd_method=kd_method, CI=CI)
    src, dst = W.nonzero()

    Xa = np.vstack(Xs)
    dx2 = np.linalg.norm(Xa[src] - Xa[dst], axis=1) **2
    V1 = np.exp(-dx2 / sigx2)
    V2 = np.exp(-dx2)
  
    if Fs is not None:
        Fa = np.vstack(Fs)
        df2 = np.linalg.norm(Fa[src] - Fa[dst], axis=1) ** 2
        V1 = np.exp(-dx2 / (sigx2*lambdax)  - df2 / sigf2)
        V2 = np.exp(- (dx2*df2) **0.5 )

    W1 = sparse.csr_matrix((V1, (src, dst)), shape=W.shape)
    L1 = sparse.diags(np.array(W1.sum(1)).ravel()) - W1

    W2  = sparse.csr_matrix((V2, (src, dst)), shape=W.shape)
    L2 = sparse.diags(np.array(W2.sum(1)).ravel()) - W2

    Ex1 = (V1 * G[dst]).sum()/G.shape[0]
    Ex2 = (V2 * G[dst]).sum()/G.shape[0]
    Ex3 = np.trace(G[:,None].T @ (L1 @ G[:,None]))/G.shape[0]
    Ex4 = np.trace(G[:,None].T @ (L2 @ G[:,None]))/G.shape[0]
    Ex5 = np.trace(G[:,None].T @ (W @ G[:,None]))/G.shape[0]

    R = np.array([Ex1, Ex2, Ex3, Ex4, Ex5])
    return R

gwCOLORS = ['#b240ce', '#b6b51f', '#0780cf', '#765005', '#fa6d1d', 
          '#2f3ea8', '#da1f18', '#701866', '#f47a75', '#009db2', 
          '#024b51', '#0780cf', '#765005', '#6beffc', '#3b45dd', 
          '#ad94ec', '#00749d', '#6ed0a7', '#0e2c82', '#706c01', 
          '#9be4ff', '#d70000']

def gwplt_hist(dfs, figsize=(6, 10), y_lim = (0,100), ys_min = None, ys_max = None, 
             scores= [2,3,4,5,6],
             palette= gwCOLORS,
             linewidth=1, save=None):
    import seaborn as sns

    fig, axs = plt.subplots( len(scores), 1, figsize=figsize,
                            constrained_layout=False, sharex=False, sharey=False)
    for i, iscore in enumerate(scores):
        if len(scores) == 1:
            iax = axs
        else:
            iax = axs[i]
        idf = dfs.loc[:, iscore].copy()

        sns.barplot( dfs, y=dfs.index, x =iscore,  linewidth=linewidth, 
                    palette =palette[:dfs.shape[0]], orient= 'h',
                    ax=iax)
        iax.grid(False)
        iax.set_title(iscore)
        # legend = iax.get_legend()
        # legend.set_bbox_to_anchor((1.05, 0.5))
        # legend.set_loc('center left')  

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.gw.score.hist.pdf',transparent=True, format='pdf')
        dfs.to_csv(f'{save}.gw.score.hist.csv')
    plt.show()

def gwplt_hist1(dfs, figsize=(6, 10), y_lim = (0,100), ys_min = None, ys_max = None, 
             scores= [2,3,4,5,6],
             palette= gwCOLORS[:4],
             linewidth=1, save=None, **kargs):
    import seaborn as sns

    fig, axs = plt.subplots( len(scores), 1, figsize=figsize,
                            constrained_layout=False, sharex=False, sharey=False)
    for i, iscore in enumerate(scores):
        sns.barplot( dfs, y='Time', x =iscore, hue='Group', linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=axs[i], legend='auto', **kargs)
        axs[i].grid(False)
        axs[i].set_title(iscore)
        legend = axs[i].get_legend()
        legend.set_bbox_to_anchor((1.05, 0.5))
        legend.set_loc('center left')  

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.gw.score.hist.pdf',transparent=True, format='pdf')
        dfs.to_csv(f'{save}.gw.score_hu.hist.csv')
    plt.show()


def gwplt_hist2(dfs, groupyb='Time', y='Group', scores= [3,4,5],
             figsize=(6, 10), gorder=None,
             palette= gwCOLORS[:4], 
             sharex=True, sharey=False,
             linewidth=1, save=None, **kargs):
    import seaborn as sns
    if gorder is None:
        groups = np.unique(dfs[groupyb])
    else:
        groups = gorder
    for iscore in scores:
        fig, axs = plt.subplots( len(groups), 1, figsize=figsize,
                                constrained_layout=False, sharex=sharex, sharey=sharey)
        for i, igroup in enumerate(groups):
            idf = dfs[ (dfs[groupyb]==igroup) ].copy()

            sns.barplot( idf, y=y, x =iscore,  linewidth=linewidth, 
                        palette =palette, orient= 'h',
                        ax=axs[i], legend='auto', **kargs)
            axs[i].grid(False)
            # axs[i].set_title(iscore)
            axs[i].set_ylabel(igroup)
            # legend = axs[i].get_legend()
            # legend.set_bbox_to_anchor((1.05, 0.5))
            # legend.set_loc('center left')  

        fig.tight_layout()
        if save:
            fig.savefig(f'{save}.gw.score_{iscore}.hist.pdf',transparent=True, format='pdf')
            # dfs.to_csv(f'{save}.gw.score_{iscore}.hist.csv')
        plt.show()

def gwplt_hist_cap(dfs, figsize=(4.5, 2), y_lim = (0,100),
             scores= ['score4'],
             COLOR = ['#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
             cap_ratio =None,
             cap =None,
             cap_space = 0.05,  d = 5, wspace=0.08,
             break_ratio =0.18,
             break_line=8,
             linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    
    for i, iscore in enumerate(scores):
        idf = dfs.loc[:,iscore].copy()
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, nrows=1, sharey=True, 
                                              figsize=figsize,
                                              width_ratios=[1-break_ratio, break_ratio],
                                              gridspec_kw={'wspace':wspace})
        sns.barplot( idf, linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=ax_left, legend=False)
        sns.barplot( idf,  linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=ax_right, legend=False)

        ax_left.grid(False)
        ax_right.grid(False)
        # ax_left.set_xlabel('')
        # ax_right.set_xlabel('')
        ax_left.set_title(iscore)

        L = idf.max()
        if cap is None:
            cap_ratio = cap_ratio or 0.75
            cap = L * cap_ratio
        else:
            capl = min(cap, L*0.98)
        capp = cap_space * L

        ax_right.set_xlim(left=capl+capp, right=L+capp)   # those limits are fake
        ax_left.set_xlim(0,capl)

        ax_left.spines.right.set_visible(False)
        ax_right.spines.left.set_visible(False)

        ax_left.yaxis.tick_left()
        ax_left.tick_params(axis='y', which='both', right=False, labelright=False)
        ax_right.yaxis.tick_right()
        ax_right.tick_params(axis='y', which='both', right=False, labelright=False)
        # sns.despine(ax=ax_left)
        # sns.despine(ax=ax_right, right=True)
        # # ax_bottom.legend_.remove()

        d  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=break_line,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_left.plot([1, 1], [0, 0], transform=ax_left.transAxes, **kwargs)
        ax_right.plot([0, 0], [0, 0], transform=ax_right.transAxes, **kwargs)
        ax_left.plot([1, 1], [1, 1], transform=ax_left.transAxes, **kwargs)
        ax_right.plot([0, 0], [1, 1], transform=ax_right.transAxes, **kwargs)

        # fig.tight_layout()
        if save:
            fig.savefig(f'{save}.{iscore}.hist.cap.pdf',transparent=True, format='pdf')
            dfs.to_csv(f'{save}.{iscore}.hist.cap.csv')
        plt.show()

def gwplt_hist_cap1(dfs, x='score4', y ='knn', hue='group', 
                   figsize=(4.5, 2), y_lim = (0,100),
             COLOR = ['#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
             cap_ratio =None,
             cap =None,
             cap_space = 0.05,  d = 5, wspace=0.08,
             break_ratio =0.18,
             break_line=8,
             linewidth=1, save=None,  **kargs):
    import seaborn as sns
    palette = COLOR

    fig, (ax_left, ax_right) = plt.subplots(ncols=2, nrows=1, sharey=True, 
                                          figsize=figsize,
                                          width_ratios=[1-break_ratio, break_ratio],
                                          gridspec_kw={'wspace':wspace})
    sns.barplot( dfs, 
                x=x, y =y, hue=hue,
                 linewidth=linewidth, 
                palette =palette, orient= 'h',
                ax=ax_left, legend=False, **kargs)
    sns.barplot( dfs,  x=x, y =y, hue=hue, 
                linewidth=linewidth, 
                palette =palette, orient= 'h',
                ax=ax_right, legend=False,  **kargs)

    ax_left.grid(False)
    ax_right.grid(False)
    # ax_left.set_xlabel('')
    # ax_right.set_xlabel('')
    ax_left.set_title(x)

    L = dfs[x].max()
    if cap is None:
        cap_ratio = cap_ratio or 0.75
        cap = L * cap_ratio
    else:
        capl = min(cap, L*0.98)
    capp = cap_space * L

    ax_right.set_xlim(left=capl+capp, right=L+capp)   # those limits are fake
    ax_left.set_xlim(0,capl)

    ax_left.spines.right.set_visible(False)
    ax_right.spines.left.set_visible(False)

    ax_left.yaxis.tick_left()
    ax_left.tick_params(axis='y', which='both', right=False, labelright=False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(axis='y', which='both', right=False, labelright=False)
    # sns.despine(ax=ax_left)
    # sns.despine(ax=ax_right, right=True)
    # # ax_bottom.legend_.remove()

    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=break_line,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_left.plot([1, 1], [0, 0], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [0, 0], transform=ax_right.transAxes, **kwargs)
    ax_left.plot([1, 1], [1, 1], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [1, 1], transform=ax_right.transAxes, **kwargs)

    # fig.tight_layout()
    if save:
        fig.savefig(save,transparent=True, format='pdf')
    plt.show()


def pca_normals_curv(X, knn=8, radius=None,kd_method='annoy'):
    distances, indices = coord_edges(X, knn = knn, radius=radius, method=kd_method, return_array=True)
    nb = indices[:,1:]
    N = len(X)
    Nrm = np.zeros_like(X)
    Curv = np.zeros(N)
    for i in range(N):
        P = X[nb[i]] - X[nb[i]].mean(0, keepdims=True)
        C = (P.T @ P) / max(1, len(P)-1)
        w, V = np.linalg.eigh(C)   # ascending
        Nrm[i] = V[:,0]
        Curv[i] = w[0] / (w.sum() + 1e-12)
    # canonical orientation (optional)
    Nrm /= (np.linalg.norm(Nrm, axis=1, keepdims=True)+1e-12)
    return Nrm, Curv

def planarity_score(curv, tau=0.1):
    print('curv:', np.sum(curv), )
    med = float(np.median(curv))
    return float(np.exp(-med / max(tau,1e-9)))

def normal_tv_score(Nrm, A):
    A = A.tocoo()
    cosv = np.abs(np.sum(Nrm[A.row] * Nrm[A.col], axis=1))
    print('cosv:', np.sum(cosv))
    entv = float(np.mean(1.0 - cosv))
    return float(1.0 - entv)  # in [0,1]

def build_cross_slice_edges(Xs, Fs=None, knn=8, radius=None, lambda_x=1.0,
                            kd_method='sknn',
                             sigx2=1,sigf2=1, random_state=0):
    """
    返回：
      - src, dst: 全局点索引的一维数组
      - w: 边权
      - dx2_list, df2_list: 边的平方距离（用于尺度估计）
      - idx_map: 每张切片在全局的起止索引
    """
    offsets = np.cumsum([0] + [len(X) for X in Xs])
    N_total = int(offsets[-1])

    all_src, all_dst, all_w = [], [], []
    dx2_all, df2_all = [], []

    for s in range(len(Xs)-1):
        Xa, Xb = Xs[s], Xs[s+1]
        Fa, Fb = Fs[s], Fs[s+1]
        Na, Nb = len(Xa), len(Xb)

        src1, dst1, dist1 = coord_edges(Xa, Xb, knn = knn, radius=radius, method=kd_method)
        dst2, src2, dist2 = coord_edges(Xb, Xa, knn = knn, radius=radius, method=kd_method)

        src = np.concatenate([src1, src2])
        dst = np.concatenate([dst1, dst2])
        dx2 = np.linalg.norm(Xa[src] - Xb[dst], axis=1) **2
        df2 = np.linalg.norm(Fa[src] - Fb[dst], axis=1) ** 2

        print(1111,  np.mean(dx2), np.median(dx2),  np.mean(df2), np.median(df2), )
        a = -dx2 / (sigx2*lambda_x) - df2 / sigf2
        b = - (dx2 * df2)**0.5
        print(2222, np.mean(a), np.median(a), np.min(a), np.max(a), np.mean(b), np.median(b), np.min(b), np.max(b), )
        w = np.exp(-dx2 / (sigx2*lambda_x) - df2 / sigf2)
        w = np.exp(- (dx2 * df2)**0.5 )

        src += offsets[s]
        dst += offsets[s+1]
        all_src.append(src); all_dst.append(dst); all_w.append(w.ravel())
        dx2_all.append(dx2); df2_all.append(df2)

    src = np.concatenate(all_src) 
    dst = np.concatenate(all_dst)
    w   = np.concatenate(all_w)
    W = sparse.csr_matrix((w, (src, dst)), shape=(N_total, N_total))

    dx2_list = np.concatenate(dx2_all)
    df2_list = np.concatenate(df2_all)
    print(w.max(), w.min(), dx2_list.shape, df2_list.shape)

    return W, dx2_list, df2_list

def laplacian_dirichlet_score(X, F, W= None, sigma2=1.0, knn=8, radius=None,kd_method='annoy',):
    if W is None:
        src, dst, dist2 = coord_edges(X, knn = knn, radius=radius, method=kd_method)
        w  = np.ones(len(src))
        W = sparse.csr_matrix((w, (dst, src)), shape=(X.shape[0], X.shape[0]))
    src, dst = W.nonzero()
    V = np.exp( -((X[src] - X[dst])**2).sum(axis=1)/sigma2)
    V1 = np.exp( -((F[src] - F[dst])**2).sum(axis=1) )
    V2 = ((F[src] * F[dst]).sum(1) + 1)/2


    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    Ex = np.trace(F.T @ (L @ F))

    W1 = sparse.csr_matrix((V, (src, dst)), shape=W.shape)
    d1 = np.array(W1.sum(1)).ravel()
    L1 = sparse.diags(d1) - W1
    Ex1 = np.trace(X.T @ (L1 @ X))
    print(Ex)
    return Ex

    W1 = sparse.csr_matrix((V1, (src, dst)), shape=W.shape)
    d1 = np.array(W1.sum(1)).ravel()
    L1 = sparse.diags(d1) - W1
    Ex1 = np.trace(X.T @ (L1 @ X))

    W2 = sparse.csr_matrix((V2, (src, dst)), shape=W.shape)
    d2 = np.array(W2.sum(1)).ravel()
    L2 = sparse.diags(d2) - W2
    Ex2 = np.trace(X.T @ (L2 @ X))

    print(Ex, Ex1, Ex2)
    return Ex, Ex1, Ex2

def laplacian_dirichlet_score(X, F, W= None, sigma2=1.0, knn=8, radius=None,kd_method='annoy',):
    if W is None:
        src, dst, dist2 = coord_edges(X, knn = knn, radius=radius, method=kd_method)
        w  = np.ones(len(src))
        W = sparse.csr_matrix((w, (dst, src)), shape=(X.shape[0], X.shape[0]))
    src, dst = W.nonzero()
    V = np.exp( -((X[src] - X[dst])**2).sum(axis=1)/sigma2)
    W = sparse.csr_matrix((V, (src, dst)), shape=W.shape)
    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    Ex = np.trace(F.T @ (L @ F))
    print(Ex)
    return Ex


def LSS(Xs, Fs, W, dx2_list, df2_list):

    src, dst = W.nonzero()
    w = np.array(W[src, dst]).ravel()

    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    # 坐标拼接
    X_all = np.vstack(Xs)
    # 位置能量（直接按边求，无需显式 L）
    dx2_edge = ((X_all[src] - X_all[dst])**2).sum(axis=1)  # (E,)

    Ex = (w * dx2_edge).sum() 
    Ex1 = np.trace(X_all.T @ (L @ X_all))
    print(Ex,Ex/w.sum(), Ex1)

    # Ex_bar = (w * dx2_edge).sum() / (w.sum() + 1e-12)
    # S_lap_x = float(np.exp(-Ex_bar / sigx2))


    # F_all = np.vstack(Fs)
    # df2_edge = ((F_all[src] - F_all[dst])**2).sum(axis=1)
    # sigf2 = robust_median_sq(df2_list)
    # Ef_bar = (w * df2_edge).sum() / (w.sum() + 1e-12)
    # S_lap_f = float(np.exp(-Ef_bar / (sigf2 + 1e-12)))
    # return S_lap_x, S_lap_f

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= True,
                return_array = False,
                device=None,
                n_jobs = -1):

    import sys
    sys.path.append('/gpfs/home/user19/JupyterCode/')
    from CellCloudX_v130 import cellcloudx as cc

    if coordy is None:
        coordy = coordx
    
    if method=='keops':
        cknn = cc.ag.neighbors_ensemble.ko_Neighbors( coordx, Y=coordy, K=knn, device=device)
        indices, distances = cknn.knn(return_distances=True)
        indices = indices.cpu().numpy()
        distances = (distances ** 0.5).cpu().numpy()

    else:
        cknn = cc.ag.neighbors_ensemble.Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
        cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
        distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    if return_array:
        return distances, indices

    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)

    if (coordy is None) and (not keep_loops):
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        dist = dist[mask]
    # print(f'mean edges: {dist.shape[0]/coordy.shape[0]}')
    return [src, dst, dist]


# ---------- graph utils ----------
def knn_graph(X, k=16):
    tree = cKDTree(X)
    d, nb = tree.query(X, k=k+1)  # include self
    nb = nb[:,1:]
    src = np.repeat(np.arange(len(X)), k)
    dst = nb.reshape(-1)
    # symmetrize
    A = sparse.coo_array((np.ones_like(src), (src, dst)), shape=(len(X), len(X)))
    A = A + A.T
    A = A.tocsr()
    A.data[:] = 1.0
    return A

def gaussian_weights(X, A, F= None,  s2=None):
    A = A.tocoo()
    diff = X[A.row] - X[A.col]
    d2 = np.sum(diff*diff, axis=1)

    if s2 is None:
        s2 = np.median(d2) + 1e-12
    print(1111111, np.mean(d2), np.median(d2))
    w = np.exp(-d2 / s2)

    if F is not None:
        F2 = np.linalg.norm(F[A.row] - F[A.col], axis=1)**2
        # w = np.exp(-d2 / s2 + F2)
        print(11111, (d2*F2).sum() )
        w = np.exp(-d2 / s2 *F2) **0.5
    W = sparse.coo_array((w, (A.row, A.col)), shape=A.shape).tocsr()
    return W, s2



def laplacian_energy_score(X, W,  c=1.0):
    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    E = float(np.trace(X.T @ (L @ X))) 
    # print(E/len(X),  float(np.exp(- E / len(X))) )
    # S = float(np.exp(- E / (c*len(X)*s2 + 1e-12)))
    N,D = X.shape
    S = float(np.exp(- E / (N*D/2)) ) 
    return S

def feature_smoothness_score(F, W, tau=0.1):
    if F is None: return np.nan
    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    E = float(np.trace(F.T @ (L @ F))) 
    N,D = F.shape
    # varF = float(np.mean((F - F.mean(0))**2)) + 1e-12
    # print(E, E/F.shape[0], varF)
    return float(np.exp(- E / (N*D/2)))



def mean_curvature_proxy_score(X, L, tau=0.1):
    # node-wise (Lx)_i magnitude^2 median
    Lx = np.stack([(L @ X[:,j]) for j in range(3)], axis=1)
    R2 = np.sum(Lx*Lx, axis=1)
    med = float(np.median(R2))
    return float(np.exp(- med / max(tau,1e-9)))


def spectral_highfreq_ratio_score(X, L, t=0.1, m=5, tau=0.2):
    I = sparse.eye(L.shape[0], format='csr')
    H = (I - t*L).tocsr()   # one-step heat kernel approx
    Xsm = X.copy()
    for _ in range(m):
        Xsm = H @ Xsm
    num = float(np.sum((X - Xsm)**2))
    den = float(np.sum((X - X.mean(0))**2))
    R = num / den
    print(R, num , den)
    return float(np.exp(- R / max(tau,1e-9)))

def box_counting_dimension_score(X, bins_list=(16,24,32,48,64), tau=0.5):
    # voxel occupancy across multiple resolutions
    mins = X.min(0) - 1e-9
    maxs = X.max(0) + 1e-9
    rng = (maxs - mins)
    eps = []
    Ns  = []
    for B in bins_list:
        # map to [0,B)
        g = ((X - mins) / rng * B).clip(0, B-1-1e-9)
        keys = np.floor(g).astype(int)
        # unique voxels
        if len(keys)==0: continue
        uniq = np.unique(keys, axis=0)
        Ns.append(len(uniq))
        eps.append(1.0/B)
    if len(Ns) < 3: return np.nan
    x = np.log(1/np.array(eps)); y = np.log(np.array(Ns))
    D, b = np.polyfit(x, y, 1)   # slope is dimension
    return float(np.exp(- abs(D - 2.0) / max(tau,1e-9)))

def geo_feat_concordance_score(X, F, k=12, max_eval=50000):
    if F is None: return np.nan
    N = len(X)
    rs = np.random.RandomState(0)
    idx = np.arange(N)
    if N > max_eval:
        idx = rs.choice(N, max_eval, replace=False)
    tree = cKDTree(X)
    nb = tree.query(X[idx], k=k+1)[1][:,1:]
    rhos = []
    for i, neigh in zip(idx, nb):
        dx = np.linalg.norm(X[neigh] - X[i], axis=1)
        df = np.linalg.norm(F[neigh] - F[i], axis=1)
        # Spearman via ranks
        rx = np.argsort(np.argsort(dx))
        rf = np.argsort(np.argsort(df))
        rx = (rx - rx.mean()) / (rx.std()+1e-9)
        rf = (rf - rf.mean()) / (rf.std()+1e-9)
        rho = float(np.mean(rx*rf))
        rhos.append(rho)
    rho_m = float(np.mean(rhos))
    return 0.5*(1.0 + rho_m)

def scaler( X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

def center_normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    X -= X.mean(axis=0, keepdims=True)
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x

def centerlize(X, Xm=None, Xs=None):
    if ssp.issparse(X): 
        X = X.toarray()
    X = X.copy()
    N,D = X.shape
    Xm = np.mean(X, 0)

    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs

    return X

def normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x #*((self.DF/2.0)**0.5)
