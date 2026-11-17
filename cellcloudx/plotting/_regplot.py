import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

def sample_data(T, sample=None):
    if T is None:
        return T
    if type(T).__module__ == 'torch':
        T = T.detach().cpu().numpy()
    else:
        T = np.asarray(T)

    N = T.shape[0]

    if sample is not None and abs(float(sample) - 1.0) > 1e-5:
        if sample > 1.0:
            max_points = int(sample)
        else:
            max_points = int(sample * N)

        max_points = max(1, min(max_points, N))

        if max_points < N:
            idx = np.random.choice(N, max_points, replace=False)
            T = T[idx]
    return T


import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def sample_data(T, sample=None):
    if T is None:
        return T
    if type(T).__module__ == 'torch':
        T = T.detach().cpu().numpy()
    else:
        T = np.asarray(T)

    N = T.shape[0]

    if sample is not None and abs(float(sample) - 1.0) > 1e-5:
        if sample > 1.0:
            max_points = int(sample)
        else:
            max_points = int(sample * N)

        max_points = max(1, min(max_points, N))

        if max_points < N:
            idx = np.random.choice(N, max_points, replace=False)
            T = T[idx]
    return T


def TYs_track(
    X,
    TYs,
    save=None,
    ncols=4,
    s=1.0,
    alpha_x=0.4,
    alpha_ty=0.8,
    sample=None,
    figsize = None,
    fscale = 4,
    herror = 0,
    fps=8,
    title=None,
    dpi=200,

    elev=None,
    azim=None,
    roll=None,
):
    TYs = list(TYs)
    L = len(TYs)
    if TYs[0].shape[1] == 3:
        d3=True
    else:
        d3=False

    if type(s) in [int, float]:
        s = [s, s]

    if (save is None) or (save is False):
        mode = 'png'
    else:
        mode = save.split('.')[-1]

    Xp = sample_data(X, sample)
    TYps = [sample_data(TY, sample) for TY in TYs]

    all_pts = TYps if Xp is None else [Xp] + TYps
    all_pts = np.vstack(all_pts)

    xy_min = all_pts.min(axis=0)
    xy_max = all_pts.max(axis=0)
    pad = 0.04 * (xy_max - xy_min + 1e-12)

    title = title if title is not None else "iters"

    static_modes = ["grid", "png", "jpg", "jpeg", "pdf", "svg"]
    anim_modes = ["gif", "mp4"]

    if mode in static_modes:
        nrows = math.ceil(L / ncols)
        figsize = (fscale * (ncols+herror), fscale * nrows) if figsize is None else figsize

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            subplot_kw={"projection": "3d"} if d3 else None,
        )

        for i, ax in enumerate(axes.ravel()):
            if i >= L:
                ax.axis("off")
                continue

            if not Xp is None:
                if d3:
                    ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], s=s[0], alpha=alpha_x, label="X")
                else:
                    ax.scatter(Xp[:, 0], Xp[:, 1], s=s[0], alpha=alpha_x, label="X")

            if d3:
                ax.scatter(TYps[i][:, 0], TYps[i][:, 1], TYps[i][:, 2], s=s[1], alpha=alpha_ty, label=f"TY[{i}]")
            else:
                ax.scatter(TYps[i][:, 0], TYps[i][:, 1], s=s[1], alpha=alpha_ty, label=f"TY[{i}]")

            ax.set_title(f"{title}: {i+1}")
            ax.set_xlim(xy_min[0] - pad[0], xy_max[0] + pad[0])
            ax.set_ylim(xy_min[1] - pad[1], xy_max[1] + pad[1])

            if d3:
                ax.set_zlim(xy_min[2] - pad[2], xy_max[2] + pad[2])
                ax.view_init(elev=elev, azim=azim, roll=roll)

            ax.set_aspect("equal", adjustable="box")
            # ax.legend(markerscale=4, fontsize=8)

        fig.tight_layout()

        if not save in [None, False]:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        elif save is None:
            plt.show()
        else:
            plt.close(fig)

        return fig


    if mode in anim_modes:
        figsize = (fscale * (1.0+herror), fscale) if figsize is None else figsize

        if d3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize)

        if not Xp is None:
            if d3:
                ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], s=s[0], alpha=alpha_x, label="X")
            else:
                ax.scatter(Xp[:, 0], Xp[:, 1], s=s[0], alpha=alpha_x, label="X")

        if d3:
            scat = ax.scatter(
                TYps[0][:, 0],
                TYps[0][:, 1],
                TYps[0][:, 2],
                s=s[1],
                alpha=alpha_ty,
                label="TY",
            )
        else:
            scat = ax.scatter(
                TYps[0][:, 0],
                TYps[0][:, 1],
                s=s[1],
                alpha=alpha_ty,
                label="TY",
            )

        ax.set_title(f"{title}: {1}")

        ax.set_xlim(xy_min[0] - pad[0], xy_max[0] + pad[0])
        ax.set_ylim(xy_min[1] - pad[1], xy_max[1] + pad[1])

        if d3:
            ax.set_zlim(xy_min[2] - pad[2], xy_max[2] + pad[2])
            ax.view_init(elev=elev, azim=azim, roll=roll)

        ax.set_aspect("equal", adjustable="box")
        # ax.legend(markerscale=4, fontsize=8)

        def update(i):
            if d3:
                scat._offsets3d = (
                    TYps[i][:, 0],
                    TYps[i][:, 1],
                    TYps[i][:, 2],
                )
            else:
                scat.set_offsets(TYps[i])

            ax.set_title(f"{title}: {i+1}")
            return scat,

        ani = FuncAnimation(
            fig,
            update,
            frames=L,
            interval=1000 / fps,
            blit=False,
        )

        ani_save = None

        if save not in [None, False]:
            ani_save = save

            if mode == "gif":
                ani.save(ani_save, writer=PillowWriter(fps=fps), dpi=dpi)
            elif mode == "mp4":
                ani.save(ani_save, writer=FFMpegWriter(fps=fps), dpi=dpi)
            else:
                raise ValueError("mode error:  'gif', 'git', 'mp4'")

        elif save is None:
            plt.show()
        else:
            plt.close(fig)
        return fig