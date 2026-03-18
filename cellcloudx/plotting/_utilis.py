import pandas as pd
import numpy as np
import scanpy as sc

import io
import base64

from matplotlib import pyplot as plt

def spider(df, id_column=None, columns=None, max_values=None, 
            title=None, alpha=0.15, 
            color_bg='#A0A0A0', alpha_bg=0.05, 
            colors=None, fs='xx-small', fs_format = '.3f',
            padding=1.05, figsize=(8,8), rotate_label=True,
             show_legend=True, bbox_to_anchor=(0.1, 0.1),
            saveargs={}, show=True, save=None, ax=None,
            **kargs):
    columns = df._get_numeric_data().columns.tolist() if columns is None else columns
    data = df[columns].to_dict(orient='list')
    ids = df.index.tolist() if id_column is None else df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding*max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    degrees = [ np.degrees(angle-np.pi if angle>np.pi else angle ) - 90 for angle in angles]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        icolor = None if colors is None else colors[i]

        ax.plot(angles, values, c=icolor, label=model_name, **kargs)
        ax.fill(angles, values, c=icolor, alpha=alpha)
        for _x, _y, t, r in zip(angles, values, actual_values, degrees):
            t = f'{t :{fs_format}}' if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, size=fs, rotation=r,
                    rotation_mode='anchor')

    if not color_bg is None:
        ax.fill(angles, np.ones(num_vars + 1), alpha=alpha_bg)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    # ax.grid(linewidth=3)

    if rotate_label:
        for label, degree in zip(ax.get_xticklabels(), degrees):
            x,y = label.get_position()
            lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            lab.set_rotation(degree)
        ax.set_xticklabels([])

    if show_legend: ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
    if title is not None: plt.suptitle(title)

    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, **saveargs)
    if show is None:
        return fig, ax
    elif show is True:
        plt.show()
    else:
        plt.close()



def colrows(ncell, nrows=None, ncols=None, soft=False):
    import math
    if (ncols is None) and (nrows is None):
        ncols = int(np.ceil(ncell**0.5))
        soft = True
    if not ncols is None:
        nrows = math.ceil(ncell/ncols)
        ncols = min(ncell, ncols)
    elif not nrows is None:
        ncols = math.ceil(ncell/nrows)
        nrows = min(ncell, nrows)
    if soft and ncell> 1 and (ncell - ncols*(nrows-1)<=1):
        ncols += 1
        nrows -= 1
    return (nrows, ncols)


def clipdata(Data, vmin=None, vmax=None, pmin=None, pmax=None,  clips=None, tmin = None, dropmin=False,):
    Data = Data.copy()

    if not pmin is None:
        vmin = np.percentile(Data, pmin)
    if not pmax is None:
        vmax = np.percentile(Data, pmax)

    if  (not vmax is None):
        if clips is None:
            clips = (None, vmax)
        else:
            clips = (clips[0], vmax)
    if  (not vmin is None):
        if dropmin and np.ndim(Data) ==1:
            Data = Data[Data>vmin]
        if tmin is None:
            tmin = vmin
        Data[Data<vmin] = tmin


    

    if not clips is None:
        if isinstance(Data, (pd.DataFrame, pd.Series)):
            Data = Data.clip(clips[0], clips[1])
        else:
            Data = np.clip(Data, clips[0], clips[1])

        Data = np.clip(Data, vmin, vmax)
    return Data

def image2batyes(image, scale=True):
    if scale and (image.max()>255):
        amin = image.min()
        amax = image.max()
        image = np.clip(255.0 * (image-amin)/(amax-amin), 0, 255).astype(np.uint8)
    import PIL
    img_obj = PIL.Image.fromarray(image)
    prefix = "data:image/png;base64,"
    with io.BytesIO() as stream:
        img_obj.save(stream, format='png')
        b64_str = prefix + base64.b64encode(stream.getvalue()).decode('unicode_escape')
    return b64_str