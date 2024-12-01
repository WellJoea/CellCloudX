import pandas as pd
import numpy as np
import scanpy as sc
import PIL
import io
import base64

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
    img_obj = PIL.Image.fromarray(image)
    prefix = "data:image/png;base64,"
    with io.BytesIO() as stream:
        img_obj.save(stream, format='png')
        b64_str = prefix + base64.b64encode(stream.getvalue()).decode('unicode_escape')
    return b64_str