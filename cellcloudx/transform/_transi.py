import numpy as np
import skimage as ski

from ..utilis._arrays import isidentity
from joblib import Parallel, delayed

try:
    from scipy.ndimage import map_coordinates
except ImportError:
    from scipy.ndimage.interpolation import map_coordinates

from skimage import transform as skitf

from ._padding import padding

def update_tmats(align_pair, trans_pair, align_tmats, init_tmats=None, size=None):
    if size is None:
        if init_tmats is None:
            size = max(sum(align_pair,())) + 1
        else:
            size = len(init_tmats)
    if init_tmats is None:
        init_tmats = [np.eye(3, dtype=np.float64) for _ in range(size)]

    assert len(init_tmats) == size
    assert len(align_tmats) == len(align_pair)

    new_tmats = [np.eye(align_tmats[0].shape[0], dtype=np.float64) for _ in range(size)]

    for i,(fidx, midx) in enumerate(align_pair):
        new_tmats[midx] = align_tmats[i] @ init_tmats[midx]

    for ifov, imov in trans_pair:
        new_tmats[imov] = np.matmul(new_tmats[imov], new_tmats[ifov])
    return new_tmats

def imageres(image, dtype, scale_max=None):
    scale_max = scale_max or uint_scale[dtype]
    return np.clip(np.round(image * scale_max), 0, scale_max).astype(dtype)

def imagerestrans(image, sdtype, tdtype):
    return imageres(image/uint_scale[sdtype], tdtype)

def rotateion(image, degree=np.pi, keeptype=True, **kargs):
    hw = image.shape[:2]
    shift = np.eye(3)
    shift[:2,2] = [hw[0]/2, hw[1]/2]
    tform = ski.transform.EuclideanTransform(rotation=degree).params.astype(np.float64)
    tform = shift @ tform @ np.linalg.inv(shift)
    tform = tform.astype(np.float64) 
    imagen = ski.transform.warp(image,  tform, **kargs)
    if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
        imagen = imageres(imagen, image.dtype.type)
    return [imagen, tform ]

def rescale_tmat(tmat, sf, trans_scale=True):
    dimension = tmat.shape[0]-1
    scale_l  = np.eye(dimension+1)
    scale_l[range(dimension), range(dimension)] = sf
    scale_l[:dimension, dimension] = sf
    scale_r = np.eye(dimension+1)
    scale_r[range(dimension), range(dimension)] = 1/sf

    if trans_scale:
        return scale_l @ tmat @ scale_r
    else:
        return scale_l @ tmat

def resize(image, reshape, 
           order=None, 
           mode='reflect',
           cval=0, 
           clip=True, 
           method = 'skimage',
           keeptype=True,
           cv2interp = 3, **kargs):
    if method == 'skimage':
        imagen = ski.transform.resize(image,  
                                        reshape,
                                        order=order,
                                        cval=cval,
                                        mode=mode,
                                        clip=clip, **kargs)
    elif method == 'cv2':
        try:
            import cv2
        except:
            print('you can use cv2 by "pip install opencv-python", or switch method to "ski".')
        imagen = cv2.resize(image, reshape, interpolation= cv2interp )

    if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
        imagen = imageres(imagen, image.dtype.type)

    return imagen

def resizes(images, reshapes, 
           order=None, 
           mode='reflect',
           cval=0, 
           clip=True, 
           keeptype=True,
           method = 'skimage',
            n_jobs = 10,
            backend="multiprocessing",
            verbose=0,

           cv2interp = 3, **kargs):
    if type(reshapes[0]) in [int]:
        reshapes = [reshapes] * len(images)
    assert len(images) == len(reshapes), 'the length between images and reshapes must be same.'
    if n_jobs != 1:
        imagen = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)\
                            (delayed(resize)(images[i], reshapes[i],
                                    order=order, 
                                    mode=mode,
                                    cval=cval, 
                                    clip=clip, 
                                    method = method,
                                    keeptype=keeptype,
                                    cv2interp = cv2interp, **kargs)
                            for i in range(len(images)))
    else:
        imagen = [resize(images[i], reshapes[i],
                        order=order, 
                        mode=mode,
                        cval=cval, 
                        clip=clip, 
                        method = method,
                        keeptype=keeptype,
                        cv2interp = cv2interp, **kargs) for i in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def homoreshape(image, 
                tform =None,
                sf=None,
                rescale = None,
                padsize = None, **kargs):
    if (tform is None) and (sf is None) and (rescale is None) and (padsize is None):
        return image, None
    else:
        scale = sf or 1
        if rescale:
            import skimage as ski
            rsl = np.array([rescale/scale, rescale/scale, 1])[:image.ndim]
            image = ski.transform.rescale(image, rsl, **kargs)
            re_isf = rescale
        else:
            re_isf = scale

        itam_sf = rescale_tmat(tform, re_isf, trans_scale=True)
        inew_img = homotransform(image, itam_sf, **kargs)
        if not padsize is None:
            ipad = padding()
            ipad.fit_transform([ inew_img ], resize=padsize, origin='left')
            inew_img = ipad.imagesT[0]
        return inew_img, itam_sf

def swap_tmat(tmat, swap_axes=[0,1], xp=np):
    if swap_axes is None:
        return tmat
    else:
        ndim = tmat.shape[0]-1
        assert max(swap_axes) < ndim, 'swap_axes must be smaller than the dimension of the image.'
        sw = xp.eye(ndim+1, dtype=tmat.dtype, device=tmat.device)
        sw[swap_axes,:] = sw[swap_axes[::-1],:]
        return sw @ tmat @ sw.T

def homotransform(
        image,
        tmat,
        order=3,
        keeptype=True,
        interp_method = 'skimage',
        inverse=False,
        swap_xy=False,
        swap_axes=None,
        **kargs):
    tmat = np.array(tmat)
    if swap_xy:
        swap_axes = [0,1]
    if swap_axes is not None:
        tmat = swap_tmat(tmat, swap_axes=swap_axes)

    if isidentity(tmat):
        return image

    if inverse:
        tmat = np.linalg.inv(tmat)

    if interp_method=='skimage':
        imagen = ski.transform.warp(image, tmat, order=order, **kargs)

        if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
            imagen = imageres(imagen, image.dtype.type)

    return imagen

def homotransforms( images,
                    tmats,
                    order=3,
                    keeptype=True,
                    interp_method = 'skimage',
                    n_jobs = 10,
                    backend="multiprocessing",
                    verbose=0,
                    **kargs):
    assert len(images) == len(tmats), 'the length between images and tmats must be same.'
    if n_jobs != 1:
        imagen = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)\
                            (delayed(homotransform)(images[i], tmats[i],
                                    order=order,
                                    keeptype=keeptype,
                                    interp_method = interp_method,**kargs)
                            for i in range(len(images)))
    else:
        imagen = [homotransform(images[i], tmats[i],
                                order=order,
                                keeptype=keeptype,
                                interp_method = interp_method,
                                **kargs) for i  in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def fieldtransform(image, coords=None, U=None, V=None, mode='edge', order=3, use_ski=True, keeptype=True, **kargs):
    if coords is None:
        nr, nc = image.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), 
                                            np.arange(nc),
                                            indexing='ij')
        coords = np.array([row_coords + V, col_coords + U])

    if image.ndim ==3:
        imagergb = []
        for i in range(image.shape[2]):
            iimg = fieldtransform(image[:,:,i], 
                                    coords, 
                                    mode=mode,
                                    order=order, 
                                    use_ski=use_ski,
                                    **kargs)
            imagergb.append(iimg[:,:,np.newaxis])
            imagergb = np.concatenate(imagergb, axis=2)
            return(imagergb)
    else:
        if use_ski:
            imagen = skitf.warp(image, coords, mode=mode, order=order, **kargs)
        else:
            imagen = map_coordinates(image, coords, prefilter=False, 
                                        order=order, 
                                        mode=mode, **kargs)
        if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
            imagen = imageres(imagen, image.dtype.type)
        return imagen

def fieldtransforms( images,
                    coords,
                    n_jobs = 10,
                    backend="multiprocessing",
                    verbose=0,
                    **kargs):
    assert len(images) == len(coords), 'the length between images and tmats must be same.'
    if n_jobs >1:
        imagen = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)\
                            (delayed(fieldtransform)(images[i], coords[i],**kargs)
                            for i in range(len(images)))
    else:
        imagen = [fieldtransform(images[i], coords[i], **kargs) for i  in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def Transform(image, tmats, mode='edge'):
    assert isinstance(tmats, list)
    imaget = image
    for itmat in tmats:
        if (itmat.ndim==2) and (itmat.shape[0] == itmat.shape[1]):
            assert 3<=itmat.shape[0]<=4
            imaget = homotransform(image, itmat)
        elif (itmat.ndim >=3) and (image.shape[:2] == itmat.shape[1:3]):
            imaget = fieldtransform(image, itmat, mode=mode)
        else:
            raise('error tmats formats')
    return imaget

def transform_points_oftvl1(points, coords, scale_max = 10):
    points = points[:,[1,0,2]].copy()
    pointimg = np.arange(np.prod(coords[0].shape)).reshape(coords[0].shape)
    points = np.c_[points, pointimg[points[:,0], points[:,1]]]

    imgwarp = map_coordinates(pointimg, coords,
                            prefilter=False,
                            mode='nearest',
                            order=0, cval=0.0)
    def get_cub_postion(point, scale):
        XX, YY = np.meshgrid(np.arange(point[0] - scale, point[0] + scale +1),
                            np.arange(point[1] - scale, point[1] + scale +1))
        return(XX, YY)

    pointsnew = []
    for i in range(points.shape[0]):
        iloc = points[i]
        scale = 0
        X, Y = [], []
        while (len(X)==0) and (scale<=scale_max):
            ipos = pointimg[get_cub_postion(iloc[:2], scale)].flatten()
            X, Y= np.where(np.isin(imgwarp, ipos))
            scale +=1
        if len(X)>0:
            X = np.mean(X)
            Y = np.mean(Y)
        else:
            X, Y = np.inf, np.inf
        if scale>5:
            print(i, scale)
        iloc = list(iloc) + [X, Y]
        pointsnew.append(iloc)
    pointsnew = np.array(pointsnew)
    return pointsnew[:,[5,4,2]]

def rescale(image, scale=None, method='skimage',
            order=None, 
            mode='reflect',
            cval=0, 
            clip=True, 
            keeptype=True,
            cv2interp = 3,
            **kargs):
    if (scale is None) or (scale==1):
        return image

    if type(scale) in [int, float]:
        scale = [scale, scale]
    reshape = np.round(np.array(image.shape[:2]) * np.array(scale), 0).astype(np.int64)
    retmat = np.eye(3)
    retmat[[0,1], [0,1]] = scale

    if method in ['homotrans']: # error
        imagen = homotransform(image,
                                retmat,
                                order=order,
                                keeptype=keeptype,
                                interp_method = 'skimage',
                                **kargs)

    else:
        #ski.tf.rescale
        imagen = resize(image, reshape,
                        order=order, 
                        mode=mode,
                        cval=cval, 
                        clip=clip, 
                        keeptype=keeptype,
                        method = method,
                        cv2interp = cv2interp,
                        **kargs)
    return imagen

def rescales(images, scales, 
           order=None, 
           mode='reflect',
           cval=0, 
           clip=True, 
           keeptype=True,
           method = 'skimage',
           cv2interp = 3, **kargs):

    if type(scales)  in [float, int]:
        scales = [scales] * len(images)
    assert len(images) == len(scales), 'the length between images and scales must be same.'
    imagen = [rescale(images[i], scales[i],
                    order=order, 
                    mode=mode,
                    cval=cval, 
                    clip=clip, 
                    method = method,
                    keeptype=keeptype,
                    cv2interp = cv2interp, **kargs) for i in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def mirroraxis(array, points=None, x=False, y=False, z=False, axes=None):
    '''
    array :  x, y, [z,...]
    points:  list[y,z]
    '''
    if (not axes is None) and isinstance(axes, int):
        axes = [axes]
    if (not axes is None):
        assert isinstance(axes, list), 'axes must be a list or int.'
    if isinstance(array, list):
        array = np.array(array)
    if (not array is None) and (not points is None):
        assert len(array) == len(points)

    sl = [slice(None)] * array.ndim
    ts = []
    for i,k in enumerate([x,y,z]):
        if k:
            ts.append(i)
    if not axes is None:
        ts = [*ts, *axes]
    for its in ts:
        sl[its]=slice(None, None, -1)

    arrayn = array[tuple(sl)] if ts else array
    if points is None:
        return arrayn

    if isinstance(points, list) and (points[0].shape[1] == 2) and (array.ndim>=3):
        pointns = []
        for ipos in points:
            ipos = ipos.copy()
            for its in ts:
                if its != 0:
                    # ipos[:, 2-its] = array.shape[its] - ipos[:, 2-its]
                    ipos[:, its-1] = array.shape[its] - ipos[:, its-1]
            pointns.append(ipos)
        if 0 in ts:
            pointns = pointns[::-1]
    else:
        pointns = points.copy()
        for its in ts:
            pointns[:, its] = array.shape[its] - pointns[:, its]
    return arrayn, pointns

def mirroraxis0(array, points=None, x=False, y=False, z=False, axes=None):
    '''
    array : [z,] x, y
    points: [z,] x, y
    '''
    if (not axes is None) and isinstance(axes, int):
        axes = [axes]
    if (not axes is None):
        assert isinstance(axes, list), 'axes must be a list or int.'

    sl = [slice(None)] * array.ndim
    ts = []
    for i,k in enumerate([x,y,z]):
        if k:
            ts.append(i)
    if not axes is None:
        ts = [*ts, *axes]
    for its in ts:
        sl[its]=slice(None, None, -1)

    arrayn = array[tuple(sl)] if ts else array
    if points is None:
        return arrayn

    if isinstance(points, list) and (points[0].shape[1] == 2) and (array.ndim>=3):
        pointns = []
        for ipos in points:
            ipos = ipos.copy()
            if not np.all(ipos.max(0)< array.shape[-2:]):
                print('warning: image array ([z, x, y]) should be consistent with points(list of [x, y])')
            for its in ts:
                if its != 0:
                    # ipos[:, 2-its] = array.shape[its] - ipos[:, 2-its]
                    ipos[:, its-1] = array.shape[its] - ipos[:, its-1]
            pointns.append(ipos)
        if 0 in ts:
            pointns = pointns[::-1]
    else:
        pointns = points.copy()
        if not np.all(pointns.max(0) <= array.shape[-2:]):
            print('warning: image array ([x, y, z]) should be consistent with points ([x, y, z])')

        for its in ts:
            pointns[:, its] = array.shape[its] - pointns[:, its]
    return arrayn, pointns

def tmatinverse(tmat):
    return np.linalg.inv(tmat)

def flipaxis(imgshape, dim =2, axis=0):
    dim = dim or len(imgshape)
    WHC = np.array(imgshape)
    WHC[[0,1]] = WHC[[1,0]]
    N = WHC[axis]

    fmat = np.eye(dim+1)
    fmat[axis, axis] = -1
    fmat[axis, -1] = N - 1
    return fmat

def tmatflip(tmat, imgshape, dim = None, axis=0):
    return np.array(tmat) @ flipaxis(imgshape, dim = dim, axis=axis)

def Flip(img, dim = 2, axis=1):
    tmat = flipaxis(img.shape, dim=dim, axis=axis)
    return homotransform(img, tmat)

def padsize(img, pad_width=([30,30],[30,30]), constant_values= 0, mode ='constant', **kargs):
    return np.pad(img, pad_width , mode ='constant', constant_values=constant_values)
    # iimg = img.copy()
    # tp,bl = pad_width[0]
    # lf,rg = pad_width[1]
    
    # top   = np.zeros([tp] + list(iimg.shape[1:])) + constant_values
    # below = np.zeros([bl] + list(iimg.shape[1:])) + constant_values

    # iimg = np.concatenate([top, iimg, below], axis=0)
    # left  = np.zeros([iimg.shape[0], lf] + list(iimg.shape[2:])) + constant_values
    # right = np.zeros([iimg.shape[0], rg] + list(iimg.shape[2:])) + constant_values
    # iimg = np.concatenate([left, iimg, right], axis=1)
    # return iimg.astype(img.dtype)

def padcenter(imglist, hw=None, **kargs):
    if hw is None:
        max_hw = max([ max(i.shape[:2]) for i in imglist ])
        H, W = [max_hw,max_hw]
    else:
        H, W =hw[:2]
    imagen  = []
    tblr = []
    for img in imglist:
        h, w = img.shape[:2]
        tp = (H - h)//2
        bl = H - h - tp
        lf = (W - w)//2
        rg = W - w -lf
        pad_width = [(0,0)] * img.ndim
        pad_width[0] = (tp, bl)
        pad_width[1] = (lf, rg)
        iimg = padsize(img, pad_width=pad_width, **kargs)
        imagen.append(iimg)
        tblr.append([tp, bl, lf, rg])
    return [imagen,pad_width]

def ceil_d(C, precision=0):
    return np.round(C + 0.5 * 10**(-precision), precision)

def reshapeonref(refimg, movimg, precision=3):
    from ..transform._padding import padding
    refsize = np.array(refimg.shape[:2])
    movsize = np.array(movimg.shape[:2])
    iscale = np.float64((movsize/refsize).max())
    padsize = np.round(iscale * refsize,0).astype(np.int64)
    rp = padding()
    rp.fit_transform( [movimg], points = None, resize=padsize)
    movN = np.array(rp.imagesT).astype(movimg.dtype)

    movN = resize(movN[0], refsize)
    return movN, iscale

uint_scale = {
    np.float64:1.,
    np.float16:1.,
    np.float32:1.,
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.uint64: 18446744073709551615,
}

uint_mrange = {
    np.uint8: (0, 255),
    np.uint16: (255, 65535),
    np.uint32: (65535, 4294967295),
    np.uint64: (4294967295, 18446744073709551615),
}

def uint_value(dtype, maxv=None, verbose=True):
    if  np.issubdtype(dtype, np.floating):
        ttype = dtype
        dvalue = maxv or 1
    elif maxv <=1 :
        ttype = dtype
        dvalue = 1
    elif 1 < maxv <= 255 :
        ttype = np.uint8
        dvalue = 255
        if verbose and (not dtype in [np.uint8]):
            print("Warning: image dtype rescaled to 8-bit")
    elif 255 < maxv <= 65535:
        ttype = np.uint16
        dvalue = 65535
        if verbose and (not dtype in [np.uint16]):
            print("Warning: image dtype rescaled to 16-bit")
    elif 65535 < maxv <= 4294967295:
        ttype = np.uint32
        dvalue = 4294967295
        if verbose and (not dtype in [np.uint32]):
            print("Warning: image dtype rescaled to 32-bit")
    elif 4294967295 < maxv <= 18446744073709551615:
        ttype = np.uint64
        dvalue = 18446744073709551615
        if verbose and (not dtype in [np.uint64]):
            print("Warning: image dtype rescaled to 64-bit")
    else:
        ttype = dtype
        dvalue = maxv

    dvalue = maxv or dvalue
    return [ttype, dvalue]