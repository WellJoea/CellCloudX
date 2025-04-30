import numpy as np
from ..io import read_image_info, write_image_info

def padding_spatial(
                adatas,
                img_key="hires",
                basis = 'spatial', 
                library_id=None,
                inplace=True,
                resize = None):

    if not inplace:
        adatas = [ idata.copy() for idata in adatas ]
    images = []
    locs = []
    sfs = []
    for i in range(len(adatas)):
        iadata = adatas[i]
        imginfo = read_image_info(iadata, img_key=img_key,
                                    basis = basis, 
                                    library_id=library_id,
                                    get_pix_loc=False, 
                                    rescale=None)
        images.append(imginfo['img'])
        locs.append(imginfo['locs'].values)
        sfs.append(imginfo['rescale'])

    maxhw = np.vstack([i.shape[:2] for i in images]).max(0)
    resize = maxhw if resize is None  else resize
    print(f'all the image will set to the same size: {resize}.')

    cpd = padding()
    cpd.fit_transform(images, points=locs, resize=resize)
    for i in range(len(adatas)):
        isf = sfs[i]
        nimg = cpd.imagesT[i]
        nlocs = cpd.pointsT[i]/isf

        write_image_info(adatas[i], 
                         image = nimg, 
                         locs = nlocs, 
                        img_key=img_key,
                        basis = basis, 
                        library_id=library_id,
                        keepraw=False)
    if not inplace:
        return adatas

def padding_images(images, points=None, resize=None, pad_width=None,
                        origin='left', paddims=None, constant_values=0, 
                        **kargs):
    padm = padding()
    padm.fit_transform(images, points=points, resize=resize,
                        pad_width=pad_width, origin=origin,
                        paddims=paddims, constant_values=constant_values, 
                        **kargs)
    if not points is None:
        return padm.imagesT, padm.pointsT
    else:
        return padm.imagesT

class padding():
    def __init__(self):
        pass

    def fit(self, images, resize=None, pad_width=None, paddims=None, origin='center', verbose=False):
        '''
        images: list of images
        resize: list of resize size
        pad_width: list of pad_width
        '''

        if paddims is None:
            if not resize is None:
                paddims = len(resize)
            elif not pad_width is None:
                paddims = len(pad_width)
            else:
                paddims = images[0].ndim
            paddims = range(paddims)
        if verbose:
            print(f'padding dims are {paddims}')

        if (resize is None) and (pad_width is None):
            resize = np.array([i.shape for i in images]).max(0)
            print(f'resize shape is {resize}')

        Padwidth = []
        Padfront = []
        for img in images:
            iwidth = [(0,0)] * img.ndim
            ifront = [0] * img.ndim
            for idim in paddims:
                iwid = img.shape[idim]
                if not pad_width is None:
                    befor, after = pad_width[idim]
                else:
                    twid = resize[idim]
                    if origin == 'center':
                        befor = (twid - iwid)//2
                        after = twid - iwid - befor
                    elif origin == 'left':
                        befor = 0
                        after = twid - iwid
                    elif origin == 'right':
                        befor = twid - iwid
                        after = 0
                iwidth[idim] = (befor, after)
                ifront[idim] = befor
            Padwidth.append(iwidth)
            Padfront.append(ifront)

        self.images = images
        self.pad_width = Padwidth
        self.pad_front = Padfront
        self.resize = resize
        self.pad_dims = paddims

    def transform(self, images=None, pad_width=None,  constant_values= 0, **kargs):
        images = self.images if images is None else images
        pad_width = self.pad_width if pad_width is None else pad_width
        imagesT = [ self.padcrop(images[i], pad_width[i], constant_values=constant_values, **kargs)
                            for i in range(len(images)) ]
        self.imagesT = imagesT
        return imagesT

    def transform_points(self, points, pad_front=None, inversehw=False):
        if points is None:
            pointsT = None
        else:
            pad_front = self.pad_front if pad_front is None else pad_front
            pointsT = [ self.padpos(points[i], pad_front[i], inversehw=inversehw) 
                            for i in range(len(points)) ]
        self.pointsT = pointsT
        return pointsT

    def fit_transform(self, images, points=None, resize=None, pad_width=None,
                        inversehw=False, verbose=False,  origin='center',
                        paddims=None, constant_values=0, **kargs):
        self.fit( images, resize=resize, pad_width=pad_width, paddims=paddims, origin=origin, verbose=verbose)
        self.transform(constant_values=constant_values, **kargs)
        self.transform_points(points, inversehw=inversehw)
        return self

    @staticmethod
    def padsize(img, resize, constant_values= 0, mode ='constant', **kargs):
        ndim = img.ndim
        pad_width=[]
        for idx in range(ndim):
            if idx< len(resize):
                S = img.shape[idx]
                F = resize[idx] - S
                pad_width.append([0, F])
            else:
                pad_width.append([0, 0])
        return np.pad( img, pad_width , mode ='constant', constant_values=constant_values)

    @staticmethod
    def pad(img, 
                pad_width=([30,30],[30,30]),
                constant_values= 0,
                mode ='constant',
                **kargs):

        return np.pad( img, pad_width , mode ='constant', constant_values=constant_values)


    @staticmethod
    def padcrop(img, 
            pad_width=([30,30],[30,30]),
            constant_values= 0,
            mode ='constant',
            **kargs):
        if np.array(pad_width).min()>=0:
            return np.pad( img, pad_width , mode =mode, constant_values=constant_values, **kargs)
        else:
            iimg = img.copy()
            crop = np.clip(pad_width, None, 0)
            pad =  np.clip(pad_width, 0, None)
            sl = [slice(None)] * iimg.ndim
            for i in range(crop.shape[0]):
                sl[i]= slice(np.abs(crop[i][0]),
                            None if crop[i][1]==0 else crop[i][1], 
                            None)
            iimg = iimg[tuple(sl)]
            iimg = np.pad( iimg, pad , mode =mode, constant_values=constant_values, **kargs)
            return iimg.astype(img.dtype)

    @staticmethod
    def padpos(pos, pad_front=[30,30], inversehw=False):
        if len(pad_front)<pos.shape[1]:
            padfull = [0]*pos.shape[1]
            padfull[:len(pad_front)] = pad_front
        else:
            padfull = pad_front[:pos.shape[1]]
        if inversehw:
            padfull = np.asarray(padfull)
            padfull[[0,1]]=padfull[[1,0]]
        return pos + padfull
        # tp, lf = tl
        # ipos = pos.copy()
        # ipos[:,0] += lf
        # ipos[:,1] += tp
        # return ipos.astype(pos.dtype)