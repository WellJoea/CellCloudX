import os
import numpy as np
import pandas as pd
import skimage as ski
import ants

from ..transform._transi import swap_tmat
## WARP from antspy
class antsreg():
    def __init__(self, transtype='SyN', ndim=2, verbose=False,):
        self.ants = ants
        self.transtype= 'Rigid' if transtype == 'rigid' else transtype
        self.ndim= ndim
        self.verbose = verbose
        self.get_antstpara=get_antstpara
        self.get_antstmats=get_antstmats
        self.save_antstx = save_antstx

    def regist(self, fix, mov,
                    write_composite_transform=False,
                    get_mtx = True,
                    random_seed = 0,
                    verbose = False,
                    swap_xy = True,
                    **kargs):
        self.fix = fix
        self.mov = mov
        if isinstance(self.fix, np.ndarray):
            self.fixi = ants.from_numpy( self.rgb2gray( self.fix, ndim=self.ndim))
        else:
            self.fixi = self.fix

        if isinstance(self.mov, np.ndarray):
            self.movi = ants.from_numpy(self.rgb2gray( self.mov, ndim=self.ndim) )
        else:
            self.movi = self.mov

        mytx = ants.registration(fixed=self.fixi , 
                                 moving=self.movi, 
                                type_of_transform=self.transtype,
                                write_composite_transform=write_composite_transform,
                                random_seed = random_seed,
                                verbose=verbose,
                                **kargs)
        self.mytx = mytx
        if get_mtx:
            self.mytm = self.get_antstpara(mytx, swap_xy=swap_xy)
            self.tmats = [ i['tmat'] for i in  self.mytm['fwdtransforms'] ]
            self.tmatr = [ i['tmat'] for i in  self.mytm['invtransforms'] ]
        return self

    def transform(self, moving_img=None,
                   tmat=None, inverse=False, locs=None,
                   interpolator='linear',
                   imagetype=0,
                    whichtoinvert=None):
        moving_img = self.movi if moving_img is None else moving_img
        tmat = self.mytx if tmat is None else tmat

        imat = tmat['invtransforms'] if inverse else tmat['fwdtransforms']
        pmat = tmat['fwdtransforms'] if inverse else tmat['invtransforms']

        mov_out = self.ants.apply_transforms(self.fixi, 
                                          moving_img,
                                          transformlist=imat,
                                          interpolator=interpolator,
                                          whichtoinvert=whichtoinvert,
                                          imagetype=imagetype,
                                          verbose=self.verbose,
                                          )
        #cc.tf.homotransform(moving_img, self.tmatr, inverse=False, swap_xy=(not swap_xy))
        self.mov_out = mov_out
        self.mov_locs = None
        if not locs is None:
            mov_locs = self.transform_points(locs, inverse=inverse, 
                                             whichtoinvert=whichtoinvert)
            return mov_out, mov_locs
        else:
            return mov_out

    def transform_points(self, locs, tmat=None, inverse=False,  
                         whichtoinvert=None):
        tmat = self.mytx if tmat is None else tmat
        if inverse:
            pmat = tmat['invtransforms']
            mmat = self.tmatr
        else:
            pmat = tmat['fwdtransforms']
            mmat = self.tmats
        if whichtoinvert is None:
            whichtoinvert = [ True if i.shape == (self.ndim+1, self.ndim+1) else False 
                              for i in mmat]


        if isinstance(locs, np.ndarray):
            locs = pd.DataFrame(locs, columns=['x', 'y', 'z'][:self.ndim])

        mov_locs = self.ants.apply_transforms_to_points( self.ndim,
                                        locs,
                                        transformlist=pmat,
                                        whichtoinvert=whichtoinvert,
                                        verbose=self.verbose).values
        # cc.tf.homotransform_point(locs, mmat, inverse=True, swap_xy= (swap_xy))
        self.mov_locs = mov_locs
        return mov_locs

    def regist_transform(self, fix, mov,
                            write_composite_transform=False,
                            tmat=None, inverse=False, locs=None,
                            interpolator='linear',
                            imagetype=0,
                            whichtoinvert=None,
                            **kargs):
        self.regist(fix, mov, 
                    write_composite_transform=write_composite_transform,
                     **kargs)
        self.transform(tmat=tmat, inverse=inverse, locs=locs,
                        interpolator=interpolator,
                        imagetype=imagetype,
                        whichtoinvert=whichtoinvert,)
        # return self
        return [self.mov_out, self.tmats, self.mov_locs]
    
    def rgb2gray(self, image, ndim=2):
        if (ndim==2) and (image.ndim==3) and (image.shape[2]==3):
            return ski.color.rgb2gray(image)
        else:
            return image

def get_antstpara(tfile, swap_xy=False):
    if isinstance(tfile, str) and os.path.exists(tfile):
        if tfile.endswith('.h5'):
            tmats = []
            import h5py
            h5dt = h5py.File( tfile, 'r')
            for i in range(np.array(h5dt['TransformGroup']).shape[0]):
                if i >0:
                    ih5 = h5dt['TransformGroup'][str(i)]
                    ttype = str(np.array(ih5['TransformType'])[0], encoding='utf-8')
                    dim = int(ttype[-1])
                    fix_para = np.array(ih5['TransformFixedParameters'])
                    tmatx =  np.array(ih5['TransformParameters'])
                    if fix_para.shape[0]>3:
                        ishape = list(fix_para[:dim].astype(np.int64))
                        dshape = [dim, *ishape]
                        tmatx = tmatx.reshape(dshape, order='F')
                        tmatx = np.rollaxis(tmatx, 0, dim+1)
                    imat = { 'fix_para': fix_para,
                                'tinfo': tmatx, 
                                'ttype': dim}
                    imat['tmat'] = get_antstmats(imat, ndim=dim, swap_xy=swap_xy)
                    tmats.append(imat)

        elif tfile.endswith('.mat'):
            import scipy.io as sio
            infor = sio.loadmat(tfile)
            ttype = [i for i in infor.keys() if 'form_float_' in i ][0]
            tmats = { 'fix_para': infor['fixed'].flatten(),
                        'tinfo': infor[ttype].flatten(), 
                        'ttype': ttype}
            dim = int(ttype[-1])
            tmats['tmat'] = get_antstmats(tmats, ndim=dim, swap_xy=swap_xy)

        elif tfile.endswith('.nii.gz'):
            import scipy.io as sio
            infor = ants.image_read(tfile)
            fix_para = np.hstack([infor.shape, infor.origin, infor.spacing, infor.direction.flatten()])
            dim = infor.dimension
            ttype = f'DisplacementFieldTransform_float_{dim}_{dim}'
            tmats = {'fix_para': fix_para,
                        'tinfo': infor.numpy(),
                        'ttype': ttype}
            tmats['tmat'] = get_antstmats(tmats, ndim=dim, swap_xy=swap_xy)
        return tmats

    elif isinstance(tfile, dict):
        tmats = {}
        for wdtf in ['fwdtransforms', 'invtransforms']:
            if wdtf in tfile:
                ifle = tfile[wdtf]
                if isinstance(ifle, list):
                    tmats[wdtf] = [ get_antstpara(i, swap_xy=swap_xy) for i in ifle ]

                elif isinstance(ifle, str):
                    tmats[wdtf] = get_antstpara(ifle, swap_xy=swap_xy)
        return tmats

def save_antstx(mytx, path, outpre=None, fmat ='pkl'):
    import os
    import shutil
    import pickle
    import json
    txs = {}
    for k,v in mytx.items():
        if k in ['fwdtransforms', 'invtransforms']:
            nfms = []
            for ifm in v:
                base_name = os.path.basename(ifm)
                base_name = f"{k}.{base_name.split('.')[-1]}"
                if not outpre is None:
                    base_name = f'{outpre}_{base_name}'
                file_name = os.path.join(path, base_name)
                shutil.copyfile(ifm, file_name)
                nfms.append(file_name)
            txs[k] = nfms

    if not outpre is None:
        file_name = os.path.join(path, f'{outpre}_antstransform.{fmat}')
    else:
        file_name = os.path.join(path, f'antstransform.{fmat}')

    if fmat == 'pkl':
        with open(file_name, 'wb') as outp:
            pickle.dump(txs, outp, pickle.HIGHEST_PROTOCOL)
    elif fmat == 'json':
        with open(file_name, 'w') as outp:
            json.dump(txs, outp)
    else:
        raise(ValueError(f'Unsupported file format: {fmat}'))

    return txs

def get_antstmats(tinfor, ndim=None, swap_xy=False):
    fix_para = tinfor['fix_para']
    tmatx = tinfor['tinfo']
    ttype = tinfor['ttype']
    ndim = ndim or int(ttype[-1])
    if (len(fix_para) >3 and tmatx.ndim>2) or 'DisplacementField'.lower() in ttype.lower():
        iS = fix_para[:ndim]
        VUs = np.rollaxis(tmatx, -1, 0)
        return VUs

    elif (len(fix_para) <=3) and (tmatx.ndim ==1):
        ldim = ndim+1
        itmat = np.eye(ldim)
        A = tmatx[:(ndim*ndim)].reshape(ndim,ndim, order='C')
        T = tmatx[(ndim*ndim):]
        C = fix_para
        TC = T + C - A @ C
        itmat[:ndim,:ndim] = A
        itmat[:ndim, ndim] = TC
        #iS = np.eye(ldim)
        #iS[:ndim, ndim] = fix_para
        #itmat = iS  @ itmat @  np.linalg.inv(iS)

        if swap_xy:
            itmat = swap_tmat(itmat, swap_axes=[0,1])
            # print(itmat)
            # A = tmatx[:(ndim*ndim)][::-1].reshape(ndim,ndim, order='C')
            # T = tmatx[(ndim*ndim):][::-1]
            # C = fix_para[::-1]
            # TC = T + C - A @ C #itmat = iS  @ itmat @  np.linalg.inv(iS)
            # itmat[:ndim,:ndim] = A
            # itmat[:ndim, ndim] = TC
            # print(itmat)

        return itmat

def get_antstmats1(tinfor, ndim=None, center=True):
    fix_para = tinfor['fix_para']
    tmatx = tinfor['tinfo']
    ttype = tinfor['ttype']
    ndim = ndim or int(ttype[-1])
    if (len(fix_para) >3 and tmatx.ndim>2) or 'DisplacementField'.lower() in ttype.lower():
        iS = fix_para[:ndim]
        VUs = np.rollaxis(tmatx, -1, 0)
        return VUs

    elif (len(fix_para) <=3) and (tmatx.ndim ==1):
        ldim = ndim+1
        itmat = np.eye(ldim)
        itmat[:ndim,:ndim] = tmatx[:(ndim*ndim)].reshape(ndim,ndim, order='C')
        itmat[:ndim, ndim] = tmatx[(ndim*ndim):]

        iS = np.eye(ldim)
        iS[:ndim, ndim] = fix_para
        if center:
            itmat = iS  @ itmat @  np.linalg.inv(iS)

            #iS[:ndim, ndim] = t+c -A@c
        return itmat


# import ants TODO
# mi_deformed = ants.apply_transforms(fixed=ants.from_numpy(fimg),
#                                     moving=ants.from_numpy(mimg),
#                                     transformlist=rgant.mytx['fwdtransforms'],
#                                     whichtoinvert=[False, False])
# fi_deformed = ants.apply_transforms(fixed=ants.from_numpy(mimg),
#                                     moving=ants.from_numpy(fimg),
#                                     transformlist=rgant.mytx['invtransforms'],
#                                     whichtoinvert=[True, False])

# mi_locs = ants.apply_transforms_to_points(2,
#                               pd.DataFrame(mloc, columns=['x', 'y', 'z'][:2]) ,
#                               transformlist=rgant.mytx['invtransforms'],
#                               whichtoinvert=[True, False],
#                               verbose=0).values
# fi_locs = ants.apply_transforms_to_points(2,
#                               pd.DataFrame(mloc, columns=['x', 'y', 'z'][:2]) ,
#                               transformlist=rgant.mytx['fwdtransforms'],
#                               whichtoinvert=[False, False],
#                               verbose=0).values
# iiiloc = cc.tf.fieldtransform_point(mloc + rgant.tmatr[0][:2,-1], -rgant.tmatr[1][[1,0]], method='linear')
# cc.pl.qview(mi_locs, mloc, fi_locs, iiiloc, size=0.1, sharex=True, sharey=True)