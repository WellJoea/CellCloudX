import os
import numpy as np
import pandas as pd
import skimage as ski
import ants

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
                    verbose=False,
                    center = True,
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
            self.mytm = self.get_antstpara(mytx, center=center)
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
        self.mov_out = mov_out
        self.mov_locs = None
        if locs is not None:
            if isinstance(locs, np.ndarray):
                locs = pd.DataFrame(locs, columns=['x', 'y', 'z'][:self.ndim])

            mov_locs = self.ants.apply_transforms_to_points( self.ndim,
                                          locs,
                                          transformlist=pmat,
                                          whichtoinvert=whichtoinvert,
                                          verbose=self.verbose).values
            self.mov_locs = mov_locs
            return mov_out, mov_locs
        else:
            return mov_out

    def transform_points(self, locs, tmat=None, inverse=False,  whichtoinvert=None):
        tmat = self.mytx if tmat is None else tmat
        pmat = tmat['fwdtransforms'] if inverse else tmat['invtransforms']
        if isinstance(locs, np.ndarray):
            locs = pd.DataFrame(locs, columns=['x', 'y', 'z'][:self.ndim])

        mov_locs = self.ants.apply_transforms_to_points( self.ndim,
                                        locs,
                                        transformlist=pmat,
                                        whichtoinvert=whichtoinvert,
                                        verbose=self.verbose).values
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

def get_antstpara(tfile, center=True):
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
                    imat['tmat'] = get_antstmats(imat, ndim=dim, center=center)
                    tmats.append(imat)

        elif tfile.endswith('.mat'):
            import scipy.io as sio
            infor = sio.loadmat(tfile)
            ttype = [i for i in infor.keys() if 'form_float_' in i ][0]
            tmats = { 'fix_para': infor['fixed'].flatten(),
                        'tinfo': infor[ttype].flatten(), 
                        'ttype': ttype}
            tmats['tmat'] = get_antstmats(tmats, center=center)

        elif tfile.endswith('.nii.gz'):
            import scipy.io as sio
            infor = ants.image_read(tfile)
            fix_para = np.hstack([infor.shape, infor.origin, infor.spacing, infor.direction.flatten()])
            dim = infor.dimension
            ttype = f'DisplacementFieldTransform_float_{dim}_{dim}'
            tmats = {'fix_para': fix_para,
                        'tinfo': infor.numpy(),
                        'ttype': ttype}
            tmats['tmat'] = get_antstmats(tmats, center=center)
        return tmats

    elif isinstance(tfile, dict):
        tmats = {}
        for wdtf in ['fwdtransforms', 'invtransforms']:
            if wdtf in tfile:
                ifle = tfile[wdtf]
                if isinstance(ifle, list):
                    tmats[wdtf] = [ get_antstpara(i) for i in ifle ]

                elif isinstance(ifle, str):
                    tmats[wdtf] = get_antstpara(ifle)
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

def get_antstmats(tinfor, ndim=None, center=True):
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
        itmat[:ndim,:ndim] = tmatx[:(ndim*ndim)].reshape(ndim,ndim, order='F')
        itmat[:ndim, ndim] = tmatx[(ndim*ndim):][::-1]

        iS = np.eye(ldim)
        iS[:ndim, ndim] = fix_para
        if center:
            itmat = iS  @ itmat @  np.linalg.inv(iS)
        return itmat