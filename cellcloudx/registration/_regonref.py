import numpy as np
import json
from skimage.metrics import structural_similarity
from joblib import Parallel, delayed

from ._feature_reg import feature_regist
from ..registration._antsreg import antsreg
from ..registration._itkreg import itkregist
from ..transform._transi import homotransform

def img_similarity(img1, img2, max_p= 4095 ):
    assert img1.shape == img2.shape
    return structural_similarity(img1, 
                                 img2, 
                                 data_range=max_p, 
                                 channel_axis=(img1.ndim-1))

def homoregist(fixed_image,  moving_image, 
                  verify_ratio=0.85,
                  CI = 0.95,
                  drawmatch=False,
                  drawfeature=False,
                  edgeThreshold = 50, # 100
                  nfeatures= None,
                  sigma = 1.8,
                  nOctaveLayers = 18, 
                  contrastThreshold = 0.03,
                  threads = 50,
                  inverse = False,
                  residual_trials = 20, 
                  stop_merror= 1e-3,
                  f_method = 'sift',
                  match_method ='knn',
                  method=['ski', 'itk'], 
                  transcolor= None,
                  verbose = 1,
                  pass_error=False,
                  Optimizer='AdaptiveStochasticGradientDescent',
                  MaximumStepLength=10,
                  MaximumNumberOfIterations=5000,
                  MinimumStepLength=None,
                  NumberOfResolutions=5,
                  resolutions=20, GridSpacing=15,
                  NumberOfSpatialSamples=4000,
                  transtype=['rigid', 'affine'],
                  order=3,
                  antsargs = {},):
    #moving_image = match_histograms(moving_image, fixed_image, channel_axis=None)
    itmat = np.eye(3)
    imov = moving_image
    movS = []
    for imeth, itrans in zip(method, transtype):
        if imeth == 'ski':
            sargs = dict(
                drawmatch=drawmatch,
                feature_args = {'method':f_method,
                                'drawfeature':drawfeature, 
                                'nfeatures':nfeatures,
                                'contrastThreshold':contrastThreshold,
                                'nOctaveLayers':nOctaveLayers,
                                'edgeThreshold':edgeThreshold,
                                'sigma':sigma},
                CI = CI,
                match_args= {'verbose':0, 'verify_ratio': verify_ratio, 'method':match_method,},
                ransac_args={'verbose':0, 'residual_trials': residual_trials, 'stop_merror': stop_merror},
                verbose=verbose,
                transcolor=transcolor
            )
            rgski = feature_regist(transtype=itrans)
            if pass_error:
                try:
                    rgski.regist_transform(fixed_image, moving_image, **sargs)
                except:
                    print('ski pass')
            else:
                rgski.regist_transform(fixed_image, moving_image, **sargs)

            if hasattr(rgski, 'tmat'):
                itmat = itmat @ rgski.tmat
                imov = rgski.mov_out
                movS.append(imov)

        if imeth == 'ants':
            rgant = antsreg(transtype=itrans)
            rgant.regist_transform(fixed_image, imov, **antsargs)
            itmat = itmat @ rgant.tmats[0]
            imov = rgant.mov_out.numpy()
            movS.append(imov)

        if imeth == 'itk':
            rgitk = itkregist(transtype = itrans, resolutions=resolutions, GridSpacing=GridSpacing)
            # rgitk.params.SetParameter(0, "Optimizer", "RegularStepGradientDescent")
            # rgitk.params.SetParameter(0, "Optimizer", "AdaptiveStochasticGradientDescent")
            rgitk.params.SetParameter(0, "Optimizer", Optimizer)
            rgitk.params.SetParameter(0,  "NumberOfResolutions", str(NumberOfResolutions))
            rgitk.params.SetParameter(0, "MaximumStepLength", str(MaximumStepLength))
            rgitk.params.SetParameter(0, "NumberOfSpatialSamples", str(NumberOfSpatialSamples))
            rgitk.params.SetParameter(0, "MaximumNumberOfIterations", str(MaximumNumberOfIterations))
            if not MinimumStepLength is None:
                rgitk.params.SetParameter(0, "MinimumStepLength", str(MinimumStepLength))
            # rgitk.params.SetParameter(0, "NumberOfHistogramBins", ['16', '32', '64'])

            # rgitk.params.SetParameter(0, "NumberOfResolutions", "2")
            # rgitk.params.SetParameter(0, "ImagePyramidSchedule", "8 8 4 4 2 2 1 1")
            # rgitk.params.SetParameter(itr, "NumberOfHistogramBins", "16 32 64")

            if pass_error:
                try:
                    rgitk.regist_transform( fixed_image, imov, log_to_console=False,  number_of_threads=threads)
                except:
                    print('itk pass')
            else:
                rgitk.regist_transform( fixed_image, imov, log_to_console=False,  number_of_threads=threads)
            if hasattr(rgitk, 'tmat'):
                itmat = itmat @ rgitk.tmat
                imov = rgitk.mov_out
                movS.append(imov)

    if inverse:
        mov_out = homotransform(fixed_image, itmat, order =order, inverse=True) # inverse
        itmat = np.linalg.inv(itmat)
    else:
        mov_out = homotransform(moving_image, itmat, order =order, inverse=False) # inverse
    return itmat, mov_out #, rgitk.mov_out, mov_out2

def regpair(ifix, imov, method=['ski', 'itk'], 
                  transtype=['rigid', 'affine'], 
                  inverse=False, **kargs):
    if inverse:
        itmat, imov_out = homoregist(imov, ifix, method=method, transtype=transtype,  inverse=inverse, **kargs) # inverse
    else:
        itmat, imov_out = homoregist(ifix, imov, method=method, transtype=transtype, inverse=inverse, **kargs) # inverse
    isimi = img_similarity(ifix, imov_out)
    return [itmat, isimi, imov_out]

def split_window(flen, mlen, windows=30):
    start = np.linspace(0, flen-windows, mlen).astype(int)
    end = start + windows
    return np.c_[start, end]

def regslice(Fixings, Movings, idxm, windows = 30, top = None, strange=None, 
                 method=['ski', 'itk'], transtype=['rigist', 'affine'],
                 inverse=True,
                **kargs):

    flen = Fixings.shape[0]
    mlen = Movings.shape[0]
    stranges = split_window(flen, mlen, windows=windows)

    if strange is None:
        idxf = np.arange(*stranges[idxm])
    else:
        idxf = np.arange(*strange)

    top = len(idxf) if top is None else top

    print(f'{idxm} -> {idxf}')
    rfixs = Fixings[idxf]
    imov = Movings[idxm]

    results = Parallel( n_jobs= len(idxf)+1, backend='threading', verbose=5)\
                        (delayed(regpair)(ifix, imov, method=method, transtype=transtype, inverse=inverse, **kargs) for ifix in rfixs)
    simis = [ i[1] for i in results]

    top_idx  = np.argsort(-np.array(simis))[:top]
    tmat_n = [ results[itd][0] for itd in top_idx ]
    simi_n = [ results[itd][1] for itd in top_idx ]
    mov_n = [ results[itd][2] for itd in top_idx ]
    return [idxm, idxf[top_idx], tmat_n, simi_n, mov_n]