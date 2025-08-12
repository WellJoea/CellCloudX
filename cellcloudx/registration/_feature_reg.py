import skimage as ski
import skimage.transform as skitf
import numpy as np
try:
    import cv2
except ImportError:
    pass

from ..transform._transp import homotransform_point
from ..transform._transi import homotransform
from ..plotting._imageview import drawMatches
from ._features import Features, matchers, autoresidual

class feature_regist():
    def __init__(self, transtype=None):
        self.transtype = 'rigid' if transtype is None else transtype

        self.TRANS = {
                'rigid':skitf.EuclideanTransform,
                'euclidean':skitf.EuclideanTransform,
                'similarity':skitf.SimilarityTransform,
                'affine':skitf.AffineTransform,
                'projective':skitf.ProjectiveTransform,
                'homography':skitf.ProjectiveTransform,
                'piecewise-affine':skitf.PiecewiseAffineTransform,
                'fundamental': skitf.FundamentalMatrixTransform,
                'essential': skitf.EssentialMatrixTransform,
                'polynomial': skitf.PolynomialTransform,
        }
        self.F = Features()
        self.F.set_fargs({
                        'method': 'sift',
                        'nfeatures':10000,
                        'nOctaveLayers':12,
                        'contrastThreshold':0.03,
                        'edgeThreshold':50,
                        'hessianThreshold': 400,
                        'nOctaves':4, 
                        'sigma' : 1.6,
                        'surf_nOctaveLayers':3,
                        'orb_nfeatures':2000,
                        'scaleFactor':1.2,
                        'nlevels':8,
                        'orb_edgeThreshold':31,
                        'firstLevel':0,
                        'WTA_K':2,
                        #scoreType=0, 
                        'kpbinary': 'corner',
                        'patchSize':31, 
                        'fastThreshold':20,
                        'extended':False, 
                        'upright':False,
                        'drawfeature':True,
                        'color':(255,0,0),
                        'flags':4})

        self.matchers = matchers
        self.match_args = {
                        'method':'knn',
                        'min_matches': 8,
                        'drawpoints':2000,
                        'verify_ratio' : 0.8,
                        'reprojThresh' : 5.0,
                        'feature_method' : None,
                        'table_number' : 6, # 12
                        'key_size' : 12, # 20
                        'knn_k' : 2,
                        'multi_probe_level' : 1,
                        'trees':5,
                        'checks':50,
                        'verbose':1}

        self.autoresidual = autoresidual
        self.ransac_args = {
                    'is_data_valid': None, 
                    'is_model_valid': None, 
                    'stop_sample_num': np.inf,
                    'stop_residuals_sum': 0, 
                    'stop_probability': 1, 
                    'residual_trials': 20,
                    'min_pair':10,
                    'stop_merror': 1e-3,
                    'verbose':0,
                    'initial_inliers': None}

        self.drawMatches = drawMatches
        self.drawm_args = {
                        'line_color': None,
                        'size':5,
                        'line_width' : 0.5,
                        'line_alpha':None, 
                        'hide_axis':False,
                        'cmap': None,
                        'fsize': 7,
                        'werror' : 0, 
                        'herror':0,
                        'sharex':False, 
                        'sharey':False,
                        'line_sample': 2000,
                        'equal_aspect' : True, 
                }

    def regist(self, fixed_img, moving_img,
                    transtype=None,
                    transcolor=None,
                    filter=None,

                    kpds_fix = None, 
                    kpds_mov = None,
                    nomaltype = None,
                    feature_args = {},

                    fixmov_pts = None,
                    match_args = {},

                    use_Ransac = True,
                    min_samples=3, 
                    max_trials=150,
                    CI = 0.95,
                    ransac_args = {},

                    drawmatch =True,
                    drawm_args = {},
                    verbose = 1,
                    **kargs):

        # self.feature_args.update(feature_args)
        transtype =self.transtype if transtype is None else transtype
        tfmethod = self.TRANS[self.transtype]
        # import inspect
        # sig = inspect.signature(self.regist)
        # params = sig.parameters
        # print(params)

        img1 = fixed_img.copy()
        img2 = moving_img.copy()
        if not filter is None:
            img1 = self.filters(img1, filter=filter)
            img2 = self.filters(img2, filter=filter)

        img1_g  = self.colortrans(img1, transcolor=transcolor) if img1.ndim==3 else img1
        img2_g  = self.colortrans(img2, transcolor=transcolor) if img2.ndim==3 else img2

        if kpds_fix is None:
            kp1, ds1, nomaltype = self.F.features(img1_g, **feature_args)
        else:
            kp1, ds1 = kpds_fix

        if kpds_mov is None:
            kp2, ds2, nomaltype = self.F.features(img2_g, **feature_args)
        else:
            kp2, ds2 =kpds_mov
        self.match_args.update(match_args)
        self.match_args['nomaltype'] = nomaltype
        self.match_args['feature_method'] = self.F.fargs['method']

        if fixmov_pts is None:
            fix_pts_raw, mov_pts_raw= self.matchers(kp1, ds1, kp2, ds2, **self.match_args)
        else:
            fix_pts_raw, mov_pts_raw = fixmov_pts

        self.ransac_args.update(ransac_args)
        if use_Ransac:
            fix_pts, mov_pts, inliers, model = self.autoresidual(
                    fix_pts_raw, mov_pts_raw, model_class=tfmethod,
                    CI = CI,
                    min_samples=min_samples,
                    max_trials=max_trials,
                    **self.ransac_args)
        else:
            fix_pts, mov_pts = fix_pts_raw, mov_pts_raw
            inliers = [True] * len(fix_pts)

        verbose and print(f'fix <-> mov {len(mov_pts_raw)}  -> paired {len(fix_pts)}')
        if drawmatch:
            self.drawm_args.update(drawm_args)
            self.drawMatches([fix_pts, mov_pts], bgs=[img1, img2], **self.drawm_args)

        tmat = self.estimate( fix_pts, mov_pts, tfmethod=tfmethod)

        self.fixed_img = fixed_img
        self.moving_img = moving_img
        self.kp1 = kp1
        self.ds1 = ds1
        self.kp2 = kp2
        self.ds2 = ds2
        self.nomaltype = nomaltype

        self.fix_pts_raw = fix_pts_raw
        self.mov_pts_raw = mov_pts_raw
        self.fix_pts = fix_pts
        self.mov_pts = mov_pts
        self.inliers = inliers

        self.tmat = np.float64(tmat)
        return self

    def transform(self, 
                    moving_img=None, 
                    tmat=None, 
                    locs=None,
                    order=None,
                    **kargs
                    ):
        moving_img = self.moving_img if moving_img is None else moving_img
        tmat = self.tmat if tmat is None else tmat

        mov_out = homotransform(moving_img, tmat, inverse=False, order=order, **kargs)
        self.mov_out = mov_out
        self.mov_locs = None
        if locs is not None:
            mov_locs = homotransform_point(locs, tmat, inverse=True, swap_xy=True)
            self.mov_locs = mov_locs
            return mov_out, mov_locs
        else:
            return mov_out

    def regist_transform(self, fixed_img, moving_img, 
                            locs=None, 
                            order=None,
                            targs={},
                            **kargs):
        self.regist(fixed_img, moving_img, **kargs)
        self.transform(moving_img, locs=locs, order=order, **targs)
        # return self
        return [self.mov_out, self.tmat, self.mov_locs]

    def estimate(self, fix_pts, mov_pts, tfmethod=skitf.AffineTransform):
        # tmat = skitf.estimate_transform(transtype,  mov_pts, fix_pts)
        tmat = tfmethod(dimensionality=fix_pts.shape[1])
        tmat.estimate(fix_pts, mov_pts)
        # tmat.estimate(mov_pts, fix_pts)
        return tmat

    @staticmethod
    def filters(image, filter=None):
        if filter is None:
            image = image
        elif filter.lower() in ['blur', 'mean']:
            image = cv2.blur(image, (5,5))
        elif filter.lower() in ['gaussian']:
            image = cv2.GaussianBlur(image,(5,5),0)
        elif filter.lower() in ['median']:
            image = cv2.medianBlur(image, 5)
        elif filter.lower() == 'bilateral':
            image = cv2.bilateralFilter(image, 15,85,85 )
        return image

    @staticmethod
    def colortrans(image, transcolor='rgb2gray', *args, **kargs):
        colorfunc = eval(f'ski.color.{transcolor or "rgb2gray"}')
        image = colorfunc(image, *args, **kargs)
        return image