import skimage as ski
import scipy as sci
import skimage.transform as skitf
import numpy as np
import copy

try:
    import cv2
except ImportError:
    pass

from ..plotting._imageview import qview, drawMatches
from ..transform._transp import homotransform_point

class Features(object):
    def __init__(self):
        self.default_args = dict(
            sift = dict(
                        nfeatures = 10000,
                        nOctaveLayers =15,
                        contrastThreshold=0.03, 
                        edgeThreshold=50,
                        sigma =1.6,), 
            surf = dict(hessianThreshold = 400,
                            nOctaves=4,  
                            nOctaveLayers=15, 
                            extended=False, 
                            upright=False),
            orb = dict(nfeatures=10000,
                            scaleFactor=1.2,
                            nlevels=8,
                            edgeThreshold=31,
                            firstLevel=0,
                            WTA_K=2,
                            scoreType=0, 
                            patchSize=31, 
                            fastThreshold=20),
            SIFT = dict(upsampling=2, n_octaves=15, n_scales=5, sigma_min=1.6, 
                            sigma_in=0.5, c_dog=0.013333333333333334, c_edge=50, 
                            n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8),
            ORB = dict(downscale=1.2, 
                            n_scales=8, 
                            n_keypoints=5000, 
                            fast_n=9, fast_threshold=0.08, harris_k=0.04),
            BRIEF = dict( descriptor_size=256, patch_size=49, mode='normal', sigma=1, rng=200504),
            harris = dict(method='k', k=0.05, eps=1e-06, sigma=1),
            peaks = dict(min_distance=1, threshold_rel=0.01, threshold_abs=None, exclude_border=5),
            censure = dict(min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10),
            method='sift',
            kpbinary = 'corner',
            drawfeature=False,
            color=(255,0,0),
            flags=4,
        )
        self._fargs = copy.deepcopy(self.default_args)

    @property
    def fargs(self):
        return self._fargs

    # @fargs.setter
    def set_fargs(self, kwargs):
        if kwargs:
            for k,v in kwargs.items():
                if (k in self._fargs):
                    if (isinstance(v, dict)):
                        self._fargs[k] = self.update_arg(self._fargs[k], v, INPLACE=False)
                    else:
                        self._fargs[k] = v
                else:
                    for rk, rv in self._fargs.items():
                        if (isinstance(rv, dict)):
                            self._fargs[rk] = self.update_arg(rv, {k:v}, INPLACE=False)

    def get_fargs(self, method=None, args={}):
        self.set_fargs(args)
        if method:
            return self.fargs[method]
        else:
            return self.fargs

    @staticmethod
    def update_arg(r_args, u_args, INPLACE=False, **kwargs):
        if INPLACE:
            for k, v in r_args.items():
                r_args[k] = kwargs.get(k, u_args.get(k, v))
        else:
            return { k: kwargs.get(k, u_args.get(k, v)) for k, v in r_args.items() }

    def features(self, image, **kwargs):
        self.set_fargs(kwargs)

        method= self.fargs['method']
        kpbinary = self.fargs['kpbinary']
        drawfeature= self.fargs['drawfeature']
        color=self.fargs['color']
        flags=self.fargs['flags']

        if np.issubdtype(image.dtype, np.floating) and (image.max() <=1):
            image = (image * 255).astype(np.uint8)

        if method == 'sift':
            import cv2
            detector = cv2.SIFT_create(**self.get_fargs('sift', kwargs))
            nomaltype = cv2.NORM_L2
            kp, ds = detector.detectAndCompute(image, None)
        elif method == 'surf':
            import cv2
            detector = cv2.xfeatures2d_SURF.create(**self.get_fargs('surf', kwargs))
            nomaltype = cv2.NORM_L2
            kp, ds = detector.detectAndCompute(image, None)
        elif method == 'orb':
            import cv2
            detector = cv2.ORB_create( **self.get_fargs('orb', kwargs))
            nomaltype = cv2.NORM_HAMMING
            kp, ds = detector.detectAndCompute(image, None)
        elif method == 'SIFT':
            detector = ski.feature.SIFT(**self.get_fargs('SIFT', kwargs))
            detector.detect_and_extract(image)
            # kp = detector.keypoints.astype(np.float64)
            kp = detector.positions.astype(np.float64)
            ds = detector.descriptors.astype(np.float32)
            nomaltype = 4
        elif method == 'ORB':
            detector = ski.feature.ORB(**self.get_fargs('ORB', kwargs))
            detector.detect_and_extract(image)
            kp = detector.keypoints.astype(np.float64)
            ds = detector.descriptors.astype(np.uint8)
            nomaltype = 6
        elif method == 'BRIEF':
            if kpbinary == 'corner':
                harris = ski.feature.corner_harris(image, **self.get_fargs('harris', kwargs))
                kp = ski.feature.corner_peaks(harris, **self.get_fargs('peaks', kwargs))
            elif kpbinary == 'censure':
                censure = ski.feature.CENSURE(**self.get_fargs('censure', kwargs))
                censure.detect(image)
                kp = censure.keypoints
            detector = ski.feature.BRIEF(**self.get_fargs('BRIEF', kwargs))
            detector.extract(image, kp)
            ds = detector.descriptors.astype(np.uint8)
            kp = kp.astype(np.float64)
            nomaltype = 6

        if drawfeature:
            import cv2
            if isinstance(kp[0], cv2.KeyPoint):
                kpcv = kp
            else:
                if hasattr(detector, 'scales'):
                    ssize = detector.scales
                else:
                    ssize = [1]*len(kp)
                kpcv = tuple([cv2.KeyPoint(kp[i][1], kp[i][0], float(ssize[i])) for i in range(len(kp))])
            imgdkp=cv2.drawKeypoints(image,
                                        kpcv,
                                        image,
                                        color=color, 
                                        flags=flags)
            qview(imgdkp)

        if isinstance(kp[0], cv2.KeyPoint):
            kp = np.float64([ list(ikp.pt) for ikp in kp ])
        else:
            kp[:,[0,1]] = kp[:,[1,0]]
        return kp, ds, nomaltype

def matchers(kp1, ds1, kp2, ds2,
            nomaltype=4,
            method='knn',
            min_matches = 8,
            verify_ratio = 0.7,
            # reprojThresh = 5.0,
            feature_method = None,
            table_number = 6, # 12
            key_size = 12, # 20
            multi_probe_level = 1,
            knn_k = 2,
            trees=5,
            checks=50,
            verbose=True,
            **kargs):
    verbose = int(verbose)
    nomaltype = nomaltype or 4
    if method=='cross':
        import cv2
        bf = cv2.BFMatcher(nomaltype, crossCheck=True)
        matches = bf.match(ds1, ds2)
        matches = sorted(matches, key=lambda x:x.distance)

    elif method=='knn':
        import cv2
        bf = cv2.BFMatcher(nomaltype)
        matches = bf.knnMatch(ds1, ds2, k=knn_k)

    elif method=='flann':
        import cv2
        if feature_method in ['orb', 'ORB', 'BRIEF']:
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = table_number, # 12
                                key_size = key_size, # 20
                                multi_probe_level = multi_probe_level) #2
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)

        search_params = dict(checks=checks)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(ds1, ds2,k=2)
    elif method=='mnn':
        matches = ski.feature.match_descriptors(ds1, ds2, metric=None, p=2,
                                                 cross_check=True, max_ratio=verify_ratio)

    if isinstance(kp1[0], cv2.KeyPoint):
        kp1 = np.float64([ ikp.pt for ikp in kp1 ])
    if isinstance(kp2[0], cv2.KeyPoint):
        kp2 = np.float64([ ikp.pt for ikp in kp2 ])

    if method=='mnn':
        fix_pts = kp1[matches[:, 0]].astype(np.float64)
        mov_pts = kp2[matches[:, 1]].astype(np.float64)
        matchesMask = [[1,0]] * len(matches)
        verify_matches = matches
    else:
        if method =='cross':
            matchesMask = [[1,0]] * len(matches)
            verify_matches = matches
        else:
            verify_matches = []
            matchesMask = [[0,0] for i in range(len(matches))]
            for i,(m1,m2) in enumerate(matches):
                if m1.distance < verify_ratio * m2.distance:
                    matchesMask[i]=[1,0]
                    verify_matches.append(m1)

        fix_pts = []
        mov_pts = []
        for m in verify_matches:
            # fix_pts.append(kp1[m.queryIdx].pt)
            # mov_pts.append(kp2[m.trainIdx].pt)
            fix_pts.append(kp1[m.queryIdx])
            mov_pts.append(kp2[m.trainIdx])

        fix_pts = np.array(fix_pts).astype(np.float64)
        mov_pts = np.array(mov_pts).astype(np.float64)

    verbose and print(f'find {len(verify_matches)} matches.')
    assert len(verify_matches)> min_matches, f'low paired matches: {len(verify_matches)}.'
    return [fix_pts, mov_pts]

def autoresidual(src_pts, dst_pts, 
                 model_class=skitf.AffineTransform,
                min_samples=3, residual_threshold =1, 
                residual_trials=50, max_trials=100,
                seed=200504, CI=0.95, stop_merror=1e-3,
                min_pair=10,
                is_data_valid=None, is_model_valid=None,
                 verbose=0, **kargs):

    verbose = int(verbose)
    src_ptw = src_pts.copy()
    dst_ptw = dst_pts.copy()

    if not residual_threshold is None:
        assert 0<residual_threshold <=1
    model_record = np.eye(3)

    n_feature = len(src_pts)
    Inliers = np.arange(n_feature)
    multsig = sci.stats.norm.ppf((1+CI)/2 , 0, 1)
    dist = np.linalg.norm(src_pts - dst_pts, axis=1)
    threshold = np.max(dist)
    stop_counts = 0

    for i in range(residual_trials):
        model, inbool = ski.measure.ransac(
                (src_pts, dst_pts),
                model_class, 
                min_samples=min_samples,
                residual_threshold=threshold, 
                max_trials=max_trials,
                rng=seed,
                is_data_valid=is_data_valid, 
                is_model_valid=is_model_valid, 
                **kargs )

        residuals = model.residuals(src_pts, dst_pts)
        norm_thred = sci.stats.norm.ppf(CI, np.mean(residuals), np.std(residuals))
        sigma_thred = np.mean(residuals) + multsig*np.std(residuals)
        threshold = np.mean([norm_thred, sigma_thred]) *residual_threshold

        n_inliers = np.sum(inbool)
        if verbose>=2:
            print(f'points states: before {len(inbool)} -> after {n_inliers}. {threshold}')

        # assert n_inliers >= min_pair, f'nn pairs is lower than {min_pair}.'
        if n_inliers < min_pair:
            break

        dst_ptn = homotransform_point( dst_pts, model, inverse=False)
        if verbose>=5:
            # drawMatches([src_pts, dst_pts, dst_ptn], show_line=False, ncols=3, sharex=False, sharey=False)
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1,1)
            ax.scatter(src_pts[:,0], src_pts[:,1], marker='o', c='r', s=1, label='src_pts')
            ax.scatter(dst_pts[:,0], dst_pts[:,1], marker='+', c='blue', s=1, label='dst_pts')
            ax.scatter(dst_ptn[:,0], dst_ptn[:,1], marker='*', c='green', s=1, label='dst_ptn')
            ax.legend(markerscale=4)
            plt.show()

        src_pts = src_pts[inbool]
        dst_pts = dst_ptn[inbool]
        Inliers = Inliers[inbool]

        model_new = model_class()
        model_new.estimate(src_ptw[Inliers], dst_ptw[Inliers])
        merror = np.abs((np.array(model_new)-np.array(model_record)).mean())

        if verbose>=3:
            print(f'model error: {merror}')
        model_record = model_new
        if  (merror <= stop_merror):
            stop_counts += 1
            if (stop_counts >=2):
                break

    if verbose:
        print(f'ransac points states: before {n_feature} -> after {len(Inliers)}.')
    model = model_class(dimensionality=src_ptw.shape[1])
    model.estimate(src_ptw[Inliers], dst_ptw[Inliers])
    return [src_ptw[Inliers], dst_ptw[Inliers], Inliers, model]

def gcransac(src_ptw, dst_ptw, model_class):
    #pyransac
    #pydegensac
    #gcransac
    #magsac
    # TODO
    #pygcransac.findRigidTransform 
    pass

def prosac():
    # TODO
    #https://github.com/willGuimont/PROSAC/tree/master
    #https://github.com/nianticlabs/scoring-without-correspondences?tab=readme-ov-file
    pass

def match_locations(img0, img1, coords0, coords1, radius=5, sigma=3):
    """
    Match image locations using SSD minimization (from skimage).

    Areas from `img0` are matched with areas from `img1`. These areas
    are defined as patches located around pixels with Gaussian
    weights.

    Parameters
    ----------
    img0, img1 : 2D array
        Input images.
    coords0 : (2, m) array_like
        Centers of the reference patches in `img0`.
    coords1 : (2, n) array_like
        Centers of the candidate patches in `img1`.
    radius : int
        Radius of the considered patches.
    sigma : float
        Standard deviation of the Gaussian kernel centered over the patches.

    Returns
    -------
    match_coords: (2, m) array
        The points in `coords1` that are the closest corresponding matches to
        those in `coords0` as determined by the (Gaussian weighted) sum of
        squared differences between patches surrounding each point.
    """
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    weights = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    weights /= 2 * np.pi * sigma * sigma

    match_list = []
    for r0, c0 in coords0:
        roi0 = img0[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
        roi1_list = [img1[r1 - radius:r1 + radius + 1,
                          c1 - radius:c1 + radius + 1] for r1, c1 in coords1]
        # sum of squared differences
        ssd_list = [np.sum(weights * (roi0 - roi1) ** 2) for roi1 in roi1_list]
        match_list.append(coords1[np.argmin(ssd_list)])

    return np.array(match_list)

def RansacFilter(src_pts, dst_pts, model_class=skitf.AffineTransform,
                min_samples=3, residual_threshold=100, max_trials=2000,
                is_data_valid=None, is_model_valid=None, use_cv2=False, **kargs
                ):
    model, inliers = ski.measure.ransac(
            (src_pts, dst_pts),
            model_class, min_samples=min_samples,
            residual_threshold=residual_threshold, max_trials=max_trials,
            is_data_valid=is_data_valid, is_model_valid=is_model_valid, 
            **kargs )
    dist = np.linalg.norm(src_pts - dst_pts, axis=1)
    residuals = np.abs(model.residuals(*(src_pts, dst_pts)))
    n_inliers = np.sum(inliers)
    print(f'points states: before {len(inliers)} -> after {n_inliers}.')

    if use_cv2:
        src_pts_inter = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        dst_pts_inter = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        pairs_match = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        #image3 = cv2.drawMatches(img1, src_pts_inter, img2, dst_pts_inter, pairs_match, None)

        src_pts = np.float64([ src_pts_inter[m.queryIdx].pt for m in pairs_match ]).reshape(-1, 2)
        dst_pts = np.float64([ dst_pts_inter[m.trainIdx].pt for m in pairs_match ]).reshape(-1, 2)
    else:
        src_pts = src_pts[inliers]
        dst_pts = dst_pts[inliers]
    return [src_pts, dst_pts, inliers, model]

def _drawMatches(img1, kp1, img2, kp2, 
                    verify_matches,
                    inliers = None,
                    matchesMask = None,
                    matchesThickness=2,
                    drawpairs=2000,
                    matchColor= (0,255,0),
                    singlePointColor = (255,0,0),
                    flags=0,):
    draw_params = dict(#matchesThickness=matchesThickness,
                        matchColor = matchColor,
                        singlePointColor = singlePointColor,
                        #matchesMask = matchesMask,
                        flags = flags)
    verify_matchesf = []
    if not inliers is None:
        verify_matchesf = [ imh for imh, inl in zip(verify_matches, inliers) if inl]
    else:
        verify_matchesf = verify_matches

    drawpairs = min(len(verify_matchesf), drawpairs)
    imgdm = cv2.drawMatches(img1, kp1, img2, kp2, 
                                    tuple(verify_matchesf[:drawpairs]), 
                                None, **draw_params)
    #imgdm = cv2.drawMatches(img1, k1, img2, k2, verify_matches, None, (0,0,255), flags=2)
    qview(imgdm)
    return imgdm
