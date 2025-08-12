import numpy as np
import skimage.transform as skitf
import torch as th
import scipy as sci
import matplotlib.pyplot as plt
from time import sleep

from .operation_th import thopt
from .shared_wnn import swnn
from ...plotting._imageview import drawMatches
from ...transform import homotransform_point
from ...io._logger import logger

class pwAnsac(thopt):
    '''
    aggregated neighbours sample consensus 
    '''
    TRANS = {
        'rigid':skitf.EuclideanTransform,
        'euclidean':skitf.EuclideanTransform,
        'E':skitf.EuclideanTransform,
        'similarity':skitf.SimilarityTransform,
        'S':skitf.SimilarityTransform,
        'affine':skitf.AffineTransform, #6
        'A':skitf.AffineTransform, #6
        'projective':skitf.ProjectiveTransform, # 8
        'P':skitf.ProjectiveTransform, # 8
        'homography':skitf.ProjectiveTransform,
        'piecewise-affine':skitf.PiecewiseAffineTransform,
        'fundamental': skitf.FundamentalMatrixTransform,
        'essential': skitf.EssentialMatrixTransform,
        'polynomial': skitf.PolynomialTransform,
    }
    def __init__(self, X, Y, X_feat=None, Y_feat=None, device=None, 
                 transformer = 'E',
                 floatx='float32', floatxx='float64', verbose=2, 
                 seed=491001,
                 **kwargs):
        '''
        Xs = [ th.rand(10000,3), th.rand(20000,3) ]
        XFs = [[th.rand(10000,30), th.rand(10000,50)], [th.rand(20000,35)], ]
        Y = th.rand(8000, 3)
        YFs = [[th.rand(8000,30), th.rand(8000,50)], None, ]
        '''

        super().__init__(device=device, 
                          floatx=floatx, floatxx=floatxx, seed=seed)

        self.init_points(X, Y, X_feat, Y_feat)
        self.TY = self.Y.clone()
        self.D = self.Y.shape[1]

        self.transmodel = self.TRANS[transformer] if transformer in self.TRANS else transformer
        self.verbose = verbose

    def register(self, 
        pairs = None,
        mscore = None,

        kd_method='keops', sp_method = 'sknn',
        mnn=10, snn=30, fnn=50,
        scale_locs=True, scale_feats=True, use_soft=False,
        lower = 0.01, upper=0.995,
        min_score= 0.3,
        max_pairs=5e4,
        min_pairs=500,
        swnn_version=1,

        use_weight=True,
        stop_pairs=300,
        stop_merror=1e-1,stop_derror=1e-1, stop_hits=2,
        min_samples=40, residual_threshold=1., 
        maxiter=None, max_trials=500, CI = 0.95,
        seed = 200504,
        ):

        maxiter = maxiter if maxiter is not None else 50

        if pairs is None:
            self.verbose>0 and logger.info('Compute sswnn...')
            pairs, mscore = [], []
            for iL in range(self.L):
                ipair, imscore = swnn(
                    self.Xs[iL], self.Y, 
                    self.XF[iL], self.YF[iL],
                    kd_method=kd_method, sp_method = sp_method,
                    mnn=mnn, snn=snn, fnn=fnn,
                    scale_locs=scale_locs, scale_feats=scale_feats,
                    use_soft=use_soft,
                    lower = lower, upper=upper,
                    min_score=min_score,
                    max_pairs=max_pairs,
                    min_pairs=min_pairs,
                    swnn_version=swnn_version,
                    device=self.device, dtype=self.floatx, verbose=self.verbose)
                pairs.append(ipair.detach().cpu())
                mscore.append(imscore.detach().cpu())

        assert len(pairs) == self.L

        pos_weight, n_split, rpos, qpos = [], [], [], []
        for iL in range(self.L):
            ipair = pairs[iL]
            if mscore is not None:
                imscore = mscore[iL]
            else:
                imscore = th.ones(ipair.shape[0])
    
            irpos = self.Xs[iL][ipair[:, 0]]
            iqpos = self.Y[ipair[:, 1]]

            n_split.append(ipair.shape[0])
            pos_weight.append(imscore)
            rpos.append(irpos)
            qpos.append(iqpos)

        pairs = th.cat(pairs, dim=0).detach().cpu().to(self.floatxx).numpy()
        pos_weight = th.cat(pos_weight, dim=0).detach().cpu().to(self.floatxx).numpy()
        rpos = th.cat(rpos, dim=0).detach().cpu().to(self.floatxx).numpy()
        qpos = th.cat(qpos, dim=0).detach().cpu().to(self.floatxx).numpy()

        if pairs.shape[0] >= (self.L * 1e5):
            self.verbose >1 and \
                logger.warning('The number of max match pairs is too large. '
                       'Please specify `max_pairs` to speed up adapt ransac.')

        self.verbose >0 and logger.info(f'Evaluate Transformation: {pairs.shape[0]} pairs...')

        inliers, model = adapt_ransac(rpos, qpos, self.transmodel,
                                min_samples=min_samples,
                                data_weight = (pos_weight if use_weight else None),
                                max_trials=max_trials,
                                CI=CI,
                                maxiter=maxiter,
                                verbose=self.verbose,
                                seed=seed,
                                stop_merror=stop_merror,
                                stop_derror=stop_derror, 
                                stop_hits=stop_hits,
                                stop_pairs=stop_pairs,
                                residual_threshold=residual_threshold)

        model_final = self.transmodel(dimensionality = self.D )
        model_final.estimate(qpos[inliers], rpos[inliers])

        self.inliers = inliers
        self.pairs = pairs
        self.pairs_score = pos_weight
        self.n_split = n_split
        self.tform = np.float64(model_final)
        self.tforminv = np.linalg.inv(self.tform)

        if self.verbose >0:
            logger.info(f'Finish ansac registration: {len(inliers)} pairs.')

        self.detach_to_cpu()
        self.TY = self.transform_point(inverse=False)
        return self

    def init_points(self, X, Y, X_feat, Y_feat):
        self.Y = self.to_tensor(Y, device='cpu', dtype=self.floatxx)
        Xa = self.check_tolist(X)
        L = len(Xa)

        Xs , XF, YF = [], [], []
        if not (X_feat is None) and not (Y_feat is None):
            if not isinstance(X_feat, (list, tuple)):
                X_feat = [X_feat]
            if not isinstance(Y_feat, (list, tuple)):
                Y_feat = [Y_feat]

            assert len(X_feat) == L, "X_feat must have the same length with X"
            assert len(Y_feat) == L, "Y_feat must have the same length with Y"

            for iL in range(L):
                XFl = self.check_feature(X_feat[iL], Xa[iL].shape[0])
                YFl = self.check_feature(Y_feat[iL], self.Y.shape[0])
                if XFl is None or YFl is None:
                    fexist = False
                else:
                    assert len(XFl) == len(YFl), "Features must have the same length"
                    fexist = True
                    FL = len(XFl)

                    for il in range(FL):
                        assert XFl[il].shape[1] == YFl[il].shape[1], "Features must have the same dimension"
                        Xs.append(Xa[iL])
                        XF.append(XFl[il])
                        YF.append(YFl[il])
        else:
            Xs = Xa
        self.L = len(Xs)
        self.Xs = Xs
        self.XF = XF
        self.YF = YF

    def check_feature(self, Fs, N):
        if Fs is None or ( isinstance(Fs, (list, tuple)) and  len(Fs) == 0 ) or (Fs.shape[0]==0):
            Fa =  None
        elif isinstance(Fs, (list, tuple)):
            FL = len(Fs)
            Fa = []
            for l in range(FL):
                # assert not Fs[l] is None
                assert N == len(Fs[l]), "Features must have the same points number with X,Y"
                iF = self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone()
                Fa.append(iF)
        else:
            Fa = [self.to_tensor(Fs, dtype=self.floatxx, device='cpu').clone()]
        return Fa

    def check_tolist(self, X):
        if X is None:
            return None

        if not isinstance(X, (list, tuple)):
            X = [X]
        Xs = []
        for iX in X:
            try:
                iX = self.to_tensor(iX, dtype=self.floatxx, device='cpu')
            except:
                raise TypeError('X must be tensor or array or list of tensors or arrays')
            Xs.append(iX)
        return Xs
    
    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['X', 'Y', 'TY', 'X_feat', 'Y_feat', 'inliers', 'pairs', 'pairs_score', 'tform', 'tforminv']
        for attr in attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                try:
                    value = value.detach().cpu().numpy() if to_numpy else value.detach().cpu()
                except:
                    pass
                setattr(self, attr, value)
        
    def draw_match2D(self,  pairs=None, score=None, inliers=None, 
                    line_sample=None,
                    line_width=0.5, line_alpha=0.5,
                    point_size=1,
                    fsize=5,
                    line_color = (None, None), 
                    equal_aspect=True,
                    **kargs,
                    ):
        if pairs is None:
            pairs = self.pairs
            score = self.pairs_score
            inliers = self.inliers

        mridx = pairs[:, 0]
        mqidx = pairs[:, 1]
    
        src_pts0 = self.X[mridx][:,:2]
        dst_pts0 = self.TY[mqidx][:,:2]

        if inliers is not None:
            src_pts = src_pts0[inliers]
            dst_pts = dst_pts0[inliers]
            if score is None:
                line_widths= None
            else:
                line_widths = [score, score[inliers]]
            pair_name = ['fixed(before)', 'moving(before)',
                         'fixed(after)', 'moving(after)', ]
            drawMatches( (src_pts0, dst_pts0, src_pts, dst_pts), 
                            bgs =(self.X, self.TY, self.X, self.TY),
                            line_color = line_color, ncols=2,
                            pairidx=[(0,1), (2,3)], fsize=fsize,
                            titles= pair_name,
                            size=point_size,
                            equal_aspect = equal_aspect,
                            line_sample=line_sample,
                            line_alpha=line_alpha,
                            line_width=line_width,
                            line_widths=line_widths,
                            **kargs)
        else:
            if score is None:
                line_widths= None
            else:
                line_widths = [score]
            pair_name = ['fixed', 'moving', ]
            drawMatches( (src_pts0, dst_pts0,), 
                            bgs =(self.X, self.TY),
                            line_color = line_color, ncols=2,
                            pairidx=[(0,1)], fsize=fsize,
                            titles= pair_name,
                            size=point_size,
                            equal_aspect = equal_aspect,
                            line_sample=line_sample,
                            line_alpha=line_alpha,
                            line_width=line_width,
                            line_widths=line_widths,
                            **kargs)

    def transform_point(self, Y=None, tform=None, inverse=False,):
        Y = self.Y if Y is None else Y
        tform = self.tform if tform is None else tform

        if tform.shape[0]  == tform.shape[1] == (Y.shape[1]+1):
            return homotransform_point(Y, tform, inverse=inverse)
        else:
            raise ValueError('tform shape is not correct')
    
    def get_transformer(self):
        return { 'tform': self.tform, 'pairs': self.pairs, 'pairs_score': self.pairs_score, 'inliers': self.inliers}

def adapt_ransac(src_pts, dst_pts, model_class,
                data_weight=None,
                min_samples=50, residual_threshold =1., 
                maxiter=50, max_trials=500,
                seed=200504, CI=0.95, 
                stop_merror=1e-3,stop_derror=1e-3, stop_hits=2,
                init_ratio = 0.75,
                stop_pairs=300, kernel='norm',
                drawhist=False, verbose=1, **kargs):
    src_ptw = src_pts.copy()
    dst_ptw = dst_pts.copy()
    D = src_pts.shape[1]
    maxiter = 50 if maxiter is None else maxiter
    if not residual_threshold is None:
        assert 0<residual_threshold <=1

    model_record = np.eye(D+1)
    Inliers = np.arange(len(src_pts))
    multsig = sci.stats.norm.ppf((1+CI)/2 , 0, 1)
    dist = np.linalg.norm(src_pts - dst_pts, axis=1)
    threshold = np.quantile(dist, init_ratio)

    distmean = dist.mean()
    # threshold = Invervals(dist, CI=CI, kernel=kernel, tailed ='two')[2]
    # threshold *= residual_threshold
    stop_counts = 0

    if verbose >1:
        print('max_iter, num_trials, pairs, threshold, model_error, mes_error')
    for i in range(maxiter):
        model, inbool, num_trials = ransac(
                (src_pts, dst_pts),
                model_class, 
                threshold,
                min_samples=min_samples,
                max_trials=max_trials,
                data_weight=data_weight,
                rng=seed,
                **kargs )

        n_inliers = np.sum(inbool)
        if (n_inliers <= stop_pairs):
            break
        residuals = model.residuals(src_pts, dst_pts)
        Inliers = Inliers[inbool]
        model_final = model_class(dimensionality=D)
        model_final.estimate(src_ptw[Inliers], dst_ptw[Inliers])
        dst_ptf = homotransform_point( dst_ptw, model_final, inverse=True)
        distNmean = (np.linalg.norm(src_ptw[Inliers] - dst_ptf[Inliers], axis=1)).mean()
        # sigma2 = np.sum(np.square(src_ptw[Inliers] - dst_ptf[Inliers]))/(D * len(Inliers))
        # disterror = (distmean - distNmean)/distmean
        disterror = np.abs(distmean - distNmean)
        merror = np.abs(np.array(model)-np.array(model_record)).mean()
        
        if verbose >1:
            print(f'{i :<3}, {num_trials :<3}, {len(inbool)} -> {n_inliers}, {threshold :.4e}, '
                  f'{merror :.4e}, {disterror :.4e} ')

        if  (merror <= stop_merror) or ( np.abs(disterror) <=stop_derror):
            stop_counts += 1
            if (stop_counts >=stop_hits):
                break

        model_record = model
        distmean = distNmean

        norm_thred = sci.stats.norm.ppf(CI, np.mean(residuals), np.std(residuals))
        sigma_thred = np.mean(residuals) + multsig*np.std(residuals)
        threshold = np.mean([norm_thred, sigma_thred]) *residual_threshold

        # threshold = Invervals(residuals, CI=CI, kernel=kernel, tailed ='two')[2]
        # threshold *= residual_threshold

        if drawhist:
            fig, ax=plt.subplots(1,1, figsize=(5,5))
            ax.hist(residuals, bins=50)
            ax.axvline(norm_thred, color='red')
            ax.axvline(sigma_thred, color='blue')
            ax.axvline(threshold, color='black')
            plt.show()

        # assert n_inliers >= stop_pairs, f'nn pairs is lower than {stop_pairs}.'
        dst_ptn = homotransform_point( dst_pts, model, inverse=True) #model(src_pts)
        src_pts = src_pts[inbool]
        dst_pts = dst_ptn[inbool]

        if not data_weight is None:
            data_weight = data_weight[inbool]

    verbose>2 and print(f'Final match pairs: {len(Inliers)}')
    return Inliers, model_final

'''adapted from skimage ransac'''
def ransac(
    data,
    model_class,
    residual_threshold,
    min_samples=20,
    data_weight=None,
    is_data_valid=None,
    is_model_valid=None,
    max_trials=200,
    stop_ratio=0.95,
    stop_mse=1e-3,
    stop_probability=0.9999,
    rng=None,
    initial_inliers=None,
):

    best_inlier_num = 0
    best_inlier_mse = np.inf
    best_inliers = []
    validate_model = is_model_valid is not None
    validate_data = is_data_valid is not None

    D = data[0].shape[1]
    rng = np.random.default_rng(rng)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data,)
    num_samples = len(data[0])

    if not (0 < min_samples <= num_samples):
        raise ValueError(f"`min_samples` must be in range (0, {num_samples}]")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError(
            f"RANSAC received a vector of initial inliers (length "
            f"{len(initial_inliers)}) that didn't match the number of "
            f"samples ({num_samples}). The vector of initial inliers should "
            f"have the same length as the number of samples and contain only "
            f"True (this sample is an initial inlier) and False (this one "
            f"isn't) values."
        )
    if not data_weight is None:
        data_weight = data_weight/data_weight.sum()
    # for the first run use initial guess of inliers
    spl_idxs = (
        initial_inliers
        if initial_inliers is not None
        else rng.choice(num_samples, min_samples, replace=False, p=data_weight)
    )

    # estimate model for current random sample set
    model = model_class(dimensionality=D)

    num_trials = 0
    # max_trials can be updated inside the loop, so this cannot be a for-loop
    while num_trials < max_trials:
        num_trials += 1

        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]

        # for next iteration choose random sample set and be sure that
        # no samples repeat
        spl_idxs = rng.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if validate_data and not is_data_valid(*samples):
            continue

        success = model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if validate_model and not is_model_valid(model, *samples):
            continue

        residuals = np.abs(model.residuals(*data))
        # consensus set / inliers
        inliers = residuals < residual_threshold
        mse = np.mean(residuals)


        # choose as new best model if number of inliers is maximal
        inliers_count = np.count_nonzero(inliers)
        if (
            # more inliers
            inliers_count > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (
                inliers_count == best_inlier_num
                and mse < best_inlier_mse
            )
        ):
            best_inlier_num = inliers_count
            best_inlier_mse = mse
            best_mse = np.mean(residuals[inliers])
            best_inliers = inliers
            max_trials = min(
                max_trials,
                _dynamic_max_trials(
                    best_inlier_num, num_samples, min_samples, stop_probability
                ),
            )
            if (
                best_inlier_num/num_samples >= stop_ratio
                or best_mse <= stop_mse
            ):
                break

    # estimate final model using all inliers
    if any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        if validate_model and not is_model_valid(model, *data_inliers):
            logger.warning("Estimated model is not valid. Try increasing max_trials.")
    else:
        model = None
        best_inliers = None
        logger.warning("No inliers found. Model not fitted")

    return model, best_inliers, num_trials

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability, _EPSILON = np.spacing(1)):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.
    """
    if probability == 0:
        return 0
    if n_inliers == 0:
        return np.inf
    inlier_ratio = n_inliers / n_samples
    nom = max(_EPSILON, 1 - probability)
    # denom = 1- max(_EPSILON, inlier_ratio**(min_samples*probability))
    denom = max(_EPSILON, 1- max(_EPSILON, inlier_ratio**min_samples))
    return np.ceil(np.log(nom) / np.log(denom))