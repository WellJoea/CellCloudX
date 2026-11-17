import numpy as np
import pandas as pd
import collections
import skimage.transform as skitf
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt

from ._sswnn import sswnn_pair
from ....plotting._imageview import drawMatches
from ....transform import homotransform_point, homotransform_points, homoreshape
from ....utilis._arrays import isidentity, list_iter
from .autothreshold_ransac import atransac

class ansac():
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
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                   *args, **kwargs):

        self.X = np.array(X, dtype=np.float64).copy()
        self.Y = np.array(Y, dtype=np.float64).copy()
        
        self.X_feat = X_feat
        self.Y_feat = Y_feat
        self.TY = np.array(Y, dtype=np.float64).copy()
        self.D = self.X.shape[1]

    def register(self, 
        transformer='rigid',
        pairs = None,
        pair_name = None,
        tform = None,

        kd_method='annoy', sp_method = 'sknn',
        use_dpca = False, dpca_npca = 60,
        m_neighbor=10, e_neighbor =30, s_neighbor =30,
        o_neighbor = None,
        score_threshold = 0.35,
        lower = 0.01, upper=0.995,
        max_pairs=1e5,
        min_pairs=500,

        use_weight=True,
        stop_pairs=40,
        stop_merror=1e-3,stop_derror=1e-5, stop_hits=2,
        min_samples=40, residual_threshold=1., 
        maxiter=None, max_trials=700, CI = 0.95,
        verbose=1,

        drawmatch=True,  line_sample=None,
        line_width=0.5, line_alpha=0.5,
        point_size=1,
        fsize=5,
        seed=491001,
        line_color = (None, None), 
        equal_aspect=True,
        pargs={},
        **kargs):

        maxiter = maxiter if maxiter is not None else 50
        transmodel = self.TRANS[transformer] if transformer in self.TRANS else transformer
        if pair_name is None:
            pair_name = ['fixed', 'moving']

        if not tform is None:
            tform = np.float64(tform)
            self.TY = homotransform_point(self.TY, tform, inverse=False)

        if pairs is None:
            pairs, mscore = sswnn_pair(
                self.X, self.TY, self.X_feat, self.Y_feat,
                pair_name = pair_name,
                kd_method=kd_method, sp_method = sp_method,

                use_dpca = use_dpca, dpca_npca = dpca_npca,
                m_neighbor=m_neighbor, e_neighbor =e_neighbor, 
                s_neighbor =s_neighbor, o_neighbor = o_neighbor,
                lower = lower, upper = upper, score_threshold = score_threshold,
                max_pairs=max_pairs, min_pairs=min_pairs,
                drawmatch=False, **kargs)
        else:
            mscore = np.ones(pairs.shape[0])

        if pairs.shape[0] >=1e5:
            print('The number of max match pairs is too large. '
                    'Please specify `max_pairs` to speed up atransac.')
        mridx = pairs[:, 0]
        mqidx = pairs[:, 1]

        rpos = self.X[mridx]
        qpos = self.TY[mqidx]
        print('Evaluate Transformation...')
        inliers, model = atransac(rpos, qpos, transmodel,
                                min_samples=min_samples,
                                data_weight = (mscore if use_weight else None),
                                max_trials=max_trials,
                                CI=CI,
                                maxiter=maxiter,
                                verbose=verbose,
                                seed=seed,
                                stop_merror=stop_merror,
                                stop_derror=stop_derror, 
                                stop_hits=stop_hits,
                                stop_pairs=stop_pairs,
                                residual_threshold=residual_threshold)
        # if tform is None:
        #     self.tform = np.float64(model)
        # else:
        #     self.tform = tform @ np.float64(model)

        model_final = transmodel(dimensionality = self.D )
        model_final.estimate(self.Y[mqidx][inliers], self.X[mridx][inliers])
        self.tform = np.float64(model_final)
        self.tforminv = np.linalg.inv(self.tform)

        keepidx = np.zeros_like(mridx)
        keepidx[inliers] = 1
        # self.anchors = np.vstack([mridx, mqidx, keepidx, mscore]).T
        self.anchors = pd.DataFrame({
            'midx': mridx, 'qidx': mqidx, 'keepidx': keepidx, 'score': mscore
        })
        self.model = model_final

        self.TY = self.transform_point(inverse=False)
        if drawmatch and (self.D==2):
            src_pts = rpos[inliers]
            dst_pts = self.TY[mqidx][inliers]
            pair_name = ['fixed(before)', 'moving(before)',
                         'fixed(after)', 'moving(after)', ]
            drawMatches( (rpos, self.TY[mqidx], src_pts, dst_pts), 
                        bgs =(self.X, self.TY, self.X, self.TY),
                        line_color = line_color, ncols=2,
                        pairidx=[(0,1), (2,3)], fsize=fsize,
                        titles= pair_name,
                        size=point_size,
                        equal_aspect = equal_aspect,
                        line_sample=line_sample,
                        line_alpha=line_alpha,
                        line_width=line_width,
                        **pargs)

    def transform_point(self, Y=None, tform=None, inverse=False,):
        Y = self.Y if Y is None else Y
        tform = self.tform if tform is None else tform

        if tform.shape[0]  == tform.shape[1] == (Y.shape[1]+1):
            return homotransform_point(Y, tform, inverse=inverse)
        else:
            raise ValueError('tform shape is not correct')
    
    def get_transformer(self):
        return { 'tform': self.tform, 'model': self.model, 'anchors':self.anchors}