import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp


from .rigid_registration import RigidRegistration
from .affine_registration import AffineRegistration
from .deformable_registration import DeformableRegistration
from .constrained_deformable_registration import ConstrainedDeformableRegistration
from ...tools._search import searchidx
from ...utilis._arrays import isidentity, list_iter
from ...transform import homotransform_point, homotransform_points, homoreshape

class ccd:
    TRANS = {
        'rigid':RigidRegistration, 
        'euclidean':RigidRegistration,
        'similarity':RigidRegistration, 
        'affine':AffineRegistration, 
        'deformable':DeformableRegistration,
        'constraineddeformable':ConstrainedDeformableRegistration,
    }
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                  transformer='affine', source_id = None, target_id= None, **kwargs):
        self.X = np.array(X, dtype=np.float64).copy()
        self.Y = np.array(Y, dtype=np.float64).copy()

        self.X_feat = np.array(X_feat) if not X_feat is None else None
        self.Y_feat = np.array(Y_feat) if not Y_feat is None else None
        self.TY = np.array(Y, dtype=np.float64).copy()
        self.transformer=transformer

        self.source_id = source_id
        self.target_id = target_id
        self.init_params(**kwargs)
    
    def init_params(self, **kwargs):
        fargs = {}
        if self.transformer in ['rigid', 'euclidean']:
            fargs.update({'scale': False})
        elif self.transformer in ['similarity']:
            fargs.update({'scale': True})
        elif self.transformer in ['deformable']:
            if (not self.source_id is None) and (not self.target_id is None):
                self.transformer = 'constraineddeformable'
        elif self.transformer in ['constraineddeformable']:
            assert (not self.source_id is None) and (not self.target_id is None)
        kwargs.update(fargs)
        model = ccd.TRANS[self.transformer](self.X, self.Y, 
                                            X_feat=self.X_feat, 
                                            Y_feat=self.Y_feat, 
                                            **kwargs)
        self.model = model

    def regist( self, callback=None, **kwargs):
        self.model.register(callback=callback, **kwargs)
        self.TY = self.model.TY
        self.tform = self.model.tform
        self.P = self.model.P
        self.C = self.model.C
        if hasattr(self.model, 'tforminv'):
            self.tforminv = self.model.tforminv
        self.transform_point = self.model.transform_point


