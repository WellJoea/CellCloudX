import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp

def ccd(*args, transformer='affine', **kwargs):
    from .rigid_registration import RigidRegistration
    from .affine_registration import AffineRegistration
    from .deformable_registration import DeformableRegistration
    from .constrained_deformable_registration import ConstrainedDeformableRegistration
    if transformer in ['rigid', 'euclidean']:
        kwargs.update({'scale': False})
        return RigidRegistration(*args, **kwargs)
    elif transformer in ['similarity']:
        kwargs.update({'scale': True})
        return RigidRegistration(*args, **kwargs)
    elif transformer in ['affine']:
        return AffineRegistration(*args, **kwargs)
    elif transformer in ['deformable']:
        return DeformableRegistration(*args, **kwargs)
    elif transformer in ['constraineddeformable']:
        return ConstrainedDeformableRegistration(*args, **kwargs)

def lccd(X, Y, transformer='affine',
             source_id=None, target_id=None,
             callback=None, **kwargs):
    from .lccd_registration import rigid_reg, affine_reg, deformable_reg
    TRANS = {
        'rigid': rigid_reg,
        'euclidean': rigid_reg,
        'similarity': rigid_reg,
        'affine': affine_reg,
        'deformable': deformable_reg,
        'constraineddeformable': deformable_reg,
    }

    fargs = {}
    if transformer in ['rigid', 'euclidean']:
        fargs.update({'scale': False})
    elif transformer in ['similarity']:
        fargs.update({'scale': True})
    elif transformer in ['deformable']:
        if (not source_id is None) and (not target_id is None):
            transformer = 'constraineddeformable'
    elif transformer in ['constraineddeformable']:
        assert (not source_id is None) and (not target_id is None)

    kwargs.update(fargs)
    model = TRANS[transformer](X, Y, **kwargs)
    model.register(callback)
    return model