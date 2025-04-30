import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp

def ccd_zl(*args, transformer='affine-z',  callback=None, **kwargs):
    from .linear_registration_zaxis import zlinear_registration
    model = zlinear_registration(*args, transformer=transformer, **kwargs)
    model.register( callback=callback )
    return model

def ccd(*args, transformer='affine', callback=None, **kwargs):
    from .rigid_registration import RigidRegistration
    from .affine_registration import AffineRegistration
    from .similarity_registration import SimilarityRegistration
    from .projective_registration import ProjectiveRegistration
    from .deformable_registration import DeformableRegistration
    from .constrained_deformable_registration import ConstrainedDeformableRegistration
    
    if transformer in ['rigid', 'euclidean']:
        kwargs.update({'fix_s': True})
        model = RigidRegistration(*args, **kwargs)

    elif transformer in ['rotation']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=False,
                            fix_t=True,
                            fix_s=True,))
        model = SimilarityRegistration(*args, **kwargs)
    elif transformer in ['translation']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=True,))
        model = SimilarityRegistration(*args, **kwargs)
    elif transformer in ['isoscaletranslation']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=False,))
        model = SimilarityRegistration(*args, **kwargs)
    elif transformer in ['scaletranslation']:
        kwargs.update(dict(isoscale=False, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=False,))
        model = SimilarityRegistration(*args, **kwargs)

    elif transformer in ['isosimilarity']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=False,
                            fix_t=False,
                            fix_s=False,))
        model = SimilarityRegistration(*args, **kwargs)
    elif transformer in ['similarity']:
        kwargs.update(dict(isoscale=False, 
                            fix_R=False,
                            fix_t=False,
                            fix_s=False,))
        model = SimilarityRegistration(*args, **kwargs)
    elif transformer in ['affine']:
        model = AffineRegistration(*args, **kwargs)
    elif transformer in ['projective']:
        model = ProjectiveRegistration(*args, **kwargs)
    elif transformer in ['deformable']:
        model = DeformableRegistration(*args, **kwargs)
    elif transformer in ['constraineddeformable']:
        model = ConstrainedDeformableRegistration(*args, **kwargs)
    else:
        raise ValueError('transformer must be one of rigid, affine, isosimilarity, similarity,'
                         'projectve, deformable, constraineddeformable')
    model.register( callback=callback )
    return model

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