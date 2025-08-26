import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp

def pwccd(*args, transformer='affine', callback=None, **kwargs):
    from .pairwise_rigid_registration import pwRigidRegistration
    from .pairwise_affine_registration import pwAffineRegistration
    from .pairwise_similarity_registration import pwSimilarityRegistration
    from .pairwise_projective_registration import pwProjectiveRegistration
    from .pairwise_deformable_registration import pwDeformableRegistration
    from .pairwise_constrained_deformable_registration import pwConstrainedDeformableRegistration
    
    if transformer in ['E', 'rigid', 'euclidean']:
        kwargs.update({'fix_s': True})
        model = pwRigidRegistration(*args, **kwargs)

    elif transformer in ['R', 'rotation']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=False,
                            fix_t=True,
                            fix_s=True,))
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['T', 'translation']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=True,))
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['I', 'isoscaletranslation']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=False,))
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['L', 'scaletranslation']:
        kwargs.update(dict(isoscale=False, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=False,))
        model = pwSimilarityRegistration(*args, **kwargs)

    elif transformer in ['O', 'isosimilarity']:
        kwargs.update(dict(isoscale=True, 
                            fix_R=False,
                            fix_t=False,
                            fix_s=False,))
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['S', 'similarity']:
        # kwargs.update(dict(isoscale=False, 
        #                     fix_R=False,
        #                     fix_t=False,
        #                     fix_s=False,))
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['A', 'affine']:
        model = pwAffineRegistration(*args, **kwargs)
    elif transformer in ['P', 'projective']:
        model = pwProjectiveRegistration(*args, **kwargs)
    elif transformer in ['D', 'deformable']:
        if kwargs.get('pairs', None) is None:
            model = pwDeformableRegistration(*args, **kwargs)
        else:
            model = pwConstrainedDeformableRegistration(*args, **kwargs)
    else:
        raise ValueError('transformer must be one of rigid, affine, isosimilarity, similarity,'
                         'projectve, deformable, constraineddeformable')
    model.register( callback=callback )
    return model

def gwccd(*args, transformer='affine', callback=None, **kwargs):
    from .groupwise_rigid_registration import gwRigidRegistration
    from .groupwise_affine_registration import gwAffineRegistration
    from .groupwise_similarity_registration import gwSimilarityRegistration
    from .groupwise_deformable_registration import gwDeformableRegistration
    from .groupwise_complex_registration import gwComplexRegistration

    if type(transformer) in [list, tuple]:
        if len(transformer) == 1:
            transformer = transformer[0]
        elif len(transformer) > 1:
            transformer = ''.join(transformer)
    if type(transformer) is str:
        if len(set(transformer)) == 1:
            transformer = transformer[0]
    
    if transformer in ['E', 'rigid', 'euclidean', 'gweuclidean']:
        model = gwRigidRegistration(*args, **kwargs)
    elif transformer in ['S', 'similarity']:
        model = gwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['A', 'affine']:
        model = gwAffineRegistration(*args, **kwargs)
    elif transformer in ['D', 'deformable']:
        model = gwDeformableRegistration(*args, **kwargs)
    elif all([i in 'ESADRTILO ' for i in transformer ]):
        model = gwComplexRegistration(*args, **kwargs)
    else:
        raise ValueError('transformer must be one of ESADRTILO')
    model.register(callback=callback)
    return model

def rfccd(*args, callback=None, **kwargs):
    from .reference_emregistration import rfEMRegistration
    model = rfEMRegistration(*args, **kwargs)
    model.register(callback=callback)
    return model

def ccd_zl(*args, transformer='affine-z',  callback=None, **kwargs):
    from .zshift_affine_registration import zsAffineRegistration
    model = zsAffineRegistration(*args, transformer=transformer, **kwargs)
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