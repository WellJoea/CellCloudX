import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp

def PWR(*args, transformer='affine', callback=None, **kwargs):
    from .pairwise_rigid_registration import pwRigidRegistration
    from .pairwise_affine_registration import pwAffineRegistration
    from .pairwise_similarity_registration import pwSimilarityRegistration
    from .pairwise_projective_registration import pwProjectiveRegistration
    from .pairwise_deformable_registration import pwDeformableRegistration
    from .pairwise_constrained_deformable_registration import pwConstrainedDeformableRegistration
    from .pairwise_hierarchical_registration import pwHierarchicalRegistration
    from .pairwise_diffeomorphism_registration import pwDiffeoRegistration
    
    if transformer in ['E', 'rigid', 'euclidean']:
        kwargs = {'fix_s': True, **kwargs}
        model = pwRigidRegistration(*args, **kwargs)

    elif transformer in ['R', 'rotation']:
        kwargs = {'isoscale': True, 
                  'fix_R': False, 
                  'fix_t': True,
                  'fix_s': True,
                  **kwargs}
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['T', 'translation']:
        kwargs = {'isoscale': True,
                  'fix_R': True,
                  'fix_t': False,
                  'fix_s': True,
                  **kwargs}
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['I', 'isoscaletranslation']:
        kwargs = {'isoscale': True,
                  'fix_R': True,
                  'fix_t': False,
                  'fix_s': False,
                  **kwargs}
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['L', 'scaletranslation']:
        kwargs = {'isoscale': False,
                  'fix_R': True,
                  'fix_t': False,
                  'fix_s': False,
                  **kwargs}
        model = pwSimilarityRegistration(*args, **kwargs)
    elif transformer in ['O', 'isosimilarity']:
        kwargs = {'isoscale': True,
                  'fix_R': False,
                  'fix_t': False,
                  'fix_s': False,
                  **kwargs}
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
    elif transformer in ['H', 'hierarchical']:
        model = pwHierarchicalRegistration(*args, **kwargs)
    # elif transformer in ['K', 'U', 'V', 'M', 
    #                      'rkhs', 'kernel', 'svf', 'stationary',
    #                     'svvf', 'tvvf', 'time-varying', 'time_varying',
    #                     'piecewise-stationary', 'piecewise_stationary', 'lddmm', 'shooting',
    #                     'diffeo', 'diffeomorphism', 'diffeomorphic',
    #                     'diffeo', 'diffeomorphism']:
    elif (isinstance(transformer, str) and
          transformer.strip().lower() in
          (set(pwDiffeoRegistration._ALIASES) | pwDiffeoRegistration._GENERIC_ALIASES)):
        model = pwDiffeoRegistration(*args, transformer=transformer, **kwargs)
    else:
        raise ValueError('transformer must be one of ERTILOSAPDHKUVM or rigid, affine, isosimilarity, similarity,'
                         'projectve, deformable, hierarchical,  constraineddeformable',
                         'K/rkhs, U/svf, V/svvf/tvvf, M/lddmm or diffeomorphism')
    model.register( callback=callback )
    return model

def SPR(*args, transformer='A', callback=None, **kwargs):
    from .paired_registration import spRegistration
    #TODO

def GWR(*args, transformer='affine', callback=None, **kwargs):
    from .groupwise_rigid_registration import gwRigidRegistration
    from .groupwise_affine_registration import gwAffineRegistration
    from .groupwise_similarity_registration import gwSimilarityRegistration
    from .groupwise_deformable_registration import gwDeformableRegistration
    from .groupwise_hierarchical_registration import gwHierarchicalRegistration
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
    elif transformer in ['H', 'hierarchical']:
        model = gwHierarchicalRegistration(*args, **kwargs)
    elif all([i in 'ESADHRTILON ' for i in transformer ]):
        model = gwComplexRegistration(*args, transformer=transformer, **kwargs)
    else:
        raise ValueError('transformer must be one of ESADHRTILON')
    model.register(callback=callback)
    return model

def RFR(*args, callback=None, **kwargs):
    from .reference_emregistration import rfEMRegistration
    model = rfEMRegistration(*args, **kwargs)
    model.register(callback=callback)
    return model

def RFR_z(*args, transformer='affine-z',  callback=None, **kwargs):
    from .zshift_affine_registration import zsAffineRegistration
    model = zsAffineRegistration(*args, transformer=transformer, **kwargs)
    model.register( callback=callback )
    return model
