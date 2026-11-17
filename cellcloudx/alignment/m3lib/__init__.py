#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : __init__.py
* @Author  : Wei Zhou                                     *
* @Date    : 2024/12/08 09:33:22                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* Please give me feedback if you find any problems.       *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''
from .pairwise_emregistration import pwEMRegistration
from .pairwise_rigid_registration import pwRigidRegistration
from .pairwise_affine_registration import pwAffineRegistration
from .pairwise_similarity_registration import pwSimilarityRegistration
from .pairwise_projective_registration import pwProjectiveRegistration
from .pairwise_deformable_registration import pwDeformableRegistration
from .pairwise_constrained_deformable_registration import pwConstrainedDeformableRegistration
from .pairwise_hierarchical_registration import pwHierarchicalRegistration
from .pairwise_diffeomorphism_registration import pwDiffeoRegistration


from .paired_ansac_registration import spAnsac
from .paired_emregistration import spEMRegistration
from .paired_registration import spRegistration

from .groupwise_emregistration import gwEMRegistration
from .groupwise_rigid_registration import gwRigidRegistration
from .groupwise_affine_registration import gwAffineRegistration
from .groupwise_similarity_registration import gwSimilarityRegistration
from .groupwise_deformable_registration import gwDeformableRegistration
from .groupwise_hierarchical_registration import gwHierarchicalRegistration
from .groupwise_complex_registration import gwComplexRegistration

from .reference_emregistration import rfEMRegistration
from .reference_basic_registration import rfBasicRegistration
from .reference_combined_registration import rfCombinedRegistration
from .reference_complex_registration import rfComplexRegistration
from .zshift_affine_registration import zsAffineRegistration


from .M3_registration import PWR, SPR, GWR, RFR, RFR_z
from .fpfh import FPFH, FPFH_batch, image_features, image_features_batch
from .otmatching import OTM, matching, OTM_solver, plot_matching_report, plot_matching_summary
from .label_counts import label_onehots, label_onehot, gwr_omega

from .operation_th import thopt
from . import (
    pairwise_operation,  paired_operation,  reference_operation, groupwise_operation, operation_expectation, operation_utility,  groupwise_emregistration, manifold_regularizers, 
    neighbors_ensemble, shared_wnn, xmm, pairwise_llt_registration, reference_emregistration, fpfh,
    refreshed_budgets
)
