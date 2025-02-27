import numpy as np
import skimage as ski
try:
    import SimpleITK as sitk
except:
    pass
# This class is a wrapper for SimpleITK registration.
class sitkregist:
    """
    sitkregist
    """

    def __init__(self, transtype='rigid', ndim=2):
        self.transtype = transtype
        self.ndim = ndim

    def regist(self, 
                fixed_img,
                moving_img,

                matrix_type='MMI', 
                numberOfHistogramBins = None,
                intensityDifferenceThreshold = None,
                optimizer_type='GD',
                itp_type='linear',
                matrix_kargs = {},
                optimizer_kargs = {},

                initial_transform = None,
                msp= 0.5,
                seed = 491001,
                centralRegionRadius=5, 
                smallParameterVariation=0.01,
                fixed_image_mask=None,
                number_of_threads = 64,

                shrinkFactors=[8, 4, 2, 1],
                smoothingSigmas=[4, 2, 1, 0],
                AddCommand = None,
                verbose=1,
                debugon = False,
                **kargs):

        fixed_sitk = self.image2sitk(fixed_img, self.ndim)
        moving_sitk = self.image2sitk(moving_img, self.ndim)

        if self.transtype == 'displacementfiled': #TODO
            self.reg_method = sitk.ImageRegistrationMethod()
            df_init = sitk.TransformToDisplacementFieldFilter()
            df_init.SetReferenceImage(fixed_sitk)
            # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
            initial_transform = sitk.DisplacementFieldTransform(
                df_init.Execute(sitk.Transform())
            )
            initial_transform.SetSmoothingGaussianOnUpdate(
                varianceForUpdateField=0.0, varianceForTotalField=2.0)
            self.reg_method.SetInitialTransform(initial_transform)
            self._Matrix( matrix_type=matrix_type, 
                          numberOfHistogramBins=numberOfHistogramBins,
                          intensityDifferenceThreshold=intensityDifferenceThreshold,
                        **matrix_kargs )
            self.reg_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactors)
            self.reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas)

            self.reg_method.SetInterpolator(self._interpolator(itp_type))

            self.reg_method.SetNumberOfThreads(number_of_threads)
            self._optimizer(optimizer_type=optimizer_type,
                            **optimizer_kargs)

            self.reg_method.SetOptimizerScalesFromPhysicalShift()
            self.reg_method.SetOptimizerScalesFromPhysicalShift(
                            centralRegionRadius=centralRegionRadius, 
                            smallParameterVariation=smallParameterVariation)

            if not fixed_image_mask is None:
                self.reg_method.SetMetricFixedMask(fixed_image_mask)
            self.reg_method.SetDebug(debugon)
    
        else:
            if initial_transform is None:
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_sitk, 
                    moving_sitk,
                    self._transformer(fixed_image=fixed_sitk),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

            self.reg_method = sitk.ImageRegistrationMethod()

            # Similarity metric settings.
            self._Matrix( matrix_type=matrix_type, numberOfHistogramBins=numberOfHistogramBins, 
                            **matrix_kargs)
            self.reg_method.SetMetricSamplingStrategy(self.reg_method.RANDOM)
            self.reg_method.SetMetricSamplingPercentage(msp, seed =seed )

            # Interpolator settings.
            self.reg_method.SetInterpolator(self._interpolator(itp_type))
            # Optimizer settings.
            self.reg_method.SetNumberOfThreads(number_of_threads)
            self._optimizer(optimizer_type=optimizer_type,
                            **optimizer_kargs)

            self.reg_method.SetOptimizerScalesFromPhysicalShift()
            self.reg_method.SetOptimizerScalesFromPhysicalShift(
                            centralRegionRadius=centralRegionRadius, 
                            smallParameterVariation=smallParameterVariation)

            # Setup for the multi-resolution framework.
            self.reg_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactors)
            self.reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas)
            self.reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Perform the registration in-place so that the initial_transform is modified.
            if not fixed_image_mask is None:
                self.reg_method.SetMetricFixedMask(fixed_image_mask)
            self.reg_method.SetInitialTransform(initial_transform, inPlace=True)
            self.reg_method.SetDebug(debugon)

        if not AddCommand is None:
            for icommd in AddCommand:
                if isinstance(icommd, str):
                    self.reg_method.AddCommand(*eval(icommd))
                else:
                    self.reg_method.AddCommand(*icommd)

        verbose >2 and print(self.reg_method)
        tparas = self.reg_method.Execute(
                        sitk.Cast(fixed_sitk, sitk.sitkFloat32),
                        sitk.Cast(moving_sitk, sitk.sitkFloat32))
        if transtype == 'displacementfiled':
            mov_out = sitk.DisplacementFieldTransform(tparas)
        else:
            mov_out = sitk.TransformGeometry(moving_sitk, tparas)

        # moving_resampled = sitk.Resample(
        #     moving_sitk,
        #     fixed_sitk,
        #     tparas,
        #     sitk.sitkLinear,
        #     0.0,
        #     moving_sitk.GetPixelID())

        # Query the registration method to see the metric value and the reason the
        # optimization terminated.
        verbose and print('Final metric value: {0}'.format(self.reg_method.GetMetricValue()))
        verbose and print('Optimizer\'s stopping condition, {0}'.format(
                            self.reg_method.GetOptimizerStopConditionDescription()))

        self.tparas = tparas
        self.mov_out = mov_out
        self.fixed_img = fixed_img
        self.moving_img = moving_img
        # sitk.WriteTransform(reg_method, full_file_name)
        # read_result = sitk.ReadTransform(full_file_name)
        return mov_out, tparas

    def _transformer(self, fixed_image=None, 
                     grid_physical_spacing=[50,50,50], order =3 ):
        mesh_size = [
            int(size * spacing / grid_spacing + 0.5)
            for size, spacing, grid_spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing(), grid_physical_spacing)
        ]

        transdict = {
            2:{
                'translation': sitk.TranslationTransform(2),
                "euler": sitk.Euler2DTransform(),
                "rigid": sitk.Euler2DTransform(),
                'similarity': sitk.Similarity2DTransform(),
                'scale': sitk.ScaleTransform(2),
                'affine': sitk.AffineTransform(2),
                'bspline': sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize=mesh_size, order=order),
                'displacementfiled': sitk.DisplacementFieldTransform(2),
                # 'composite': sitk.CompositeTransform(2),
                # 'transform': sitk.Transform(2)
            },
            3:{
                'transform': sitk.TranslationTransform(3),
                'versor':sitk.VersorTransform(),
                'versorrigid':sitk.VersorRigid3DTransform(),
                "euler": sitk.Euler3DTransform(),
                "rigid": sitk.Euler3DTransform(),
                'similarity': sitk.Similarity3DTransform(),
                'scale': sitk.ScaleTransform(3),
                'scaleversor': sitk.ScaleVersor3DTransform(),
                'scaleskewversor': sitk.ScaleSkewVersor3DTransform(),
                'affine': sitk.AffineTransform(3),
                'bspline': sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize=mesh_size, order=order),
                'displacementfiled': sitk.DisplacementFieldTransform(3),
                # 'composite': sitk.CompositeTransform(3),
                # 'transform': sitk.Transform(3)
            },
        }
        try:
            tranformer = transdict[self.ndim][self.transtype]
        except KeyError:
            print('local difined transformer.')
            tranformer = self.transtype
        return tranformer

    def _Matrix(self, matrix_type='MMI', 
                numberOfHistogramBins=None, radius=None,  intensityDifferenceThreshold=None, 
                varianceForJointPDFSmoothing=None):
        if matrix_type == 'MMI':
            self.reg_method.SetMetricAsMattesMutualInformation(
                                numberOfHistogramBins=numberOfHistogramBins or 32)
        elif matrix_type == 'ANTS':
            self.reg_method.SetMetricAsANTSNeighborhoodCorrelation(radius=radius or 50)
        elif matrix_type == 'CORR':
            self.reg_method.SetMetricAsCorrelation()
        elif matrix_type == 'JHMI':
            self.reg_method.SetMetricAsJointHistogramMutualInformation(
                numberOfHistogramBins=numberOfHistogramBins or 20,
                varianceForJointPDFSmoothing=varianceForJointPDFSmoothing or 1.5)
        elif matrix_type == 'MS':
            self.reg_method.SetMetricAsMeanSquares()
        elif matrix_type == 'Demon':
            self.reg_method.SetMetricAsDemons(intensityDifferenceThreshold or 1)	


        else:
            raise ValueError('Pealse input a valid matrix type: MMI, CORR, ANTS, JHMI and MS')

    def _optimizer(self, optimizer_type='GD', **kargs):
        # Optimizer settings.
        # https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/62_Registration_Tuning.html

        if optimizer_type == 'GD':
            self.reg_method.SetOptimizerAsGradientDescent(
                        learningRate=kargs.pop('learningRate',1.0),
                        numberOfIterations=kargs.pop('numberOfIterations',5000),
                        convergenceMinimumValue=kargs.pop('convergenceMinimumValue',1e-6),
                        convergenceWindowSize=kargs.pop('convergenceWindowSize',10),
                        **kargs )
            # self.reg_method.SetOptimizerScales(
            #         kargs.pop('OptimizerScales', [1, 1, 1, 1, 1, 1]))
        elif optimizer_type == 'RSGD':
            self.reg_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=kargs.pop('learningRate',1.0),
                minStep=kargs.pop('minStep',1e-4),
                numberOfIterations=kargs.pop('numberOfIterations',5000),
                relaxationFactor=kargs.pop('relaxationFactor',0.5),
                gradientMagnitudeTolerance=kargs.pop('gradientMagnitudeTolerance', 1e-6),
                # estimateLearningRate=kargs.pop('estimateLearningRate', 'Once'),
                maximumStepSizeInPhysicalUnits=kargs.pop('maximumStepSizeInPhysicalUnits',0.0) )
        elif optimizer_type == 'LBFGSB':
            self.reg_method.SetOptimizerAsLBFGSB(
                    gradientConvergenceTolerance=kargs.pop('gradientConvergenceTolerance', 1e-5),
                    numberOfIterations=kargs.pop('numberOfIterations',1000),
                    maximumNumberOfCorrections=kargs.pop('maximumNumberOfCorrections',5),
                    maximumNumberOfFunctionEvaluations=kargs.pop('maximumNumberOfFunctionEvaluations', 2000),
                    costFunctionConvergenceFactor=kargs.pop('costFunctionConvergenceFactor',1e+7),
                **kargs)
        elif optimizer_type == 'LBFGS2':
            self.reg_method.SetOptimizerAsLBFGS2(
                    numberOfIterations= kargs.pop('numberOfIterations',0),
                    hessianApproximateAccuracy= kargs.pop('hessianApproximateAccuracy',6),
                    deltaConvergenceDistance= kargs.pop('deltaConvergenceDistance',0),
                    deltaConvergenceTolerance= kargs.pop('deltaConvergenceTolerance',1e-5),
                    lineSearchMaximumEvaluations= kargs.pop('lineSearchMaximumEvaluations',40),
                    lineSearchMinimumStep= kargs.pop('lineSearchMinimumStep',1e-20),
                    lineSearchMaximumStep= kargs.pop('lineSearchMaximumStep',1e20),
                    lineSearchAccuracy= kargs.pop('lineSearchAccuracy',1e-4),
                **kargs)
        elif optimizer_type == 'Exhaustive':
            self.reg_method.SetOptimizerAsExhaustive(
                    kargs.pop('numberOfSteps',  [5]*self.ndim*2),
                    stepLength= kargs.pop('stepLength',1))

        elif optimizer_type == 'Amoeba':
            self.reg_method.SetOptimizerAsAmoeba(
                        kargs.pop('simplexDelta', 2),
                        kargs.pop('numberOfIterations', 1000),
                        parametersConvergenceTolerance= kargs.pop('parametersConvergenceTolerance',1e-8),
                        functionConvergenceTolerance= kargs.pop('functionConvergenceTolerance',1e-4),
                **kargs)
        elif optimizer_type == 'Weights':
            self.reg_method.SetOptimizerWeights( kargs.pop('weights',[1]*self.ndim*2) )
            #Euler3DTransform:[angleX, angleY, angleZ, tx, ty, tz]
        elif optimizer_type == 'Powell':
            self.reg_method.SetOptimizerAsPowell(
                    numberOfIterations= kargs.pop('maximumStepSizeInPhysicalUnits',100),
                    maximumLineIterations= kargs.pop('maximumStepSizeInPhysicalUnits',100),
                    stepLength= kargs.pop('maximumStepSizeInPhysicalUnits',1),
                    stepTolerance= kargs.pop('maximumStepSizeInPhysicalUnits',1e-6),
                    valueTolerance= kargs.pop('maximumStepSizeInPhysicalUnits',1e-6),
                **kargs)
        else:
            raise ValueError('Pealse enter a valid optimizer type: GD, RSGD, LBFGSB,'
                             'LBFGS2, Exhaustive, Amoeba, Weights and Powell')

    @staticmethod
    def _interpolator(itp_type='linear'):
        return {
                "linear": sitk.sitkLinear,
                "nearest": sitk.sitkNearestNeighbor,
                "BSpline": sitk.sitkBSpline,
            }.get(itp_type, 'Incorrect interpolator input')

    @staticmethod
    def multilayer(image, dimension):
        if image.ndim==2:
            return False
        elif image.ndim==3 and dimension == 2:
            return True
        elif image.ndim==3 and dimension == 3:
            return False
        elif image.ndim==4 and dimension == 3:
            return True
        else:
            raise ValueError('the image must have 2-4 dims.')

    @staticmethod
    def image2sitk(image, dimension):
        if isinstance(image, sitk.Image):
            return image

        mlayer = sitkregist.multilayer(image, dimension)
        if dimension == 2:
            sitkimg = ski.color.rgb2gray(image) if mlayer else image

        elif dimension == 3:
            if mlayer:
                sitkimg = np.float32([ ski.color.rgb2gray(image[...,i]) 
                                        for i in range(image.shape[-1]) ])
            else:
                sitkimg = image

        if np.issubdtype(sitkimg.dtype, np.floating):
            sitkimg = sitkimg.astype(np.float32)
        # sitkimg = sitk.GetImageFromArray(np.ascontiguousarray(sitkimg), isVector=mlayer)
        sitkimg = sitk.GetImageFromArray(sitkimg)
        return sitkimg

    @staticmethod
    def GetTmat(AA):
        #tparas.GetNumberOfTransforms()
        # AA = tparas.GetNthTransform(idx)
        #R, T0, T1 = AA.GetParameters()
        D = AA.GetDimension()
        A = np.float64(AA.GetMatrix()).reshape(D,D)
        R = AA.GetAngle()
        C = np.float64(AA.GetCenter())
        T = np.float64(AA.GetTranslation())
        O = T+C - A @ C

        AT = np.eye(D + 1)
        AT[:D,:D] = A
        AT[:D, D] = O

        Cs = np.eye(D + 1)
        Cs[:D,D] = C
        # Tmat = Cs @ AT @ np.linalg.inv(Cs)
        # T, C, A, O, AT, Cs, Tmat
        return AT