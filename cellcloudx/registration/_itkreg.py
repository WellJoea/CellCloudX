try:
    import itk 
except:
    pass
    'pip install itk-elastix itk'
import os
import numpy as np
import skimage as ski
from skimage import transform as skitf
from ..transform._transp import homotransform_point
from ..transform._transi import homotransform
from ..utilis._arrays import list_iter

class itkregist():
    def __init__(self, ndim=2, transtype=None, resolutions=None, GridSpacing=None,  ParameterFiles=None, verb=False):
        self.transtype = ['rigid'] if transtype is None else transtype
        self.ndim = 2
        self.params = self.initparameters(trans=self.transtype,
                                            resolutions=list_iter(resolutions), 
                                            GridSpacing=list_iter(GridSpacing),
                                            ParameterFiles=ParameterFiles,
                                            verb=verb)

        if verb:
            print(self.params)

    def itkinit(self, fixed_img,
                 moving_img,
                 parameter_object,
                fixed_mask=None,
                moving_mask=None,
                log_to_console=False,
                init_paramter_file=None,
                output_directory=None,
                number_of_threads=10, 
                **kargs):
        elastix_object = itk.ElastixRegistrationMethod.New(fixed_img, moving_img)
        if not fixed_mask is None:
            elastix_object.SetFixedMask(fixed_mask)
        if not moving_mask is None:
            elastix_object.SetMovingMask(moving_mask)

        elastix_object.SetParameterObject(parameter_object)
        if not init_paramter_file is None:
            elastix_object.SetInitialTransformParameterFileName('')
        elastix_object.SetNumberOfThreads(number_of_threads)
        elastix_object.SetLogToConsole(log_to_console)

        if not output_directory is None:
            elastix_object.SetOutputDirectory(output_directory)
        #elastix_object.SetOutputDirectory('./exampleoutput/')
        # elastix_object.SetComputeSpatialJacobian(True)
        # elastix_object.SetComputeDeterminantOfSpatialJacobian(True)
        elastix_object.UpdateLargestPossibleRegion()
        # elastix_object.Update()

        result_image = elastix_object.GetOutput()
        result_transform_parameters = elastix_object.GetTransformParameterObject()
        return result_image, result_transform_parameters

    def itkmehtod(self, fixed_gray,
                    moving_gray,
                    parameter_object,
                    log_to_file=False,
                    log_file_name="regist.log",
                    number_of_threads=10,
                    output_directory='.', 
                    log_to_console= False,
                    **kargs):
        mov_out, paramsnew = itk.elastix_registration_method(
                                fixed_gray,
                                moving_gray,
                                parameter_object=parameter_object,
                                log_to_file=log_to_file,
                                log_file_name=log_file_name,
                                number_of_threads=number_of_threads,
                                output_directory=output_directory, 
                                log_to_console= log_to_console,
                                **kargs)
        return mov_out, paramsnew

    def regist(self, 
                fixed_img,
                moving_img,
                params=None, 
                fixed_mask=None,
                moving_mask=None,
                log_to_file=True,
                togray=True,
                log_file_name="regist.log",
                number_of_threads=5,
                output_directory=None,
                log_to_console=False,
                **kargs,
    ):

        fixed_itk = self.to_Image(fixed_img)
        mov_itk = self.to_Image(moving_img)

        if (not fixed_mask is None) and (not isinstance(fixed_mask, itk.Image)):
            fixed_mask = itk.GetImageFromArray(np.ascontiguousarray(fixed_mask.astype(np.uint32)))
        if (not moving_mask is None) and (not isinstance(moving_mask, itk.Image)):
            moving_mask = itk.GetImageFromArray(np.ascontiguousarray(moving_mask.astype(np.uint32)))  

        params = self.params if params is None else params
        mov_out, paramsnew = self.itkinit(fixed_itk,
                                            mov_itk,
                                            params,
                                            fixed_mask=fixed_mask,
                                            moving_mask=moving_mask,
                                            log_to_file=log_to_file,
                                            log_file_name=log_file_name,
                                            number_of_threads=number_of_threads,
                                            output_directory=output_directory, 
                                            log_to_console= log_to_console, **kargs)
        ntrans = paramsnew.GetNumberOfParameterMaps()
        self.tmats = [ self.get_transmtx(paramsnew, map_num=i ) for i in range(ntrans)]
        self.tmat = np.linalg.multi_dot(self.tmats) if len(self.tmats) > 1 else self.tmats[0]
        self.wrap_mov = mov_out
        self.paramsnew = paramsnew
        self.moving_img = moving_img
        self.fixed_img = fixed_img
        self.output_directory=output_directory
        return self

    def transform(self,  moving_img=None, locs = None, paramsnew=None, 
                  interp_method='skimage', inverse=True, swap_xy=True, 
                  tmat = None, **kargs):
        moving_img = self.moving_img if moving_img is None else moving_img
        paramsnew = self.paramsnew if paramsnew is None else paramsnew
        tmat = self.tmat if tmat is None else tmat

        if interp_method == 'skimage':
            mov_out = homotransform(moving_img, tmat, inverse=False, swap_xy=False, **kargs)
        else:
            # mov_out = itk.transformix_filter( self.to_Image(moving_img), 
            #                                  #fixed_point_set_file_name=loc_file,
            #                                  #output_directory = './'
            #                                  transform_parameter_object=paramsnew)
            mov_out = np.array(self.wrap_mov)

        self.mov_out = mov_out
        self.mov_locs = None
        if locs is not None:
            if interp_method == 'skimage':
                mov_locs = homotransform_point(locs, tmat, inverse=True,  swap_xy=True)
            else: #TODO
                loc_file = f'{self.random_seq(30)}.txt'
                output_directory = './' if self.output_directory is None else self.output_directory
                self.save_locfile(locs,  os.path.join(output_directory, loc_file))
                mov_locs = itk.transformix_pointset(self.to_Image(moving_img), 
                                                    paramsnew,
                                                    fixed_point_set_file_name=loc_file,
                                                    output_directory = output_directory)
            self.mov_locs = mov_locs
            return mov_out, mov_locs
        else:
            return mov_out

    def regist_transform(self, fixed_img, moving_img, 
                            locs=None, 
                            order=None,
                            targs={},
                            inverse=True,
                            swap_xy=True,
                            interp_method='skimage',
                            **kargs):
        self.regist(fixed_img, moving_img, **kargs)
        self.transform(moving_img, locs=locs, order=order, interp_method=interp_method, 
                        swap_xy=swap_xy, inverse=inverse, **targs)
        # return self
        return [self.mov_out, self.tmat, self.mov_locs]

    def to_Image(self, img):
        if not isinstance(img, itk.Image):
            img_itk = ski.color.rgb2gray(img) if img.ndim > self.ndim else img
            img_itk = itk.GetImageFromArray(np.ascontiguousarray(img_itk.astype(np.float32)))
        else:
            img_itk = img_itk
        return img_itk

    def random_seq(self, size):
        import random, string
        return ''.join([random.choice( 
                            string.ascii_letters + string.digits) 
                            for n in range(size)]) 

    def save_locfile(self, locs, path):
        np.savetxt(path, locs, fmt='%.18f', newline='\n', comments='', header=f'point\n{len(locs)}')

    def get_locfile(self, path):
        pass

    @staticmethod
    def get_transmtx(params, map_num=0 ):
        #https://elastix.lumc.nl/doxygen/classelastix_1_1SimilarityTransformElastix.html
        #https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html#x26-1170003.9
        #https://github.com/ANTsX/ANTsPy/issues/255
        #https://discourse.slicer.org/t/how-to-convert-between-itk-transforms-and-slicer-transform-nodes/34074
        #https://github.com/InsightSoftwareConsortium/ITKElastix/issues/145
        #https://github.com/dipy/dipy/discussions/2165
        #https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
        fixdim = params.GetParameter(map_num, 'FixedImageDimension')[0]
        tranty = params.GetParameter(map_num, 'Transform')[0]
        center = np.asarray(params.GetParameter(map_num, 'CenterOfRotationPoint')).astype(np.float64)
        trans = np.asarray(params.GetParameter(map_num, 'TransformParameters')).astype(np.float64)

        tform = None
        if (fixdim=='2') and (tranty=='EulerTransform'):
            tform = skitf.EuclideanTransform(rotation=trans[0],
                                             translation=trans[1:3]).params.astype(np.float64)
            shif = skitf.EuclideanTransform(translation=center, dimensionality=2).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        elif (fixdim=='2') and (tranty=='SimilarityTransform'):
            tform = skitf.SimilarityTransform(scale=trans[0],
                                              rotation=trans[1],
                                              translation=trans[2:4]).params.astype(np.float64)
            shif = skitf.SimilarityTransform(translation=center, dimensionality=2).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        elif (fixdim=='2') and (tranty=='AffineTransform'):
            tform = np.eye(3).astype(np.float64)
            tform[:2, :2] = np.array(trans[:4]).reshape(2,2)
            tform[:2, 2]  = trans[4:6]
            shif = skitf.AffineTransform(translation=center, dimensionality=2).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        elif (fixdim=='3') and (tranty=='AffineTransform'):
            tform = np.eye(4).astype(np.float64)
            tform[:3, :3] = np.array(trans[:9]).reshape(3,3)
            tform[:3, 3]  = trans[9:12]
            shif = skitf.SimilarityTransform(translation=center, dimensionality=3).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        else:
            tform = np.eye(3).astype(np.float64)
        return tform.astype(np.float64)

    @staticmethod
    def multilayer(image):
        if image.ndim==2:
            return False
        elif image.ndim==3:
            return True
        else:
            raise ValueError('the image must have 2 or 3 dims.')

    @staticmethod
    def colortrans(image, transcolor='rgb2gray', *args, **kargs):
        colorfunc = eval(f'ski.color.{transcolor}')
        image = colorfunc(image, *args, **kargs)
        return image

    @staticmethod
    def scaledimg(images):
        if (np.issubdtype(images.dtype, np.integer) or
            (images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])) and \
            (images.max() > 1):
            return False
        else:
            return True

    @staticmethod
    def initparameters(trans=None, resolutions=None, GridSpacing=None, verb=False, ParameterFiles = None):
        parameters = itk.ParameterObject.New()
        TRANS = ['translation', 'rigid', 'similarity', 'affine', 'bspline', 'spline', 'groupwise']

        trans = list_iter(trans, default=['rigid', 'bspline']).tolist()
        resolutions = list_iter(resolutions, default=[15, 10])
        GridSpacing = list_iter(GridSpacing, default=[15, 10])

        # if (set(trans) - set(TRANS)):
        #     raise TypeError(f'The valid transform type are {TRANS}.')

        for i, itran in enumerate(trans):
            ires = resolutions[i]
            igrid= GridSpacing[i]
            if  itran== 'similarity':
                default_para = parameters.GetDefaultParameterMap('rigid', ires, igrid)
                parameters.AddParameterMap(default_para)
                parameters.SetParameter(i, "Transform", "SimilarityTransform")
                parameters.SetParameter(i, "AutomaticScalesEstimation", "true")
                # parameters.SetParameter(i, "AutomaticTransformInitialization", "false")
                parameters.SetParameter(i, "AutomaticTransformInitializationMethod ", "GeometricalCenter")
            else:
                try:
                    default_para = parameters.GetDefaultParameterMap(itran, ires, igrid)
                    parameters.AddParameterMap(default_para)
                except:
                    print(f'Cannot set {itran}  as the valid transtype! Will be replaced by translation.')
                    default_para = parameters.GetDefaultParameterMap('translation', ires, igrid)
                    parameters.AddParameterMap(default_para)

            if not ParameterFiles is None:
                parameters.AddParameterFile(ParameterFiles[i])

        for itr, itran in enumerate(trans):
            # https://elastix.lumc.nl/doxygen/parameter.html
            # parameters.SetParameter(itr, "Optimizer", "RegularStepGradientDescent")
            parameters.SetParameter(itr, "Optimizer", "AdaptiveStochasticGradientDescent")
            parameters.SetParameter(itr, "MaximumNumberOfIterations", "3000")
            parameters.SetParameter(itr, "MaximumStepLength", "10")
            # parameters.SetParameter(0, "MinimumStepLength", "0.001")
            parameters.SetParameter(itr, "RelaxationFactor", "0.5")

            parameters.SetParameter(itr, "NumberOfGradientMeasurements", "0")
            parameters.SetParameter(itr, "NumberOfSpatialSamples", "5000")
            parameters.SetParameter(itr, "NumberOfSamplesForExactGradient", "150000")
            parameters.SetParameter(itr, "FixedImagePyramid", "FixedRecursiveImagePyramid")
            parameters.SetParameter(itr, "MovingImagePyramid", "MovingRecursiveImagePyramid")
            parameters.SetParameter(itr, "NumberOfResolutions", "8")

            if itran== 'bspline':
                parameters.SetParameter(itr, "FinalGridSpacingInPhysicalUnits", "16")
                parameters.SetParameter(itr, "HowToCombineTransforms", "Compose")
                parameters.SetParameter(itr, "NumberOfResolutions", f'"{ires}"')
                parameters.SetParameter(itr, "Interpolator", "BSplineInterpolator")
                parameters.SetParameter(itr, "ResampleInterpolator", "FinalBSplineInterpolator")
                parameters.SetParameter(itr, "BSplineInterpolationOrder", "1")
                parameters.SetParameter(itr, "FinalBSplineInterpolationOrder", "3")

            #parameters.SetParameter(itr, "NumberOfHistogramBins", ["16", "32" ,"64"])
            # parameters.SetParameter(itr, "NumberOfResolutions", "5")
            # parameters.SetParameter(0, "ImagePyramidSchedule",list(np.repeat([32, 16, 8, 4, 1], 2).astype(str)))
            # parameters.SetParameter(0, "FixedImagePyramidRescaleSchedule",list(np.repeat([64, 32, 16, 8, 4, 1], 2).astype(str)))
            # parameters.SetParameter(0, "MovingImagePyramidRescaleSchedule",list(np.repeat([64, 32, 16, 8, 4, 1], 2).astype(str)))

            # parameters.SetParameter(itr, "Metric", "AdvancedMattesMutualInformation")

            # parameters.SetParameter(itr, "UseDirectionCosines", "true")
            # parameters.SetParameter(itr, "FixedInternalImagePixelType", "float")
            # parameters.SetParameter(itr, "MovingInternalImagePixelType", "float")
            # parameters.SetParameter(itr, "AutomaticTransformInitialization", "true")
            # parameters.SetParameter(itr, "AutomaticTransformInitializationMethod", "GeometricCenter")
            # parameters.SetParameter(itr, "WriteResultImage ", "false")
        #parameters.RemoveParameter("ResultImageFormat")
        if verb:
            print(parameters)
        return parameters