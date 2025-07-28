import numpy as np
import skimage as ski
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
try:
    from scipy.ndimage import map_coordinates
except ImportError:
    from scipy.ndimage.interpolation import map_coordinates

from ..transform._transi import fieldtransform, fieldtransforms
from ..transform._transp import fieldtransform_point, fieldtransform_points

class optical_flow():
    def __init__(self):
        self.fieldtransform = fieldtransform
        self.fieldtransform_point = fieldtransform_point

    def regist(self, fixed_img, moving_img,
               method='tlv1',
                    togray=True, 
                    attachment=15,
                    tightness=0.3,
                    num_warp=5,
                    num_iter=10,
                    tol=0.0001,
                    prefilter=False,
                    radius = 20,
                    gaussian = False,
                    dtype = np.float64,
                    verbose = 1,
                    **kargs):
        assert fixed_img.shape == moving_img.shape
        self.fixed_img = fixed_img
        self.moving_img = moving_img

        if togray and fixed_img.ndim==3:
            fixed_img = ski.color.rgb2gray(fixed_img)
            moving_img = ski.color.rgb2gray(moving_img)

        if (fixed_img.max()>1) or (moving_img.max()>1):
            verbose and print('Warning: not gray scale image.')
            fixed_img = fixed_img.copy()/fixed_img.max()
            moving_img = moving_img.copy()/moving_img.max()

        if method == 'tlv1':
            V, U = optical_flow_tvl1(fixed_img.astype(np.float32), 
                                    moving_img.astype(np.float32),
                                    attachment=attachment, #5
                                    tightness=tightness, 
                                    num_warp=num_warp, #5
                                    num_iter=num_iter,
                                    tol=tol, #0.001
                                    prefilter=prefilter,
                                    dtype=dtype)
        elif method == 'ilk':
            V, U = optical_flow_ilk(fixed_img.astype(np.float32), 
                                    moving_img.astype(np.float32),
                                    radius = radius,
                                    num_warp= num_warp,
                                    gaussian=gaussian,
                                    prefilter=prefilter,
                                    dtype=dtype)
        else:
            raise ValueError('Unknown method: {}'.format(method))
        nr, nc = fixed_img.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), 
                                            np.arange(nc),
                                            indexing='ij')
        coords = np.array([row_coords + V, col_coords + U])

        self.V = V
        self.U = U
        self.VUs = np.float64([V,U])
        self.coords = coords
        return coords

    def transform(self, moving_img=None,  coords=None, locs=None,  VUs = None, 
                  interp_method='linear', **kargs):
        moving_img = self.moving_img if moving_img is None else moving_img
        coords = self.coords if coords is None else coords
        mov_out = self.fieldtransform(moving_img, coords, **kargs)
        self.mov_out = mov_out
        self.mov_locs = None

        if locs is not None:
            VUs = self.VUs if VUs is None else VUs
            mov_locs = self.fieldtransform_point(locs, VUs, method=interp_method)
            self.mov_locs = mov_locs
            return mov_out, mov_locs
        else:
            return mov_out

    def regist_transform(self, fixed_img, moving_img, locs=None, mode='edge',
                          interp_method='linear', order=1, **kargs):
        self.regist(fixed_img, moving_img, **kargs)
        self.transform(mode=mode, locs=locs, order=order, interp_method=interp_method)
        return [self.mov_out, self.VUs, self.mov_locs]
