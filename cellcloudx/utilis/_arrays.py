from typing import Any
import numpy as np
import imageio.v3 as iio
import pandas as pd
from scipy.sparse import issparse, diags
import collections

class list_iter():
    def __init__(self, terms, default = None, last_as_default = False, l2d=None, force=False):
        '''
        l2d :  last_as_default(l2d) last term as default
        '''
        self.default = default

        if (terms is None):
            terms = self.default

        if force:
            terms = [terms]
            
        if ((type(terms) in [str, float, int, bool]) 
            or isinstance(terms, (str, bytes))
            or np.isscalar(terms) 
            or (terms is None)):
            self.default = terms if default is None else default
            self.terms = [terms]
        elif type(terms) in [list, tuple, np.ndarray]:
            self.terms = terms
        elif isinstance(terms, (pd.Series, pd.Index)):
            self.terms = terms.tolist()
        elif isinstance(terms, (collections.abc.KeysView, dict)):
            self.terms = list(terms)
        else:
            self.terms = terms

        if l2d is not None:
            last_as_default = l2d
        if last_as_default:
            self.default = self.terms[-1]

    def __len__(self):
        return len(self.terms)
    
    def tolist(self):
        return self.terms

    def __call__(self, idx = None) -> Any:
        if idx is None:
            return self.terms[0]
        else:
            if idx < len(self.terms):
                return self.terms[idx]
            else:
                return self.default

    def __getitem__(self, idx = None) -> Any:
        return self.__call__(idx)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.terms):
            result = self.terms[self.n]
            self.n += 1
            return result
        else:
            return self.default
    def __repr__(self) -> str:
            """Representation showing available terms and default value."""
            return f"list_iter(terms={self.tolist()}, default={self.default})"

def list_get(l, idx, default=None):
    try:
        return l[idx]
    except:
        return default
def vartype(vector):
    if isinstance(vector, (list, tuple)):
        vector = np.asarray(vector)
    if isinstance(vector, pd.Series):
        dtype = vector.dtype
    elif isinstance(vector, pd.DataFrame):
        dtype = vector.dtypes.iloc[0]
    elif isinstance(vector, np.ndarray):
        dtype = vector.dtype
    else:
        raise('wrong numpy array or pandas object.')

    if (
        isinstance(dtype,  pd.CategoricalDtype) 
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):  
        return "discrete"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "continuous"
    else:
        raise('wrong numpy array or pandas object.')

def checksymmetric(adjMat, rtol=1e-05, atol=1e-08):
    if issparse(adjMat):
        #from scipy.linalg import issymmetric
        #return issymmetric(adjMat.astype(np.float64), atol=atol, rtol=rtol)
        adjMat = adjMat.toarray()
    return np.allclose(adjMat, adjMat.T, rtol=rtol, atol=atol)

def isidentity(M, equal_nan=True):
    if (M.shape[0] == M.shape[1]) and \
        np.allclose(M, np.eye(M.shape[0]), equal_nan=equal_nan):
        return True
    else:
        return False

def transsymmetric(mtx):
    if not checksymmetric(mtx):
        return (mtx + mtx.T)/2
    else:
        return mtx

def set_diagonal(mtx, val=1, inplace=False):
    assert mtx.shape[0] == mtx.shape[1], "Matrix must be square"
    if issparse(mtx):
        diamtx = diags(val- mtx.diagonal(), dtype=mtx.dtype)
        mtx = mtx + diamtx
        mtx.sort_indices()
        mtx.eliminate_zeros()
        return mtx

    elif isinstance(mtx, np.ndarray):
        mtx = mtx if inplace else mtx.copy()
        np.fill_diagonal(mtx, val)
        return mtx

def vars(mtx, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    if issparse(mtx):
        mtx_v = mtx.copy()
        mtx_v.data **= 2
        return mtx_v.mean(axis) - np.square(mtx.mean(axis))
    elif isinstance(mtx, np.ndarray):
        return np.var(mtx, axis)

def take_data(array, index, axis):
    sl = [slice(None)] * array.ndim
    sl[axis] = index
    return array[tuple(sl)]

def img2pos(img, thred=0):
    pos = np.where(img>thred)
    value = img[pos]
    return np.c_[(*pos, value)]

def loc2mask(locus, size, axes=[2,0,1]):
    img = np.zeros(size, dtype=np.int64)
    if locus.shape[1] - len(size) >= 1:
        values = locus[:,len(size)]
    elif locus.shape[1] - len(size) == 0:
        values = np.ones(locus.shape[0])

    pos = np.round(locus[:,:len(size)]).astype(np.int64)
    print(pos.shape, img.shape, pos.max(0))
    img[tuple(pos.T)] = values
    if not axes is None:
        img = np.transpose(img, axes=axes)
    return img

def sort_array(array, ascending=False):
    try:
        from scipy.sparse import csr_matrix as csr_array
    except:
        from scipy.sparse import csr_array
    arr_sp = csr_array(array)
    arr_dt = arr_sp.data
    arr_rc = arr_sp.nonzero()
    arr_st = np.vstack([arr_rc, arr_dt]).T
    if ascending:
        arr_st = arr_st[arr_st[:,2].argsort()]
    else:
        arr_st = arr_st[arr_st[:,2].argsort()[::-1]]
    return arr_st

def Info(sitkimg):
    print('***************INFO***************')
    print(f"origin: {sitkimg.GetOrigin()}")
    try:
        print(f"size: {sitkimg.GetSize()}")
    except:
        pass
    print(f"spacing: {sitkimg.GetSpacing()}")
    print(f"direction: {sitkimg.GetDirection()}")
    try:
        print( f"dimension: {sitkimg.GetDimension()}" )
    except:
        print( f"dimension: {sitkimg.GetImageDimension()}" )
    try:
        print( f"width: {sitkimg.GetWidth()}" )
        print( f"height: {sitkimg.GetHeight()}" )
        print( f"depth: {sitkimg.GetDepth()}" )
        print( f"pixelid value: {sitkimg.GetPixelIDValue()}" )
        print( f"pixelid type: {sitkimg.GetPixelIDTypeAsString()}" )
    except:
        pass
    print( f"number of components per pixel: {sitkimg.GetNumberOfComponentsPerPixel()}" )
    print('*'*34)

def spimage(adata, file=None, image=None, img_key="hires", basis = None, rescale=None,
           library_id=None, **kargs):
    if not file is None:
        image = iio.imread(file, **kargs)
    
    basis = 'spatial' if basis is None else basis
    library_id = 'slice0' if (library_id is None) else library_id
    
    rescale = 1 if rescale is None else rescale
    if rescale != 1:
        import cv2
        rsize = np.ceil(np.array(image.shape[:2])*rescale)[::-1].astype(np.int32)
        if image.ndim==3:
            image = cv2.resize(image[:,:,::-1], rsize, interpolation= cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, rsize, interpolation= cv2.INTER_LINEAR)

    if (basis in adata.uns.keys()) and (library_id in adata.uns[basis].keys()):
        adata.uns[basis][library_id]['images'][img_key] = image
        adata.uns[basis][library_id]['scalefactors'][f'tissue_{img_key}_scalef'] = rescale
    else:
        img_dict ={
            'images':{img_key: image},
            #unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
            #spot.radius <-  unnormalized.radius / max(dim(x = image))
            'scalefactors': {'spot_diameter_fullres': 1, ##??
                             'fiducial_diameter_fullres': 1 ,
                             f'tissue_{img_key}_scalef':rescale,
                             'spot.radius':1, 
                            },
            'metadata': {'chemistry_description': 'custom',
                           'spot.radius':  1, 
                          }
        }
        adata.uns[basis] = {library_id: img_dict}
