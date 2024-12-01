import skimage as ski
import scipy as sci
import numpy as np
import pandas as pd

from typing import Optional, Union
from anndata import AnnData
from numpy import ndarray
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilis._arrays import list_iter, vartype
from joblib import Parallel, delayed

def interp_value(points, values, points_intp, method = 'sci', 
                 keys=None,
                   rescale=False,
                   radius = None, 
                   n_points=31,
                    null_strategy = 1,
                    sharp=2,
                    power=2,
                    null_value = np.nan,
                    drop_aggna=True,
                    n_job = 3,
                    agg = None,
                    kernel = 'linear'):
    '''
    null_strategy: MASK_POINTS = 0 , NULL_VALUE = 1 , CLOSEST_POINT = 2
    '''
    values, keys = to_narray(values, points, keys=keys)
    points, points_intp = to_nparray(points, points_intp)

    dtype = np.array([ vartype(values[:,i]) for i in range(len(keys)) ])
    outs = pd.DataFrame(columns = keys)
    dis_idx = dtype == 'discrete'
    con_idx = dtype == 'continuous'

    if np.any(dis_idx):
        values_s = values[:,dis_idx]
        values_l = np.arange(values_s.shape[0]).astype(np.int64)
        keys_s = np.array(keys)[dis_idx]
        if method == 'sci':
            import scipy as sci
            out_idx = sci.interpolate.griddata(points, values_l, points_intp,  rescale=rescale, 
                                               fill_value=null_value, method='nearest').astype(np.int64)
            outs_s = values_s[out_idx,:]
        elif method == 'kdt':
            out_idx = kd_interpolation(points, values_l, points_intp, n_points = n_points, radius=radius, 
                                        agg= (agg or 'mode'), drop_aggna=drop_aggna, null_value=null_value).astype(np.int64)
            outs_s = values_s[out_idx,:]
        elif method == 'vtk':
            out_idx = vtk_interpolation(points, values_l, points_intp,
                                        keys = ['index'],
                                            radius = radius, 
                                            n_points=n_points,
                                            null_strategy = 2,
                                            sharp=sharp,
                                            power=power,
                                            null_value = null_value,
                                            kernel = 'voronoi') 
            out_idx = out_idx['index'].astype(np.uint64).values
            outs_s = np.vstack([ values_s[:, i][out_idx] for i in range(values_s.shape[1]) ]).T

        outs[keys_s] = outs_s

    if np.any(con_idx):
        values_s = values[:,con_idx]
        keys_s = np.array(keys)[con_idx]

        if method == 'sci':
            import scipy as sci
            outs_s = Parallel( n_jobs=n_job, verbose=0)( delayed(sci.interpolate.griddata)
                                (points, values_s[:,i], points_intp, rescale=rescale, 
                                 fill_value=null_value, method=kernel) 
                            for i in range(values_s.shape[1]))
            outs_s = np.vstack(outs_s).T
        elif method == 'kdt':
            outs_s = kd_interpolation(points, values_s, points_intp, n_points = n_points, radius=radius, 
                                        agg= (agg or 'mean'), drop_aggna=drop_aggna, null_value=null_value)
        elif method == 'vtk':
            outs_s = vtk_interpolation(points, values_s, points_intp,
                                        keys = keys_s,
                                            radius = radius, 
                                            n_points=n_points,
                                            null_strategy = null_strategy,
                                            sharp=sharp,
                                            power=power,
                                            null_value = null_value,
                                            kernel = kernel) 
        outs[keys_s] = outs_s
    return outs

def kd_interpolation(points, values, points_intp, n_points = None, radius=None, 
                     agg='mean', drop_aggna=True, null_value=np.nan):
    from scipy import spatial
    from scipy import stats
    points = np.asarray(points)
    values = np.asarray(values)
    points_intp = np.asarray(points_intp)
    if agg == 'mean':
        aggfun = np.nanmean if drop_aggna else np.mean 
    elif agg == 'sum':
        aggfun = np.nansum if drop_aggna else np.sum
    elif agg == 'max':
        aggfun = np.nanmax if drop_aggna else np.max
    elif agg == 'min':
        aggfun = np.nanmin if drop_aggna else np.min
    elif agg == 'median':
        aggfun = np.nanmedian if drop_aggna else np.median
    elif agg == 'mode':
        def aggfun(x, **kargs):
            #lambda x: stats.mode(x, nan_policy='omit')[0][0]
            if drop_aggna:
                return stats.mode(x, nan_policy='omit', keepdims =True, **kargs)[0]
            else:
                return stats.mode(x, nan_policy='propagate', keepdims =True,**kargs)[0]
    elif callable(agg):
        aggfun = agg
    else:
        aggfun = eval(agg)

    p_tree = spatial.cKDTree(points)
    R, C = points_intp.shape[0], values.shape[1]
    values_intp = np.empty([R, C])
    values_intp[:] = null_value

    if not n_points is None:
        qidx = p_tree.query(points_intp, k=n_points)
        if not radius is None:
            qbool = ( qidx[0] <= radius )
            kinds = [ qidx[1][i][qbool[i]] for i in range(R) ]
        else:
            kinds = qidx[1]
    elif not radius is None:
        kinds = p_tree.query_ball_point(points_intp, radius)
    else:
        raise ValueError('Either n_points or radius must be specified.')

    for i in range(R):
        if len(kinds[i]):
            values_intp[i] = aggfun(values[kinds[i]], axis=0)
    return values_intp

def vtk_interpolation(points, values, points_intp,
                      keys = None,
                        radius = None, 
                        n_points=31,
                        null_strategy = 1,
                        sharp=2,
                        power=2,
                        null_value = np.nan,
                        null_to_closest = False,
                        kernel = 'linear'):
    '''
    null_strategy: MASK_POINTS = 0 , NULL_VALUE = 1 , CLOSEST_POINT = 2
    '''
    import vtk
    import pyvista as pv
    values, keys = to_narray(values, points, keys=keys)

    source_model = pv.PolyData(np.asarray(points))
    for i, ikey in enumerate(keys):
        source_model.point_data[ikey] = values[:,i]
    target_model = pv.PolyData(np.asarray(points_intp))

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(source_model)
    locator.BuildLocator()

    if kernel.lower() == "shepard":
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(power)
    elif kernel.lower() == "gaussian":
        kern = vtk.vtkGaussianKernel()
        kern.SetSharpness(sharp)
    elif kernel.lower() == "linear":
        kern = vtk.vtkLinearKernel()
    elif kernel.lower() == "voronoi":
        kern = vtk.vtkVoronoiKernel()
    else:
        raise ValueError(
            "Available `kernel` are: shepard, gaussian, linear, voronoi."
        )

    if kernel.lower() != "voronoi":
        if radius is None and not n_points:
            raise ValueError("Please set either radius or n_points")
        elif n_points:
            kern.SetNumberOfPoints(n_points)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(target_model)
    interpolator.SetSourceData(source_model)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    interpolator.PassFieldArraysOff()
    interpolator.SetNullPointsStrategy(null_strategy)
    interpolator.SetNullValue(null_value)

    if kernel.lower() == "voronoi"  and null_to_closest :
        interpolator.SetNullPointsStrategyToClosestPoint()
    # interpolator.SetValidPointsMaskArrayName("ValidPointMask") #
    interpolator.Update()
    cpoly = interpolator.GetOutput()
    interpolated_model = pv.wrap(cpoly)
    inter_val = pd.DataFrame({ikey: interpolated_model[ikey] for ikey in keys})[keys]
    return inter_val


def to_nparray(*arrays, to_column=False):
    iarrs = []
    for iarr in arrays:
        if isinstance(iarr, (pd.Series, pd.DataFrame)):
            iarr = iarr.to_numpy()
        elif isinstance(iarr, (list)):
            iarr = np.array(iarr)

        if isinstance(iarr, np.ndarray):
            if to_column & (iarr.ndim == 1):
                iarr = iarr.reshape(-1,1)
            iarr = iarr
        else:
            raise ValueError('values must be a numpy array or pandas Series or DataFrame or list')

        iarrs.append(iarr)
    return iarrs
            

def to_narray(values, points, keys = None):
    if isinstance(values, np.ndarray):
        if (values.ndim == 1) and (values.shape[0]==points.shape[0]):
            values = values.reshape(-1,1)
        else:
            assert (values.ndim == 2) and (values.shape[0]==points.shape[0])
        keys_ = [ f'group_{i}'for i in range(values.shape[1]) ]
    elif isinstance(values, pd.Series):
        keys_= [values.name]
        values = np.array(values).reshape(-1,1)
    elif isinstance(values, pd.DataFrame):
        keys_ = values.columns.tolist()
        values = values.values
    else:
        raise ValueError('values must be a numpy array or pandas Series or DataFrame')
    if keys is None:
        keys = keys_
    else:
        assert len(keys) == values.shape[1]
    return values, keys

def gaussian_density(points, xy_grid=None, normal=True):
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(points.T)
    if xy_grid is None:
        values = points
    else:
        values = xy_grid
    z = kernel(values.T)

    if normal:
        min_z = np.min(z)
        max_z = np.max(z)

        # Scale between 0 and 1
        z = (z - min_z) / (max_z - min_z)
    return z

def gaussian_intp(points, xy_grid, value=None):
    from scipy.interpolate import RBFInterpolator
    if value is None:
        values = np.ones((points.shape[0]))
    print(values.shape)
    kernel = RBFInterpolator(points, values)
    z = kernel(xy_grid)
    return z

def counts_ratio(x, y, xthred=10, ythred=10, EPS=1, state_method ='raw',):
    x = np.array(x)
    Y = np.array(y)
    idx0 = (y<=ythred) & (x<=xthred)
    X = x.copy()
    Y = y.copy()

    X[idx0] = 0
    Y[idx0] = 0
    if state_method == 'raw':
        ifc = (X+EPS)/(Y+EPS)
        ifc[idx0] = 0
        return  ifc

    elif state_method == 'log2fc':
        return np.log2(X+EPS) - np.log2(Y+EPS)

    elif state_method == 'inversal':
        ifc = (X+EPS)/(Y+EPS) - 1
        inv = (Y+EPS)/(X+EPS) - 1
        ifc[ifc<0] = -inv[ifc<0]
        return  ifc

    elif state_method == 'log2raw':
        ifc = (X+EPS)/(Y+EPS)
        ifc[idx0] = 0
        return  np.log2(ifc+EPS)