import scipy as sci
import numpy as np
import pandas as pd


from ..utilis._arrays import list_iter, vartype
from ..alignment.ccflib.xmm import kernel_xmm_k
from joblib import Parallel, delayed

def interp_value(points, values, points_intp, method = 'sci', 
                 keys=None,
                   rescale=False,
                   radius = None, 
                   n_points=31,
                    null_strategy = 1,
                    sharp=2,
                    power=2,
                    temp=1,
                    null_value = None,
                    drop_aggna=True,
                    n_job = 3,
                    agg = None,
                    threshold=0,
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

    if method == 'prob':
        pitp = prob_interpolation(points, points_intp, n_points=n_points,
                                  radius=radius, temp=temp)
        outs[keys] = pitp(values, threshold=threshold)[0]
        return outs
    
    null_value = null_value or np.nan
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
            outs_s = values_s[out_idx[:,0],:]
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

    if values.ndim ==1:
        values = values[:, np.newaxis]

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

class onehot():
    def __init__(self):
        self.order = None
        self.order_dict = None
        self.num_classes = None

    def fit(self, X, order=None, null_value='None'):
        if order is None:
            try:
                order = X.cat.categories
            except:
                order = np.unique(X)
        if null_value is not None:
            order = np.r_[order, [null_value]]
        assert len(order) >= np.unique(X).shape[0]

        self.order = np.array(order)
        self.order_dict = {k: i for i, k in enumerate(order)}
        self.num_classes = len(order)

        return self.transform(X)

    def transform(self, X):
        X_id = np.array([self.order_dict[k] for k in X])
        src = np.arange(len(X))
        dst = X_id
        dist = np.ones(len(X))
        onehot = sci.sparse.csr_array((dist, (src, dst)), 
                                      shape=(len(X), self.num_classes))
        return onehot
    
    def fit_transform(self, X):
        return self.fit(X)
    
    def inverse_transform(self, X):
        return self.order[np.array(X).astype(np.int64)]

class prob_interpolation():
    def __init__(self, X, Y, sigma2=None, temp=1, method='sknn', 
                   radius = None, 
                   n_points=None):
        self.X = X
        self.Y = Y
        if (n_points is None) and (radius is None):
            n_points = 25
        self.W = self.gaussian_prob(sigma2=sigma2, temp=temp, method=method, 
                                    n_points=n_points, radius=radius)
    def gaussian_prob(self, sigma2=None, temp=1, method='sknn', n_points=31, 
                      radius=None):
        P, R, simga2 = kernel_xmm_k( self.Y, self.X,
                                    sigma2=sigma2, temp=temp, method=method, 
                                    knn=n_points, radius=radius)
        W = P/np.clip(P.sum(0), 1e-8, None) #N*M
        W = W.T
        W.eliminate_zeros()
        return W

    def smoothed_probs(self, weights, onehot_labels):
        return weights.dot(onehot_labels)

    def predict_labels(self, prob, threshold=0):
        if threshold > 0:
            prob = prob.toarray().copy()
            prob[:,-1] = threshold
        return prob.argmax(1), prob.max(1).toarray().flatten()

    def get_W_dict(self):
        sdzip = zip(self.W.nonzero()[0].tolist(),
                    self.W.nonzero()[1].tolist())
        W_dict = {}
        for s,d in sdzip:
            if s not in W_dict:
                W_dict[s] = [d]
            else:
                W_dict[s].append(d)
        self.W_dict = W_dict

    def predict_maxidx(self, ivalue, prob_label=None):
        if prob_label is None:
            W_mask = self.W * ivalue
        else:
            # W_mask = prob_label[:, None] == ivalue[None,:]
            # W_mask = self.W * W_mask
            key_set = {}
            for i, k in enumerate(ivalue):
                if k not in key_set:
                    key_set[k] = [i]
                else:
                    key_set[k].append(i)

            incs = []
            for i, k in enumerate(prob_label):
                iv = list( set(key_set.get(k, [])) & set(self.W_dict.get(i, [])) )
                incs.append(iv)
            # incs = [ key_set.get(v, []) for v in prob_label ] #large array
            src = np.concatenate(incs, axis=0)
            dst = np.repeat(np.arange(len(incs)), list(map(len, incs)))
            val = np.ones(len(src))

            W_mask = sci.sparse.csr_array((val, (dst, src)), 
                                          shape=self.W.shape)
            del src, dst, val, key_set, incs
            W_mask = self.W * W_mask

        P_idx = W_mask.argmax(1)
        P_didx = W_mask.max(1).toarray().flatten() <= 0.0
        P_idx[P_didx] = -1
        return P_idx

    def __call__(self, values, threshold=0.0, null_value='_NONE', predict_idx=False):
        values = np.array(values)
        if values.ndim == 1:
            values = values.reshape(-1,1)
        dtype = np.array([ vartype(values[:,i]) for i in range(values.shape[1]) ])

        outvalue = []
        outscore = []
        outsindx = []
        if ('discrete' in dtype) and predict_idx:
            self.get_W_dict()

        for i, itype in enumerate(dtype):
            ivalue = values[:,i]
            if itype == 'discrete':
                OH = onehot()
                IV = OH.fit(ivalue, null_value=null_value)
                P = self.smoothed_probs(self.W, IV)
                PI, PV = self.predict_labels(P, threshold=threshold)
                PL = OH.inverse_transform(PI)
                outvalue.append(PL)
                outscore.append(PV)
                if predict_idx:
                    PD = self.predict_maxidx(ivalue, prob_label=PL)
                    outsindx.append(PD)
            elif itype == 'continuous':
                IV = ivalue
                P = self.smoothed_probs(self.W, IV)
                outvalue.append(P)
                outscore.append(P)

                if predict_idx:
                    PD = self.predict_maxidx(ivalue)
                    outsindx.append(PD)
            else:
                raise ValueError('Unknown variable type')
        return np.array(outvalue).T, np.array(outscore).T, np.array(outsindx).T


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