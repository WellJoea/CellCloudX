import numpy as np
from skimage import filters
import collections
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
import scipy as sci
from joblib import Parallel, delayed

def Invervals(vector, CI = 0.995, tailed ='two', kernel='poi'):
    #from KDEpy import FFTKDE
    #x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(Dist).evaluate()
    #(x*y).sum() (((x - (x*y).sum())**2) * y).sum()
    mu = np.mean(vector)
    std = np.std(vector)

    if kernel in ['gaussian', 'norm']:
        # ippf = sci.stats.norm.ppf((1+CI)/2 , 0, 1)
        # greater = np.mean(vector) + ippf*np.std(vector) #/np.sqrt(N)
        # less = np.mean(vector) - ippf*np.std(vector)
        less, greater = sci.stats.norm.interval(CI, mu, std)
    elif kernel in ['poisson', 'poi']: 
        less, greater = sci.stats.poisson.interval(CI, mu, loc= mu)
    elif kernel in ['nbinom', 'nb']: 
        n = (mu**2)/(std**2 - mu)
        p = mu/(std**2)
        less, greater = sci.stats.nbinom.interval(CI, n, p, loc= mu)

    if tailed == 'two':
        Y = (vector>less) & (vector<greater)
        return [np.float32(Y), less, greater]
    elif tailed == 'less':
        Y = (vector>less)
        return [np.float32(Y), less]
    elif tailed == 'greater':
        Y = (vector<greater)
        return [np.float32(Y), greater]

def vectorclip(vector, method='all', agg='mean', CI = 0.995, tailed ='greater', kernel='gaussian'):
    methods = {
                'isodata': filters.threshold_isodata,
                'li': filters.threshold_li,
                'mean': filters.threshold_mean,
                'minimum': filters.threshold_minimum,
                'otsu': filters.threshold_otsu,
                'triangle': filters.threshold_triangle,
                'yen': filters.threshold_yen,
                'ci':Invervals}
    if method == 'all':
        threds = {}
        for k,ifilter in methods.items() :
            threds[k] = vectorclip( vector, method=k, CI = CI, tailed =tailed, kernel=kernel)
        thred = eval(f'np.{agg}')(np.sort(list(threds.values()))[1:-1])

    elif method in methods.keys():
        ifilter = methods[method]
        thred = vector.max() if tailed in ['greater'] else vector.min()

        if method == 'ci':
            thred = ifilter(vector, CI = CI, tailed =tailed)[1]
        elif method in ['isodata', 'minimum']:
            try:
                thred = ifilter(vector, nbins=100)
            except:
                pass
        else:
            try:
                 thred = ifilter(vector)
            except:
                pass
        return thred
    else:
        raise('the filter must be in one of "all, ci, isodata, li, mean, minimum, otsu, triangle, yen"')
    return thred

def outlier_detector(X, outliers_fraction = 0.05, seed = 200504, n_neighbors=20, backend='threading', n_jobs=6):
    # TO DO: PYOD 
    if X.shape[1] > 1e6:
        kargs = {'tol': 5e-4, 'cache_size': 10000, 'kernel': 'linear' }
    elif X.shape[1] > 1e5:
        kargs = {'tol': 5e-5, 'cache_size': 5000, 'kernel': 'linear' }
    else:
        kargs = {'tol': 1e-5, 'cache_size': 300, 'kernel': 'rbf', 'gamma': 0.1 }
    algorithms = {
        'RConvar': EllipticEnvelope(contamination=outliers_fraction, random_state=seed),
        'ocSVM': svm.OneClassSVM(nu=outliers_fraction, **kargs), # slow
        'SGDosSVM': make_pipeline(
                        Nystroem(gamma=0.1, random_state=seed, n_components=150),
                        SGDOneClassSVM(
                            nu=outliers_fraction,
                            shuffle=True,
                            fit_intercept=True,
                            random_state=seed,
                            max_iter = 2000,
                            tol=1e-6),),
        'IsoForest':IsolationForest(contamination=outliers_fraction, random_state=seed),
        'localFactor': LocalOutlierFactor(n_neighbors=n_neighbors, contamination=outliers_fraction),
    }

    def model_pre(Xs, k, model):
        model.fit(Xs)
        if k == "localFactor":
            y_pred = model.fit_predict(Xs)
        else:
            y_pred = model.fit(Xs).predict(Xs)
        return y_pred

    if n_jobs == 1:
        predicts = []
        for k, model in algorithms.items():
            y_pred = model_pre(X, k, model)
            predicts.append(y_pred)
    else:
         predicts = Parallel( n_jobs= n_jobs, backend=backend)(
                         delayed(model_pre)(X, k, model) for k, model in algorithms.items())
    predicts = np.r_[predicts].T
    smodel = algorithms['SGDosSVM']
    Y = smodel.fit(predicts).predict(predicts)
    return Y