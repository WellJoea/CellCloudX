from typing import Optional
import numpy as np

class GaussTransform_strict(object):
    def __init__(self, source, h, xp=None, device=None, floatx=None, nsplit=None ):
        if hasattr(source ,'detach'):
            import torch as th
            xp = th
        else:
            xp = np if xp is None else xp
        self.nsplit = 1000 if (xp.__name__ == 'torch') and (nsplit is None) else nsplit
        self.xp = xp
        _source = xp.asarray(source)
        self.floatx = _source.dtype if floatx is None else eval(f'xp.{floatx}')
        self.device = _source.device if device is None else device

        self._source = xp.asarray(_source, dtype=self.floatx, device=self.device)
        self._h = h
        self._m = source.shape[0]

    def compute(self, target, weights=None):
        if weights is None:
            weights = np.ones(self._m)
        target = self.xp.asarray(target, dtype=self.floatx, device=self.device)
        weights = self.xp.asarray(weights, dtype=self.floatx, device=self.device)
        return gausstransform_direct(self._source, target, weights, self._h, xp=self.xp, nsplit=self.nsplit)

def gausstransform_direct(source, target, weights, h, xp=np, nsplit=None): # TODO use keops  
    r""" Calculate Gauss Transform
    source : M,D
    target : N,D
    weights: X,M
    out: N,X
    h : beta
    \sum_{j} weights[j] * \exp{ - \frac{||target[i] - source[j]||^2}{h^2} }
    """
    # 计算高斯变换
    h2 = h * h
    fn1 = lambda t: weights @ xp.exp(-xp.sum(xp.square(t - source), 1) / h2)
    fn2 = lambda t: weights @ xp.exp(-xp.sum(xp.square(t[None,...] - source[:,None,...]), -1) / h2)
    if xp == np:
        if nsplit:
            return np.hstack([ fn2(x_i) for x_i in xp.array_split(target, nsplit)]).T
        else:
            return np.apply_along_axis(fn1, 1, target)
    else:
        if nsplit:
            return xp.hstack([ fn2(x_i) for x_i in xp.chunk(target, nsplit) ]).T
        else:
            return xp.stack([ fn1(x_i) for x_i in xp.unbind(target, dim=0) ], dim=0)

class GaussTransform(object):
    """Calculate Gauss Transform

    Args:
        source (numpy.ndarray): Source data.
        h (float): Bandwidth parameter of the Gaussian.
        eps (float): Small floating point used in Gauss Transform.
        sw_h (float): Value of the bandwidth parameter to
            switch between direct method and IFGT.
    """

    def __init__(self, source, h: float, eps: float = 1.0e-10, sw_h: float = 0.01, use_strict=None, **kargs):
        self._m = source.shape[0]
        self.use_strict = use_strict or (h < sw_h)
        if self.use_strict:
            self._impl = GaussTransform_strict(source, h,**kargs)
        else:
            from . import _ifgt
            self._impl = _ifgt.Ifgt(self.to_numpy(source), h, eps)

    def to_numpy(self, X):
        try:
            return X.detach().cpu().numpy()
        except:
            return np.asarray(X)

    def compute(self, target, weights = None):
        """Compute gauss transform

        Args:
            target (numpy.ndarray): Target data.
            weights (numpy.ndarray): Weights of Gauss Transform.
        """
        if weights is None:
            weights = np.ones(self._m)
        target = self.to_numpy(target)
        weights = self.to_numpy(weights)
        if not self.use_strict:
            if weights.ndim == 1:
                return self._impl.compute(target, weights)
            else:
                return np.r_[[self._impl.compute(target, w) for w in weights]].T
        else:
            return self._impl.compute(target, weights)


class GaussTransform_fgt(object):
    """Calculate Gauss Transform

    Args:
        source (numpy.ndarray): Source data.
        h (float): Bandwidth parameter of the Gaussian.
        eps (float): Small floating point used in Gauss Transform.
        sw_h (float): Value of the bandwidth parameter to
            switch between direct method and IFGT.
    """

    def __init__(self, source, h: float, eps: float = 1.0e-10, sw_h: float = 0.01,  **kargs):
        self._m = source.shape[0]
        from . import _ifgt
        self._impl = _ifgt.Ifgt(source, h, eps)
        self.weights = np.ones(self._m)

    def compute(self, target, weights = None):
        """Compute gauss transform

        Args:
            target (numpy.ndarray): Target data.
            weights (numpy.ndarray): Weights of Gauss Transform.
        """
        if weights is None:
            weights = self.weights

        if weights.ndim == 1:
            return self._impl.compute(target, weights)
        else:
            return np.r_[[self._impl.compute(target, w) for w in weights]].T