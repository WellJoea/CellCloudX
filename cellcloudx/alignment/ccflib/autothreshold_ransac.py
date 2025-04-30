import numpy as np
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt
from warnings import warn

from ...transform import homotransform_point

def atransac(src_pts, dst_pts, model_class,
                data_weight=None,
                min_samples=50, residual_threshold =1., 
                maxiter=50, max_trials=500,
                seed=200504, CI=0.95, 
                stop_merror=1e-3,stop_derror=1e-3, stop_hits=2,
                init_ratio = 0.75,
                stop_pairs=40, kernel='norm',
                drawhist=False, verbose=1, **kargs):
    src_ptw = src_pts.copy()
    dst_ptw = dst_pts.copy()
    D = src_pts.shape[1]
    maxiter = 50 if maxiter is None else maxiter
    if not residual_threshold is None:
        assert 0<residual_threshold <=1

    model_record = np.eye(D+1)

    Inliers = np.arange(len(src_pts))
    multsig = sci.stats.norm.ppf((1+CI)/2 , 0, 1)
    dist = np.linalg.norm(src_pts - dst_pts, axis=1)
    threshold = np.quantile(dist, init_ratio)

    distmean = dist.mean()
    # threshold = Invervals(dist, CI=CI, kernel=kernel, tailed ='two')[2]
    # threshold *= residual_threshold
    stop_counts = 0

    if verbose:
        print('max_iter, num_trials, pairs, threshold, model_error, mes_error')
    for i in range(maxiter):
        model, inbool, num_trials = ransac(
                (src_pts, dst_pts),
                model_class, 
                threshold,
                min_samples=min_samples,
                max_trials=max_trials,
                data_weight=data_weight,
                rng=seed,
                **kargs )

        n_inliers = np.sum(inbool)
        if (n_inliers <= stop_pairs):
            break
        residuals = model.residuals(src_pts, dst_pts)
        Inliers = Inliers[inbool]
        model_final = model_class(dimensionality=D)
        model_final.estimate(src_ptw[Inliers], dst_ptw[Inliers])
        dst_ptf = homotransform_point( dst_ptw, model_final, inverse=True)
        distNmean = (np.linalg.norm(src_ptw[Inliers] - dst_ptf[Inliers], axis=1)).mean()
        # sigma2 = np.sum(np.square(src_ptw[Inliers] - dst_ptf[Inliers]))/(D * len(Inliers))
        # disterror = (distmean - distNmean)/distmean
        disterror = np.abs(distmean - distNmean)
        merror = np.abs(np.array(model)-np.array(model_record)).mean()

        if verbose:
            print(f'{i :<3}, {num_trials :<3}, {len(inbool)} -> {n_inliers}, {threshold :.4e}, '
                  f'{merror :.4e}, {disterror :.4e} ')

        if  (merror <= stop_merror) or ( np.abs(disterror) <=stop_derror):
            stop_counts += 1
            if (stop_counts >=stop_hits):
                break

        model_record = model
        distmean = distNmean

        norm_thred = sci.stats.norm.ppf(CI, np.mean(residuals), np.std(residuals))
        sigma_thred = np.mean(residuals) + multsig*np.std(residuals)
        threshold = np.mean([norm_thred, sigma_thred]) *residual_threshold

        # threshold = Invervals(residuals, CI=CI, kernel=kernel, tailed ='two')[2]
        # threshold *= residual_threshold

        if drawhist:
            fig, ax=plt.subplots(1,1, figsize=(5,5))
            ax.hist(residuals, bins=50)
            ax.axvline(norm_thred, color='red')
            ax.axvline(sigma_thred, color='blue')
            ax.axvline(threshold, color='black')
            plt.show()

        # assert n_inliers >= stop_pairs, f'nn pairs is lower than {stop_pairs}.'
        dst_ptn = homotransform_point( dst_pts, model, inverse=True) #model(src_pts)
        src_pts = src_pts[inbool]
        dst_pts = dst_ptn[inbool]

        if not data_weight is None:
            data_weight = data_weight[inbool]

    print(f'final match pairs: {len(Inliers)}')
    return Inliers, model_final


'''adapted from skimage ransac'''
def ransac(
    data,
    model_class,
    residual_threshold,
    min_samples=20,
    data_weight=None,
    is_data_valid=None,
    is_model_valid=None,
    max_trials=200,
    stop_ratio=0.95,
    stop_mse=1e-3,
    stop_probability=0.9999,
    rng=None,
    initial_inliers=None,
):

    best_inlier_num = 0
    best_inlier_mse = np.inf
    best_inliers = []
    validate_model = is_model_valid is not None
    validate_data = is_data_valid is not None

    D = data[0].shape[1]
    rng = np.random.default_rng(rng)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data,)
    num_samples = len(data[0])

    if not (0 < min_samples <= num_samples):
        raise ValueError(f"`min_samples` must be in range (0, {num_samples}]")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError(
            f"RANSAC received a vector of initial inliers (length "
            f"{len(initial_inliers)}) that didn't match the number of "
            f"samples ({num_samples}). The vector of initial inliers should "
            f"have the same length as the number of samples and contain only "
            f"True (this sample is an initial inlier) and False (this one "
            f"isn't) values."
        )
    if not data_weight is None:
        data_weight = data_weight/data_weight.sum()
    # for the first run use initial guess of inliers
    spl_idxs = (
        initial_inliers
        if initial_inliers is not None
        else rng.choice(num_samples, min_samples, replace=False, p=data_weight)
    )

    # estimate model for current random sample set
    model = model_class(dimensionality=D)

    num_trials = 0
    # max_trials can be updated inside the loop, so this cannot be a for-loop
    while num_trials < max_trials:
        num_trials += 1

        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]

        # for next iteration choose random sample set and be sure that
        # no samples repeat
        spl_idxs = rng.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if validate_data and not is_data_valid(*samples):
            continue

        success = model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if validate_model and not is_model_valid(model, *samples):
            continue

        residuals = np.abs(model.residuals(*data))
        # consensus set / inliers
        inliers = residuals < residual_threshold
        mse = np.mean(residuals)


        # choose as new best model if number of inliers is maximal
        inliers_count = np.count_nonzero(inliers)
        if (
            # more inliers
            inliers_count > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (
                inliers_count == best_inlier_num
                and mse < best_inlier_mse
            )
        ):
            best_inlier_num = inliers_count
            best_inlier_mse = mse
            best_mse = np.mean(residuals[inliers])
            best_inliers = inliers
            max_trials = min(
                max_trials,
                _dynamic_max_trials(
                    best_inlier_num, num_samples, min_samples, stop_probability
                ),
            )
            if (
                best_inlier_num/num_samples >= stop_ratio
                or best_mse <= stop_mse
            ):
                break

    # estimate final model using all inliers
    if any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        if validate_model and not is_model_valid(model, *data_inliers):
            warn("Estimated model is not valid. Try increasing max_trials.")
    else:
        model = None
        best_inliers = None
        warn("No inliers found. Model not fitted")

    return model, best_inliers, num_trials

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability, _EPSILON = np.spacing(1)):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.
    """
    if probability == 0:
        return 0
    if n_inliers == 0:
        return np.inf
    inlier_ratio = n_inliers / n_samples
    nom = max(_EPSILON, 1 - probability)
    # denom = 1- max(_EPSILON, inlier_ratio**(min_samples*probability))
    denom = max(_EPSILON, 1- max(_EPSILON, inlier_ratio**min_samples))
    return np.ceil(np.log(nom) / np.log(denom))