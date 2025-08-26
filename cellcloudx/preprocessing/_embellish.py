import numpy as np
import pandas as pd
import skimage as ski
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def img_similarity(img1, img2, match_hist=False, similar='ssmi'):
    if match_hist:
        matched = match_histograms(img2, img1, channel_axis=None)
    else:
        matched = img2
    if similar=='ssmi':
        return ski.metrics.structural_similarity(img1, matched, data_range=1 - img1.min())
    elif similar=='nmi':
        return ski.metrics.normalized_mutual_information(img1, matched)
    else:
        return np.nan
        #ski.metrics.structural_similarity(img1, matched, data_range=matched.max() - matched.min())
        #ski.metrics.normalized_mutual_information(img1, img2)
        #ski.metrics.normalized_root_mse(img2, img1) ##nonsym
        #ski.metrics.variation_of_information(img1, img2),
        #ski.metrics.mean_squared_error(img1, img2),
        #ski.metrics.peak_signal_noise_ratio(img1, img2),
        #ski.metrics.adapted_rand_error(img1, img2)
        #ski.metrics.contingency_table(img1, img2)

def maskbg(iimg,
            clips=None,
            peak=None,
            error=0.05,
            bgcolor=0,
            layout='rgb',
            peak_type = 'max',
            show_peak = True,
            figsize_peak=(20,5),
            figsize=(5,20),
            peak_range=None,
            bins=200,
            show=True):

    if iimg.ndim ==2:
        nlayer = 1
        iimg = iimg[...,None]

    if iimg.ndim ==3:
        nlayer = iimg.shape[2]

    if clips is None:
        if peak is None:
            peak= get_force_peak(iimg.copy(), layout=layout,bins=bins,
                                    peak_range=peak_range, peak_type=peak_type, figsize=figsize_peak, show=show_peak)
        if show_peak:
            print(peak)
        
        if error is None:
            error = 0.05
        if isinstance(error, (int)):
            error = [[-error, error]] * nlayer
        elif isinstance(error, (float)):
            error = [[-error * iimg.max(), error * iimg.max()]] * nlayer
        assert np.array(error).shape == (nlayer, 2)
        clips = np.array(peak)[:,None] + np.array(error)
    
    clips = np.array(clips)
    assert np.array(clips).shape == (nlayer, 2)

    img_np = iimg.copy()
    mask = np.zeros(img_np.shape[:2], dtype=bool)
    for i in range(nlayer):
        imask = (iimg[...,i]>= clips[i,0]) & (iimg[...,i]<= clips[i,1])
        mask = mask | imask
    img_np[mask,] = bgcolor

    img_rk = img_np.copy()
    img_rk[ img_rk != iimg ] = 0
    #img_rk[idx,] = 255
    #img_rk[idx,] = (100,200,255)
    if show:
        fig, axs = plt.subplots(1,2,figsize=figsize)
        axs[0].imshow(mask.astype(np.int64), cmap='gray')
        #axs[0].imshow(img_rk)
        axs[1].imshow(img_np)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        fig.show()
    
    if nlayer == 1:
        img_np = img_np.squeeze(-1)
    return img_np, mask

def get_force_peak(img, layout='rgb', figsize=(20,5), 
                   logyscale=True, max_vule= None, step = None,
                   peak_range= None, bins=200,
                   peak_type = 'max', height = 0, prominence=0.3,
                   bin = None, iterval=None, show=True,):
    #from scipy.signal import find_peaks
    #find_peaks(counts, distance =10, width =3)
    peaks= []

    iimg = img.copy()
    if len(iimg.shape) ==2:
        iimg = iimg[:,:,np.newaxis]
    if np.round(iimg.max())<=1:
        iterval=(0, 1)
        xtick = np.round(np.linspace(0,1,50, endpoint=True), 2)
    else:
        max_vule = max_vule or np.ceil(img.max()).astype(int)
        xtick = np.round(np.linspace(0,max_vule, 50, endpoint=True), 2)
    iimg = iimg[:,:,:]

    fig, ax = plt.subplots(1,1, figsize=figsize)
    for i in range(iimg.shape[2]):
        x = iimg[:,:,i].flatten()
        if peak_range is None:
            peak_range = [x.min(), x.max()]
        counts, values=np.histogram(x, bins=bins, range=iterval)
        midval = (values[:-1] + values[1:])/2
        idx = (midval>=peak_range[0]) & (midval<=peak_range[1])

        midcot = counts[idx]
        midval = midval[idx]
        if peak_type == 'max':
            ipeak = midval[np.argmax(midcot)]
        elif peak_type == 'min':
            ipeak = midval[np.argmin(midcot)]

        # peak, _ = find_peaks(x, height=height, prominence=prominence)
        # from findpeaks import findpeaks
        # fp = findpeaks(method='peakdetect')
        # ipeak = x[peak[peak_idx]]

        peaks.append(ipeak)
        xrange = np.array([values[:-1], values[1:]]).mean(0)
        ax.plot(xrange, counts, label=f"{i} {layout[i]} {ipeak :.3f}", color=layout[i])
        ax.axvline(x=ipeak, color=layout[i], linestyle='-.')

    ax.legend(loc="best")

    ax.set_xticks(xtick)
    ax.set_xticklabels(
        xtick,
        rotation=90, 
        ha='center',
        va='center_baseline',
        fontsize=10,
    )
    if logyscale:
        ax.set_yscale('log')
    #ax.set_axis_on()
    if show:
        plt.show()
    else:
        plt.close()
    return np.array(peaks)

def scaledimg(images):
    if (np.issubdtype(images.dtype, np.integer) or
        (images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])) and \
        (images.max() > 1):
        return False
    else:
        return True

def onechannels(image):
    if image.ndim==3:
        return True
    elif image.ndim==4:
        return False
    else:
        raise ValueError('the image must have 3 or 4_ dims.')

def step_similarity(images, isscale=None, back_for = [3,0], 
                    match_hist=False, similar='ssmi',
                    nascore=1, plot=True):
    if isscale is None:
        isscale = onechannels(images)
    if not isscale:
        images  = ski.color.rgb2gray(images)
    # if images.ndim==4:
    #     images = images[...,0]

    similarity = []
    for i in range(images.shape[0]):
        wds = [*range(i-back_for[0], i) , *range(i, i+back_for[1]+1)]
        simi = []
        for iwd in wds:
            if (iwd < 0) or (iwd == i) or (iwd>=images.shape[0]):
                score = nascore
            else:
                score = img_similarity(images[i], images[iwd], match_hist=match_hist, similar=similar)
            simi.append(round(score, 8))
        similarity.append(simi)
    similarity = pd.DataFrame(similarity,
                              index=np.arange(len(similarity)) +1,
                              columns=list(range(-back_for[0], back_for[1]+1)))

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,1, figsize=(30,7))
        sns.heatmap(similarity.T, cmap = 'viridis', linewidth=.5, ax=ax[0])
        sns.lineplot(data=similarity, ax=ax[1])
        fig.show()

    return similarity