import skimage as ski
import scipy as sci
import numpy as np

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

def points2bmask(posn, maskb=None, rgbcol=(255, 255, 255), rgbbg=(0,0,0), 
               csize = 5, dsize=5, thresh=127, maxval=255, shape=None,
               method='circle', margin = 0):
    ipos = np.int64(np.round(posn))
    if shape is None:
        shape = ipos.max(0).astype(np.int64) + 1
    h, w = shape

    if isinstance( rgbcol, (int, float) ):
        assert  isinstance( rgbbg, type(rgbcol))
        img = np.full((h, w), rgbbg)
        val = [rgbcol] * posn.shape[0]

    elif isinstance( rgbcol, (list, tuple) ) and (3<= len(rgbcol)<=4):
        assert  isinstance( rgbbg, type(rgbcol))
        img = np.full((h, w, len(rgbcol)), rgbbg)
        val = [rgbcol] * posn.shape[0]

        if isinstance(dsize, int):
            dsize = [dsize, dsize, 1, 1][:len(rgbcol)]

    elif len(rgbcol) == posn.shape[0]:
        assert  isinstance( rgbbg, (int, float))
        img = np.full((h, w), rgbbg)
        val = rgbcol
    else:
        raise('error type of rgbcol')

    if method =='dilation':
        val = np.array(val)
        from scipy.ndimage import grey_dilation, grey_erosion
        img[ipos[:,0], ipos[:,1]] = val
        # img[ipos[:,0]+1, ipos[:,1]+1] = rgbcol
        # img[ipos[:,0], ipos[:,1]+1] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]-1] = rgbcol
        # img[ipos[:,0], ipos[:,1]-1] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]] = rgbcol
        img = grey_dilation(img, size=dsize)
        # img = grey_erosion(img, size=(5,5,1));

    else:
        import cv2
        ipos = ipos[:,[1,0]]
        ndix = range(ipos.shape[0])
        for i in ndix:
            cv2.circle(img, ipos[i], csize, val[i], -1)
    return img
    # binary_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary_mask = cv2.threshold(binary_mask, thresh, maxval, cv2.THRESH_BINARY)
    # binary_mask = ski.filters.gaussian(mask1, 30)

def pointsinmask(mask, points):
    pos = np.int64(np.round(points))
    inc = (pos[:,0] < mask.shape[0]) & (pos[:,1]< mask.shape[1] )
    ipos = pos[inc]
    iinc = mask[ipos[:,0], ipos[:,1]].astype(np.bool_)
    inc[inc] = iinc
    return inc

def counts_ratio(x, y, xthred=10, ythred=10, EPS=1, state_method ='raw',):
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
        ifc[ifc<0] = 1/ifc[ifc<0]
        return  ifc

def get_mask(vertices, maplocs, Labels, csize=5, msize=80):
    points = maplocs.copy()
    xy_shift =  vertices.min(0)
    vertices -= xy_shift
    xy_max = vertices.max(0)
    xy_min = vertices.min(0)

    # p_contain = cc.tl.trim_points_from_mesh(P_2dmesh, P_locs)
    points -= xy_shift
    points[:,0] = np.clip(points[:,0], xy_min[0], xy_max[0])
    points[:,1] = np.clip(points[:,1], xy_min[1], xy_max[1])

    y_mean = vertices.min(0)[1] +  np.ptp(vertices, axis=0)[1]/2
    mask1 = points2bmask(vertices, csize=csize, rgbcol=(100,100,100), rgbbg=(255,255,255), )
    mask2 = points2bmask(vertices, csize=msize, rgbcol=(150,150,150), rgbbg=(255,255,255), )
    maskb =  points2bmask(vertices, csize=msize, rgbcol=1, rgbbg=0)
    mask1b = points2bmask(vertices, csize=csize, rgbcol=(255,255,255), rgbbg=(0,0,0), )
    masklr = mask1.copy()
    masklr[:, np.int64(y_mean):] = mask2[:, np.int64(y_mean):]

    contains = pointsinmask(maskb>0, points)
    points = points[contains]
    labels = Labels[contains]

    # cc.pl.qview(mask1, maskb)
    # cc.pl.imagemappoint(masklr, points)
    return mask1, mask2, maskb,mask1b, masklr, points, labels, y_mean

def grid_states(points, mask1, x_is_disc =False, y_is_disc=False, x_space=None, y_space=None):

    def get_bins( points, axis, is_disc, bins=None):
        if is_disc:
            i_sidx = np.unique(points[:, axis])
            i_dist = i_sidx[1:] - i_sidx[:-1]
            i_bins = i_sidx[:-1] + i_dist/2 
            i_bins = np.array([i_sidx[0]- (i_dist/2)[0], *i_bins, i_sidx[-1] + (i_dist/2)[-1] ])
            return i_bins, int(i_dist.mean())
        else:
            i_bins = np.arange(0, mask1.shape[axis]+bins, bins).clip(0, mask1.shape[axis])
            return i_bins

    if x_is_disc or y_is_disc:
        if y_is_disc:
            y_bins, bins = get_bins(points, 1, y_is_disc)
            x_bins = get_bins(points, 0, x_is_disc, bins=bins)
        else:
            x_bins, bins = get_bins(points, 0, x_is_disc)
            y_bins = get_bins(points, 1, y_is_disc, bins=bins)
    else:
        x_bins = get_bins(points, 0, x_is_disc, bins=x_space)
        y_bins = get_bins(points, 1, y_is_disc, bins=y_space)

    x_intp = (x_bins[1:] + x_bins[:-1])/2
    y_intp = (y_bins[1:] + y_bins[:-1])/2

    x_inds = np.digitize(points[:,0], x_bins, right=False)
    y_inds = np.digitize(points[:,1], y_bins, right=False)

    xy_grid = pd.DataFrame(np.c_[points, x_inds, y_inds], columns=['x', 'y', 'x_ibin', 'y_ibin'], dtype=np.float64)
    xy_grid['label'] = pd.Categorical(labels, categories=Order)
    xy_grid[['x_ibin', 'y_ibin']] = xy_grid[['x_ibin', 'y_ibin']].astype(np.int64)
    xy_grid['x_cent'] = x_intp[xy_grid['x_ibin']-1]
    xy_grid['y_cent'] = y_intp[xy_grid['y_ibin']-1]

    xy_grid['xy_bin'] = (xy_grid['x_ibin'].astype(str) + '-' +  xy_grid['y_ibin'].astype(str))
    ratio = xy_grid[['xy_bin', 'label']].value_counts().to_frame('counts').reset_index()
    ratio = ratio.pivot(index='xy_bin', columns='label', values='counts').fillna(0).reset_index()
    ratio.columns.name=None

    xy_grid = xy_grid.merge(ratio, on='xy_bin', how='left')

    ratio[['x_ibin', 'y_ibin']] = ratio['xy_bin'].str.split('-', expand=True).astype(np.int64)
    ratio['x_cent'] = x_intp[ratio['x_ibin']-1]
    ratio['y_cent'] = y_intp[ratio['y_ibin']-1]

    return [xy_grid, ratio, x_bins, y_bins]

def draw_heat( ratio, x_bins, y_bins, y_mean, state_type = 'neuron-glia',  Pn = 'P7', size=15, use_normal = True, use_sym = True, 
              show=True, clips=None , lognormal = False, use_raw= True):
    import scipy as sci
    import matplotlib.pyplot as plt

    # state_type = 'ExN-InN'
    ratio['neuron'] = ratio['ExN']+ratio['InN']
    ratio['glia'] = ratio['OPC'] + ratio['Oligodendrocyte'] +  ratio['Astrocyte'] + ratio['Microglia']

    ratioI = ratio.copy()
    if use_normal:
        for ict in ['neuron', 'glia', 'ExN', 'InN']:
            ratioI[ict] /= ratioI[ict].sum()/1e8

    ratio['ExN-InN'] = counts_ration(ratioI['ExN'], ratioI['InN'], lognormal=lognormal, use_raw=use_raw)
    ratio['neuron-glia'] = counts_ration(ratioI['neuron'], ratioI['glia'], lognormal=lognormal, use_raw=use_raw)


    if clips is None:
        if ratio[state_type].min()<-0.1:
            clip = (-4,4) if lognormal else (-10,10)
        else:
            clip = (0, None)
    else:
        clip = clips

    ccamp = pvcmap if ratio[state_type].min()< 0 else 'viridis'

    if use_sym:
        ratiol =  ratio.copy()
        ratiol['y_cent'] = y_mean*2 -ratiol['y_cent']
        ratioa = pd.concat([ ratio, ratiol], axis=0)

        # cc.pl.imagemappoint(mask1, ratioa[['x_cent','y_cent']].values, 
        #                     color_scale=ratioa['ExN-InN'].values.clip(-10,10), 
        #                     cmap=pvcmap,
        #                     legend='scatter',
        #                     legend_pad = -0.05,
        #                     legend_shift= 0.02,
        #                     legend_width = 0.015,
        #                     legend_height = 0.23,
        #                     legend_color='black',
        #                     legend_size=7,
        #                     alpha=0.8,
        #                     size=15, marker='s', axis_off=True)

        i_sidx = np.unique(ratioa['y_cent'])
        i_dist = i_sidx[1:] - i_sidx[:-1]
        i_bins = i_sidx[:-1] + i_dist/2 
        i_bins = np.array([i_sidx[0]- (i_dist/2)[0], *i_bins, i_sidx[-1] + (i_dist/2)[-1] ])
        i_intp = (i_bins[1:] + i_bins[:-1])/2

        #if Pn=='P4':
        i_intp = np.linspace(ratioa['y_cent'].min(), ratioa['y_cent'].max(), len(i_sidx), endpoint=True, )
        y_intp = (y_bins[1:] + y_bins[:-1])/2
        y_intp = i_intp

        x_intp = (x_bins[1:] + x_bins[:-1])/2
        x_intp = np.linspace(ratioa['x_cent'].min(), ratioa['x_cent'].max(), len(x_intp), endpoint=True, )

        x_cb, y_cb = np.meshgrid(x_intp, y_intp)
        intp_points = np.vstack([x_cb.flatten(), y_cb.flatten()]).T

        point_r = ratioa[['x_cent','y_cent']].values
        value_r = ratioa[state_type].values

        method='cubic' if Pn=='P4' else 'linear'
        intp_value = sci.interpolate.griddata(point_r, value_r, (x_cb, y_cb), method='linear')
        # intp_value1 = sci.interpolate.RBFInterpolator(point_r, value_r, smoothing=0)(intp_points)

        kidx = pointsinmask(maskb>0, intp_points)
        intp_points = intp_points[kidx]
        intp_value = intp_value.flatten()[kidx]
        # intp_value[np.isnan(intp_value)] = 0

        plt.hist(intp_value, bins=50)
        plt.show()
        
        # cc.pl.imagemappoint(mask1, intp_points, 
        #                     color_scale=intp_value.clip(*clip), 
        #                     cmap=ccamp,
        #                     legend='scatter',
        #                     legend_pad = 0.01, #-0.05,
        #                     legend_shift= 0.02,
        #                     legend_width = 0.015,
        #                     legend_height = 0.23,
        #                     legend_color='black',
        #                     legend_size=7,
        #                     alpha=0.8,
        #                     title=f'{Pn}_{state_type}',
        #                     show =show,
        #                     size=size, marker='s', axis_off=True)
    else:
        x_intp = (x_bins[1:] + x_bins[:-1])/2
        y_intp = (y_bins[1:] + y_bins[:-1])/2

        x_cb, y_cb = np.meshgrid(x_intp, y_intp)
        intp_points = np.vstack([x_cb.flatten(), y_cb.flatten()]).T

        point_r = ratio[['x_cent','y_cent']].values
        value_r = ratio[state_type].values

        intp_value = sci.interpolate.griddata(point_r, value_r, (x_cb, y_cb), method='linear')
        # intp_value1 = sci.interpolate.RBFInterpolator(point_r, value_r, smoothing=0)(intp_points)

        kidx = pointsinmask(maskb>0, intp_points)
        intp_points = intp_points[kidx]
        intp_value = intp_value.flatten()[kidx]

        plt.hist(intp_value, bins=50)
        plt.show()

        # cc.pl.imagemappoint(mask1, intp_points, 
        #                     color_scale=intp_value.clip(*clip), 
        #                     cmap=ccamp,
        #                     legend='scatter',
        #                     legend_pad = 0.01,
        #                     legend_shift= 0.02,
        #                     legend_width = 0.015,
        #                     legend_height = 0.23,
        #                     legend_color='black',
        #                     legend_size=7,
        #                     alpha=0.8,
        #                     title=f'{Pn}_{state_type}',
        #                     show =show,
        #                     size=size, marker='s', axis_off=True)
    return [intp_points, intp_value.clip(*clip)]

def draw_heatbrain( ratio, x_bins, y_bins, y_mean, state_type, maskb, cellts, Pn = 'P7', size=15, use_normal = True, 
                   use_frac = True, use_sym = True, 
              show=True, clips=None , lognormal = False, use_raw= True):
    import scipy as sci
    import matplotlib.pyplot as plt
    import pandas as pd
    # state_type = 'ExN-InN'
    if clips is None:
        if ratio[state_type].min()<-0.1:
            clip = (-4,4) if lognormal else (-10,10)
        else:
            clip = (0, None)
    else:
        clip = clips

    if use_frac:
        ratio['cellts'] = ratio[cellts].sum(1)
        ratio[f'{state_type}'] = counts_ration(ratio[state_type], ratio['cellts'], lognormal=lognormal, use_raw=use_raw)

    ccamp = pvcmap if ratio[state_type].min()< 0 else 'viridis'

    if use_sym:
        ratiol =  ratio.copy()
        ratiol['y_cent'] = y_mean*2 -ratiol['y_cent']
        ratioa = pd.concat([ ratio, ratiol], axis=0)

        # cc.pl.imagemappoint(mask1, ratioa[['x_cent','y_cent']].values, 
        #                     color_scale=ratioa['ExN-InN'].values.clip(-10,10), 
        #                     cmap=pvcmap,
        #                     legend='scatter',
        #                     legend_pad = -0.05,
        #                     legend_shift= 0.02,
        #                     legend_width = 0.015,
        #                     legend_height = 0.23,
        #                     legend_color='black',
        #                     legend_size=7,
        #                     alpha=0.8,
        #                     size=15, marker='s', axis_off=True)

        i_sidx = np.unique(ratioa['y_cent'])
        i_dist = i_sidx[1:] - i_sidx[:-1]
        i_bins = i_sidx[:-1] + i_dist/2 
        i_bins = np.array([i_sidx[0]- (i_dist/2)[0], *i_bins, i_sidx[-1] + (i_dist/2)[-1] ])
        i_intp = (i_bins[1:] + i_bins[:-1])/2

        #if Pn=='P4':
        i_intp = np.linspace(ratioa['y_cent'].min(), ratioa['y_cent'].max(), len(i_sidx), endpoint=True, )
        y_intp = (y_bins[1:] + y_bins[:-1])/2
        y_intp = i_intp

        x_intp = (x_bins[1:] + x_bins[:-1])/2
        x_intp = np.linspace(ratioa['x_cent'].min(), ratioa['x_cent'].max(), len(x_intp), endpoint=True, )

        x_cb, y_cb = np.meshgrid(x_intp, y_intp)
        intp_points = np.vstack([x_cb.flatten(), y_cb.flatten()]).T

        point_r = ratioa[['x_cent','y_cent']].values
        value_r = ratioa[state_type].values

        method='cubic' if Pn=='P4' else 'linear'
        intp_value = sci.interpolate.griddata(point_r, value_r, (x_cb, y_cb), method='linear')
        # intp_value1 = sci.interpolate.RBFInterpolator(point_r, value_r, smoothing=0)(intp_points)

        kidx = pointsinmask(maskb>0, intp_points)
        intp_points = intp_points[kidx]
        intp_value = intp_value.flatten()[kidx]
        # intp_value[np.isnan(intp_value)] = 0

        # plt.hist(intp_value, bins=50)
        # plt.show()
    else:
        x_intp = (x_bins[1:] + x_bins[:-1])/2
        y_intp = (y_bins[1:] + y_bins[:-1])/2

        x_cb, y_cb = np.meshgrid(x_intp, y_intp)
        intp_points = np.vstack([x_cb.flatten(), y_cb.flatten()]).T

        point_r = ratio[['x_cent','y_cent']].values
        value_r = ratio[state_type].values

        intp_value = sci.interpolate.griddata(point_r, value_r, (x_cb, y_cb), method='linear')
        # intp_value1 = sci.interpolate.RBFInterpolator(point_r, value_r, smoothing=0)(intp_points)

        kidx = pointsinmask(maskb>0, intp_points)
        intp_points = intp_points[kidx]
        intp_value = intp_value.flatten()[kidx]

        # plt.hist(intp_value, bins=50)
        # plt.show()
    return [intp_points, intp_value.clip(*clip)]