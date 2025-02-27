from scipy.ndimage import distance_transform_edt,binary_dilation, generate_binary_structure
from scipy.interpolate import interpn, RBFInterpolator
import skimage as ski
from typing import Optional, Union
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from ..utilis._arrays import list_iter

def pcd_grid(position, spacing = np.array([30., 30., 30.])):
    import numpy as np
    from scipy import spatial
    
    ndim = position.shape[1]
    spacing = np.array(spacing)[:ndim]

    xyzs = []
    for i in range(ndim):
        imin = position[:,i].min()
        imax = position[:,i].max()
        ispc = spacing[i]
        ix=np.arange(imin, imax+ispc,ispc).clip(imin, imax)
        xyzs.append(ix)
    xyzs2=np.meshgrid(*xyzs,indexing='ij')
    all_grid=np.array([ix.flatten() for ix in xyzs2 ]).T

    point_tree = spatial.cKDTree(all_grid)
    dist, index = point_tree.query(position)
    return all_grid, dist, index

def interp_pcd_grid(Points, bins = None, space = None, include_edge=True, return_bins=True):
    ndim = Points.shape[1]
    space = list_iter(space)
    bins = list_iter(bins)
    grids = []
    grids_center = []
    for i in range(ndim):
        irange = Points[:, i]
        ispace = space[i]
        ibin = bins[i]
        istart = np.min(irange)
        iend = np.max(irange)
        if not ibin is None:
            if include_edge:
                i_cent = np.linspace(istart, iend, ibin, endpoint=True)
            else:
                i_cent = np.linspace(istart, iend, ibin+1, endpoint=True)
                i_cent = (i_cent[1:]+i_cent[:-1])/2, 

        elif not ispace is None:
            if include_edge:
                i_cent = np.arange(istart, iend, ispace).clip(istart, iend)
            else:
                i_cent = np.arange(istart+ispace/2, iend+ispace/2, ispace).clip(istart, iend)
            i_cent = np.unique(i_cent)

        elif ispace is None:
            i_cent = np.unique(irange)
            ispace =  np.diff(i_cent).mean()

        i_bins = np.r_[ i_cent[0]-ispace,
                        (i_cent[1:]+i_cent[:-1])/2, 
                        i_cent[-1] + ispace ]
        assert len(i_bins) - len(i_cent) == 1
        grids.append(i_bins)
        grids_center.append(i_cent)

    cent_cpos = localize_points(Points, grids, grids_center=grids_center)
    # cent_idxstr = np.array(list(map(lambda x:'-'.join(x),  cent_index.astype(str))))

    # cent_cpos = pd.DataFrame(cent_cpos, columns=list('xyz'), index=cent_idxstr)
    # cent_cpos[['x_pos', 'y_pos', 'z_pos']] = Points
    # cent_cpos['xyz_inds'] = cent_idxstr
    return cent_cpos

def interp_shape_grid(ranges, bins = None, space = 150, include_edge=True, return_grid=True):
    ndim = len(ranges)
    space = list_iter(space)
    bins = list_iter(bins)
    grids = []
    for i in range(ndim):
        irange = ranges[i]
        ispace = space[i]
        ibin = bins[i]
        istart = irange[0]
        iend = irange[1]
        if not ibin is None:
            if include_edge:
                i_cent = np.linspace(istart, iend, ibin, endpoint=True)
            else:
                i_cent = np.linspace(istart, iend, ibin+1, endpoint=True)
                i_cent = (i_cent[1:]+i_cent[:-1])/2, 

        elif not ispace is None:
            if include_edge:
                i_cent = np.arange(istart, iend, ispace).clip(istart, iend)
            else:
                i_cent = np.arange(istart+ispace/2, iend+ispace/2, ispace).clip(istart, iend)
            i_cent = np.unique(i_cent)

        grids.append(i_cent)

    mgrid = np.meshgrid(*grids)
    if return_grid:
        return mgrid
    else:
        Points = np.vstack([ igd.ravel() for igd in mgrid ]).T
        return Points

def localize_points(Points, grids, grids_center=None, extend_outside=False):
    ndim = Points.shape[1]
    assert len(grids) == ndim
    if not grids_center is None:
        assert len(grids_center) == ndim

    cent_index = []
    cent_cpos  = [] 
    for i in range(ndim):
        irange = Points[:, i]
        i_bins = grids[i]

        if extend_outside:
            if np.min(irange) < np.min(i_bins):
                i_bins =  np.r_[np.min(irange), i_bins]
            if  np.max(irange) > np.max(i_bins):
                i_bins =  np.r_[i_bins,  np.max(irange)]
        else:
            if (np.max(irange) > np.max(i_bins)) or  (np.min(irange) < np.min(i_bins)):
                raise('points is range out of grids. Please set extend_outside=True or reset grids range.')

        if not grids_center is None:
            i_cent = grids_center[i]
        else:
            i_cent = (i_bins[1:]+i_bins[:-1])/2

        i_inds = np.digitize(irange, i_bins, right=False)
        i_cpos = i_cent[i_inds-1]
        cent_index.append(i_inds)
        cent_cpos.append(i_cpos)
    cent_index = np.c_[cent_index].T
    cent_cpos = np.c_[cent_cpos].T
    return np.c_[cent_cpos, cent_index]

def interp_pcd_zslice(point3d, axis=2, n_interp=3, 
                      interp_frac = None,
                      S=0.5, R =1, thred=0,
                      interp_self = False,
                      down_sample =True, sample_frac=None, sample_scale=1, sample_method = 'open3d',
                      use_bw=True, seed=200504, **kargs):
    zspace = np.unique(point3d[:,axis])
    axes = [0,1,2]
    axes.pop(axis)

    zspace_int = []
    interp_pos = []
    #for i in range(len(zspace)-1):
    n_epochs = len(zspace)-1
    for i in tqdm(range(n_epochs), colour='red'):
        itick = np.linspace(zspace[i], zspace[i+1], n_interp+2).astype(np.float64)
        ispace = np.linspace(0,1, n_interp+2).astype(np.float64) if interp_frac is None else  np.array([0, *interp_frac, 1])
        assert len(itick) == len(ispace), "tick and space must have the same length"

        if not interp_self:
            itick = itick[1:-1]
            ispace = ispace[1:-1]

        pointa = point3d[point3d[:,axis]==zspace[i]][:,axes].copy()
        pointb = point3d[point3d[:,axis]==zspace[i+1]][:,axes].copy()

        imageo, pointxy = interp_point(pointa.copy(), pointb.copy(), 
                                precision=ispace, S=S,R=R,
                                use_bw=use_bw, thred=thred, **kargs)
        # for isp in ispace:
        #     if isp> 0.5:
        #         imgbp, ipointxy = interp_point(pointb.copy(), pointa.copy(),
        #                                    precision=(1-isp), S=S,R=R,
        #                                         use_bw=use_bw, thred=thred, **kargs)
        #     else:
        #         imgbp, ipointxy = interp_point(pointa.copy(), pointb.copy(), 
        #                                    precision=isp, S=S,R=R,
        #                                         use_bw=use_bw, thred=thred, **kargs)
        #     pointxy.extend(ipointxy)

        pointxyz = []
        size_a, size_b = pointa.shape[0], pointb.shape[0]
        for i, (isp, itc) in enumerate(zip(ispace, itick)):
            ipointxyz = np.c_[pointxy[i],  np.full(pointxy[i].shape[0], itc)]
            size_c = sample_frac or (size_a* (1-isp)  + size_b*isp) * sample_scale
            if (down_sample and size_c<ipointxyz.shape[0]):
                ipointxyz = down_sample_point(ipointxyz, samples=size_c,
                                              k_points=None, method =sample_method,
                                              seed=seed)
            
            ipointxyz = np.c_[ipointxyz, np.full(ipointxyz.shape[0], isp)]
            pointxyz.append(ipointxyz)
        pointxyz = np.vstack(pointxyz)

        interp_pos.append(pointxyz)
        zspace_int.extend(itick)

    interp_pos = np.vstack(interp_pos)
    # if not interp_self:
    #     interp_pos = np.concatenate([point3d, interp_pos], axis=0)
    return interp_pos

def interp_point(pointA, pointB, precision=0, S=1, R =1,  HW=None, use_bw=True, thred=0, order=2, 
               csize = 5, dsize=5, method='circle',**kargs):
    S,R  = float(S), float(R)
    if HW is None:
        HW = np.c_[pointA.max(0), pointB.max(0)].max(1)[:2]
    else:
        HW = np.array(HW)
    HW = np.int64( np.ceil(HW *S) + 10 )

    # Ap = np.int64(np.round(pointA*S))
    # Bp = np.int64(np.round(pointB*S))

    # imageA = np.zeros(HW, dtype=np.int64)
    # imageB = np.zeros(HW, dtype=np.int64)

    # imageA[Ap[:,0], Ap[:,1]] = 1
    # imageB[Bp[:,0], Bp[:,1]] = 1

    imageA = points2bmask(pointA*S, shape = HW, rgbcol=1, rgbbg=0, csize = csize, dsize=dsize, method=method)
    imageB = points2bmask(pointB*S, shape = HW, rgbcol=1, rgbbg=0, csize = csize, dsize=dsize, method=method)

    imageo = interp_shape(imageA, imageB, precision, use_bw=use_bw, **kargs)
    imageo = [ ski.transform.rescale(img, R/S, order=order) for img in imageo]
    points = [ np.c_[np.where(img>thred)]/R for img in imageo]

    # imageo = [ ski.transform.rescale(img>thred, R/S, order=0, anti_aliasing=False) for img in imageo]
    # points = [ np.c_[np.where(img>0)]/R for img in imageo]
    return imageo, points

def interp_shape(top, bottom, precision=0, zscale = 2, use_bw=True, interp_method='linear',
                 kernel = 'regular', n_job =2, use_interp=True,
                 iterations=1, structure=[2,2]):
    '''
    Interpolate between two contours

    Input: top
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if isinstance(precision, (float, int)):
        precision = [precision]
    precision = np.array(precision)
    assert np.all(precision>=0) and np.all(precision<=1), "Precision must be between 0 and 1 (float)"
    precision = np.array(precision)*zscale

    if use_bw:
        top = signed_bwdist(top, iterations=iterations, structure=structure)
        bottom = signed_bwdist(bottom, iterations=iterations, structure=structure)
    r, c = top.shape
    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # interpolation
    if use_interp:
        points = (np.r_[0, zscale], np.arange(r), np.arange(c))
        xs = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))

        def interp_N_grid(points, values, r, c, xs, zpos,  method='linear'):
            xi = np.c_[np.full((r*c), zpos), xs].copy()
            out = interpn(points, values, xi, method=method)
            #yflat = RBFInterpolator(xobs, yobs)(xflat)
            out = out.reshape((r, c))
            return out
        outs = Parallel( n_jobs=n_job, verbose=0)( delayed( interp_N_grid)
                                (points, top_and_bottom, r,c, xs, zpos, method=interp_method) 
                            for zpos in precision)
    else:
        outs = []
        for zpos in precision:
            out = top*(1-zpos) + bottom*zpos
            outs.append(out)
    return outs

def interp_point3d(points, interp_frac=0.5, S=1, R =1,  HW=None, use_bw=True, 
                    down_sample =True, sample_frac=None, sample_method = 'open3d', seed=200504, 
                   thred=0, order=2, verbose=1, **kargs):
    '''
    Input: points:
    [X, Y, Z]
    '''
    if isinstance(interp_frac, (float, int)):
        interp_frac = np.array([interp_frac])

    S,R  = float(S), float(R)
    if HW is None:
        HW = points.max(0)[:2]
    else:
        HW = np.array(HW)
    HW = np.int64( np.ceil(HW *S) + 10 )

    zaxes = np.unique(points[:,2])
    zaxes_space= zaxes[1:] - zaxes[:-1]
    D = len(zaxes)

    images = np.zeros((D,*HW), dtype=np.int64)
    points_size = []
    for idx, iz in enumerate(zaxes):
        ipxy = points[points[:,2]==iz,:2]
        ipxy = np.int64(np.round(ipxy*S))
        images[idx, ipxy[:,0], ipxy[:,1]] = 1
        points_size.append(ipxy.shape[0])

    imageo = interp_shape3d(images, precision=interp_frac, use_bw=use_bw, out_size=False, **kargs)

    pointps = []
    for idx, iprc in enumerate(interp_frac):
        izspace = zaxes[:-1] + zaxes_space * iprc
        imgintp = imageo[idx]
        ipointps = []
        for islc, (izs, imgb) in enumerate(zip(izspace, imgintp)):
            imgbp = ski.transform.rescale(imgb, R/S, order=order)
            ipointp = np.c_[np.where(imgbp>thred)]/R
            # imgbp = ski.transform.rescale(imgb>thred, R/S, order=0, anti_aliasing=False)
            # ipointp = np.c_[np.where(imgbp>0)]/R

            p_size  = ipointp.shape[0]
            ipointp = np.c_[ ipointp, np.full((p_size),izs) ]
            verbose and print(f'inter_frac:{iprc}->slice {islc}-> size {p_size}')

            if down_sample:
                size_a, size_b = points_size[islc], points_size[islc+1]
                size_c = sample_frac or (size_a* (1-iprc)  + size_b*iprc)
                ipointp = down_sample_point(ipointp, samples=size_c,
                                              k_points=None, method =sample_method,
                                              seed=seed)
                verbose and print(f'size {p_size}->down sample {ipointp.shape[0]}')
            ipointps.append(ipointp)
        ipointps = np.vstack(ipointps)
        pointps.append(ipointps)
    pointps = np.vstack(pointps)
    return pointps, imageo

def interp_shape3d(imgbs, precision=0.5, use_bw=True, out_size=False, interp_method='linear',
                   iterations=1, structure=[2, 2]):
    '''
    Interpolate between two contours

    Input: imgbs
            [Z, X,Y] - Image of top contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [Z,X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if isinstance(precision, (float, int)):
        precision = np.array([precision])

    precision = np.array(precision)
    assert np.all(precision >= 0) and np.all(precision <= 1), "Precision must be between 0 and 1 (float)"

    d, r, c = imgbs.shape
    if use_bw:
        imgbm = []
        for img in imgbs:
            img = signed_bwdist(img, iterations=iterations, structure=structure)
            imgbm.append(img)
        imgbs = np.array(imgbm)
        del imgbm

    # create ndgrids
    points = (np.arange(d), np.arange(r), np.arange(c))
    D = d if out_size else d-1
    xs = np.rollaxis(np.mgrid[:D, :r, :c], 0, 4).reshape((D * r * c, -1))
    didx = np.arange(D).astype(np.float64)

    outs = []
    didc = []
    for iperc in precision: # large data
        xi = xs.copy()
        xi[:, 0] = xi[:, 0] + iperc
        # Interpolate for new plane
        out = interpn(points, imgbs, xi, method=interp_method)
        out = out.reshape((D, r, c))

        # Threshold distmap to values above 0
        # out = out > 0
        outs.append(out)
        didc.append(didx+iperc)
    return outs

def down_sample_point(POINTS, samples=None, k_points=None, method ='open3d', seed=0):
    POINTS = np.array(POINTS)
    n_point = POINTS.shape[0]
    if not samples is None:
        samples = int(samples)

    if method=='random':
        np.random.seed(seed)
        obs_idex = np.random.choice(np.arange(n_point), 
                         size=samples, 
                         replace=False, p=None)
        return POINTS[obs_idex]
    elif  method=='open3d':
        import open3d as o3d
        k_points = k_points or int(n_point/samples)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(POINTS)

        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        uni_down_pcd = pcd.uniform_down_sample(every_k_points=k_points)
        down_points = np.asarray(uni_down_pcd.points)
        return down_points
    elif method == 'pyvista':
        import pyvista as pv
        tolerance =  n_point/samples
        pld = pv.PolyData(POINTS)
        dpld = pld.clean(
                point_merging=True,
                merge_tol=tolerance,
                lines_to_points=False,
                polys_to_lines=False,
                strips_to_polys=False,
                inplace=False,
                absolute=False,
                progress_bar=False)
        down_points = np.asarray(dpld.points)
        return down_points

def bwperim(bw, n=4, iterations=2, structure=[2,2]):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    # if n not in (4,8):
    #     raise ValueError('mahotas.bwperim: n must be 4 or 8')
    # print(bw.shape)
    # rows,cols = bw.shape
    #
    # # Translate image by one pixel in all direct
    # # ions
    # north = np.zeros((rows,cols))
    # south = np.zeros((rows,cols))
    # west = np.zeros((rows,cols))
    # east = np.zeros((rows,cols))
    #
    # north[:-1,:] = bw[1:,:]
    # south[1:,:]  = bw[:-1,:]
    # west[:,:-1]  = bw[:,1:]
    # east[:,1:]   = bw[:,:-1]
    # idx = (north == bw) & \
    #       (south == bw) & \
    #       (west  == bw) & \
    #       (east  == bw)
    # plt.imshow(idx.astype(int))
    # plt.title('move')
    # plt.show()
    #
    # if n == 8:
    #     north_east = np.zeros((rows, cols))
    #     north_west = np.zeros((rows, cols))
    #     south_east = np.zeros((rows, cols))
    #     south_west = np.zeros((rows, cols))
    #     north_east[:-1, 1:]   = bw[1:, :-1]
    #     north_west[:-1, :-1]  = bw[1:, 1:]
    #     south_east[1:, 1:]    = bw[:-1, :-1]
    #     south_west[1:, :-1]   = bw[:-1, 1:]
    #     idx &= (north_east == bw) & \
    #            (south_east == bw) & \
    #            (south_west == bw) & \
    #            (north_west == bw)
    # return ~idx * bw
    # struct1 = generate_binary_structure(2, 1)
    struct = generate_binary_structure(*structure)
    return binary_dilation(bw, iterations=iterations, structure=struct, border_value=0) - bw

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im
# https://stackoverflow.com/questions/48818373/interpolate-between-two-images

def signed_bwdist(im, iterations=2, structure=[2,2]):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    imbp = bwperim(im.copy(), iterations=iterations, structure=structure)
    imbd = bwdist(imbp)
    ima = -imbd * np.logical_not(im) + imbd*im
    #@cc.pl.qview(im, imbp, imbd, ima, titles=('raw', 'diff_neib', 'dist', 'masked'), show=True)
    return ima

def points2bmask(posn, shape=None, rgbcol=(255, 255, 255), rgbbg=(0,0,0), 
               csize = 5, dsize=5,# thresh=127, maxval=255,
               method='circle', margin = [0, 0]):

    ipos = np.int64(np.round(posn))
    if shape is None:
        shape = ipos.max(0).astype(np.int64) + 1
    h, w = shape
    h += margin[0]
    w += margin[1]

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
        img = img.astype(np.int32).copy()

        for i in ndix:
            cv2.circle(img, (int(ipos[i][0]), int(ipos[i][1])), csize, val[i], -1)
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
