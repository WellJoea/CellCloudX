import numpy as np

def points2img(posn, shape=None, rgbcol=(255, 255, 255), rgbbg=(0,0,0), 
               csize = 5, dsize=5, thresh=127, maxval=255,
               method='circle', margin = 0):
    ipos = np.int64(np.round(posn))
    if shape is None:
        shape = ipos.max(0).astype(np.int64) + 1

    h, w = shape
    img = np.full((h, w, 3), rgbbg, dtype=np.uint8)
    if method =='dilation':
        if isinstance(dsize, int):
            dsize = (dsize, dsize, 1)

        rgbcol = np.array(rgbcol)
        from scipy.ndimage import grey_dilation, grey_erosion
        img[ipos[:,0], ipos[:,1]] = list(rgbcol)
        # img[ipos[:,0]+1, ipos[:,1]+1] = rgbcol
        # img[ipos[:,0], ipos[:,1]+1] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]-1] = rgbcol
        # img[ipos[:,0], ipos[:,1]-1] = rgbcol
        # img[ipos[:,0]-1, ipos[:,1]] = rgbcol
        img = grey_dilation(img, size=dsize).astype(np.uint8)
        # img = grey_erosion(img, size=(5,5,1));

    else:
        import cv2
        ipos = ipos[:,[1,0]]
        for point in ipos:
            cv2.circle(img, point, csize, rgbcol, -1)
    # binary_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary_mask = cv2.threshold(binary_mask, thresh, maxval, cv2.THRESH_BINARY)
    return img


def adata2img(idata, posn, ctcor_dict, ct_col = 'anno_new', size=(5,5,1), margin = 0):
    from scipy.ndimage import grey_dilation, grey_erosion
    posn = posn.copy().round(0) + margin

    h, w = posn.max(0).astype(np.int64) +1 + margin*2
    
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for ict in idata.obs[ct_col].unique():
        idx = (idata.obs[ct_col] == ict).values
        ipos = np.int64(np.round(posn[idx]))
        img[ipos[:,0], ipos[:,1]] = ctcor_dict[ict]
        # img[ipos[:,0]+1, ipos[:,1]+1] = ctcor_dict[ict]
        # img[ipos[:,0], ipos[:,1]+1] = ctcor_dict[ict]
        # img[ipos[:,0]-1, ipos[:,1]] = ctcor_dict[ict]
        # img[ipos[:,0]-1, ipos[:,1]-1] = ctcor_dict[ict]
        # img[ipos[:,0], ipos[:,1]-1] = ctcor_dict[ict]
        # img[ipos[:,0]-1, ipos[:,1]] = ctcor_dict[ict]
    st_img = grey_dilation(img, size=size).astype(np.uint8)
    # st_img = grey_erosion(st_img, size=(5,5,1));

    return st_img