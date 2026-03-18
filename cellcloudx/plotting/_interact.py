import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..utilis import take_data

def compare_intr(fixings, movings, axis=0, share_slice=False, cmap=None, iarkgs = {}):
    from ipywidgets import interact,fixed
    interact(
        comp_images,
        fixed_npa=fixed(fixings),
        moving_npa=fixed(movings),
        fixed_image_z=(0, fixings.shape[axis] - 1),
        moving_image_z= fixed(None) if share_slice else (0, movings.shape[axis] - 1) ,
        axis=fixed(axis),
        cmap=fixed(cmap),
        #layout=widgets.Layout(width='50%')
        **iarkgs
    )

def qview_intr(images, axis=0, cmap=None, figsize=(10,8),iarkgs={}):
    from ipywidgets import interact,fixed
    interact(
        imshows,
        image_z=(0, images.shape[axis] - 1),
        images=fixed(images),
        axis=fixed(axis),
        cmap=fixed(cmap),
        figsize=fixed(figsize),
        **iarkgs
    )

def imagemappoint_intr(images, points, axis=0, iarkgs={}, **kargs):
    from ipywidgets import interact,fixed
    if not images is None:
        images = np.array(images)
        image_z = (0, images.shape[axis] - 1)
    elif not points is None:
        image_z = (0, len(points)-1)

    #IntSlider(min=-10, max=30, step=1, value=10)
    interact(lambda x: imglocs(image_z=x, images=images, points=points, axis=axis, **kargs), x=image_z, **iarkgs)

def comp_images( fixed_npa, moving_npa, fixed_image_z, moving_image_z=None, cmap='viridis', axis=0):
    
    # Create a figure with two subplots and the specified size.
    if moving_image_z is None:
        moving_image_z = fixed_image_z
    plt.subplots(1, 2, figsize=(10, 8))
    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(take_data(fixed_npa, fixed_image_z, axis), cmap=cmap)
    plt.title("fixed image")
    plt.axis("off")

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(take_data(moving_npa, moving_image_z, axis), cmap=cmap)
    plt.title("moving image")
    plt.axis("off")

    plt.show()

def imshows(image_z, images, axis=0, cmap=None,  figsize=(8,8),**kargs):
    img=take_data(images, image_z, axis)
    plt.subplots(1, 1, figsize=figsize)
    # Draw the fixed image in the first subplot.
    plt.subplot(1, 1, 1)
    plt.imshow(img, cmap=cmap, **kargs)
    plt.axis("off")
    plt.show()

def imglocs(image_z, images, points, axis=0, cmap=None, size=1,
            origin = 'upper', 
            color='red', alpha=1, edgecolor=None, marker='.', swap_xy=True,
            equal_aspect=True,
            grid=False,
            iargs = {}, **kargs):
    if not images is None:
        img=take_data(np.array(images), image_z, axis)
        plt.imshow(img, cmap=cmap, origin=origin, **iargs)
    if not points is None:
        ipoint = points[image_z]
        if isinstance(ipoint, pd.DataFrame):
            ipoint = ipoint.values
        if swap_xy:
            pointxy =ipoint[:,[1,0]]
        else:
            pointxy =ipoint
        plt.scatter(pointxy[:,0], pointxy[:,1], s=size, c=color, 
                     edgecolor=edgecolor, alpha=alpha, marker=marker, **kargs)

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(grid)
    plt.axis("off")
    plt.show()

def comp_images_alpha(image_z, alpha, fixed, moving, cmap=plt.cm.Greys_r, axis=0):
    ifix=take_data(fixed, image_z, axis)
    imov=take_data(moving, image_z, axis)
    img = (1.0 - alpha) * ifix + alpha * imov
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()




# interact(
#     comp_images_alpha,
#     image_z=(0, transformed.shape[2] - 1),
#     alpha=(0.0, 1.0, 0.05),
#     axis=2,
#     fixed=fixed(static),
#     moving=fixed(transformed),
# );

# interact(
#     comp_images,
#     fixed_image_z=(0, static.shape[2] - 1),
#     moving_image_z=(0, transformed.shape[2] - 1),
#     axis=(0, 2),
#     fixed_npa=fixed(static),
#     moving_npa=fixed(transformed),

# );