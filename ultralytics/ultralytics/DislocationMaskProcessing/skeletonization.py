
from skimage.morphology import medial_axis, skeletonize, thin
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def bounding_box(image, threshold=0):
    mask = image > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ylim = np.where(rows)[0][[0, -1]]
    xlim = np.where(cols)[0][[0, -1]]
    return xlim, ylim



def separate_overlapping_segments(pixel_group_image: np.ndarray,
                                  threshold_pca_points=0.7,
                                  smoothing_window_size=35):
    """Use PCA to split lines that are nearby and therefore connected

    Adapted and extended from https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    Quote: "If you have two arrays of scattered x and y values that are not too curvy, then you can transform the
    points into PCA space, sort them in PCA space, and then transform them back. (I've also added in some bonus
    smoothing functionality)."

    This functions assumes that background pixel have value =0 and feature pixel are >0.
    You might have to fine tune the smoothing window.

    Exampple:

    ```
    path = Path("data/masks/mask002.png")
    p = MaskPreprocessor(path)
    p.thresholding()
    _, pixel_groups, _ = p.connected_components(width_limits=(20, 500), height_limits=(20, 500), area_limits=(200, 50000))

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 20))
    plt.tight_layout()

    for pixel_group, ax in zip(pixel_groups, axes.ravel()):
        original_points, pca_points = separate_overlapping_segments(pixel_group)
        ax.scatter(*original_points.T, s=1)
        ax.scatter(*pca_points.T, s=6, c='C3')
        ax.set(aspect="equal")
    ```


    :param pixel_group_image: Contains only 1 connected component, e.g., obtained from the list returned by
                              connected_components. It usually has the same shape as the "full" image.
    :param threshold_pca_points: how many PCA points must overlap with the original mask so that we assume that
                                 the line created from PCA is fully overlapping with the mask (0..1)
    :param smoothing_window_size: size of the window for smoothing of the line data in PCA space
    :return:
    """

    # This is the original code from the stackoverflow posting:
    def XYclean(x, y):
        xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

        # make PCA object
        pca = PCA(2)

        # fit on data
        pca.fit(xy)

        # transform into pca space
        xypca = pca.transform(xy)
        newx = xypca[:, 0]
        newy = xypca[:, 1]

        # sort
        indexSort = np.argsort(x)
        newx = newx[indexSort]
        newy = newy[indexSort]

        # add some more points (optional)
        f = interpolate.interp1d(newx, newy, kind='linear')
        newX = np.linspace(np.min(newx), np.max(newx), 100)
        newY = f(newX)
        # newX, newY = newx, newy

        # smooth with a filter (optional)
        window = smoothing_window_size  # 43
        newY = savgol_filter(newY, window, 2)

        # return back to old coordinates
        xyclean = pca.inverse_transform(np.concatenate((newX.reshape(-1, 1), newY.reshape(-1, 1)), axis=1))
        xc = xyclean[:, 0]
        yc = xyclean[:, 1]

        return xc, yc

    mask = np.array(pixel_group_image > 0, dtype=bool)
    shape = pixel_group_image.shape
    xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
    x = xx[mask].ravel()
    y = yy[mask].ravel()


    # This gives now points on some kind of "average line" that is in the middle of to nearby, roughly parallel lines
    xc, yc = XYclean(x, y)

    # How much "overlap" is between these points and the original points? If no much then the "average line" is between
    # two lines.
    original_points = np.vstack((x, y)).T
    pca_points = np.vstack((xc, yc)).T

    # We're using sets in order to compute the intersection
    original_points_set = set(tuple(np.array(aa, dtype=int)) for aa in original_points)
    pca_points_set = set(tuple(np.array(aa, dtype=int)) for aa in pca_points)
    len(original_points_set.intersection(pca_points_set)), len(original_points_set), len(pca_points_set)

    # for a "difference image" that only contains those pixel that were "cut away"
    diff_img = np.zeros_like(pixel_group_image)
    pixel_group_image = pixel_group_image.copy()

    if len(original_points_set.intersection(pca_points_set)) >= threshold_pca_points * len(pca_points_set):
        # it is one fully connected line segment
        has_single_component = True
    else:
        # it might be two nearby segments -- split them by the pca line
        for i, j in pca_points_set:
            # delete pixel at the following positions
            indices_for_erasing = [(j, i),
                                   (min(shape[1], j+1), i),
                                   (j, min(i+1, shape[0]))
                                  ]
            for idx in indices_for_erasing:
                if pixel_group_image[idx] > 0:
                    diff_img[idx] = 1
                    pixel_group_image[idx] = 0

        has_single_component = False

    return has_single_component, pixel_group_image, diff_img, original_points, pca_points


def show_skeletonized_image(image: np.ndarray):
    """Shows three different skeletonization methods of scikit learn"""
    # https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html

    mask = image > 0
    shape = image.shape
    xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
    x = xx[mask].ravel()
    y = yy[mask].ravel()

    xlim, ylim = bounding_box(image)
    w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]

    fig, ax = plt.subplots(nrows=3 if h < w else 1, ncols=3 if h >= w else 1,
                           figsize=(17, 17), sharey=True, sharex=True)

    # 1st figure -----------------------------------------------
    skel, distance = medial_axis(image, mask, return_distance=True)
    dist_on_skel = distance * skel
    ax[0].imshow(np.ma.masked_array(dist_on_skel > 0, mask=np.isclose(dist_on_skel, 0)),
                 cmap='gray', origin='lower')
    ax[0].scatter(x, y, s=1, marker='.')
    ax[0].set(title="medial axis", xlim=xlim, ylim=ylim)

    # 2nd figure -----------------------------------------------
    skeleton_lee = skeletonize(image, method='lee')
    ax[1].imshow(np.ma.masked_array(skeleton_lee > 0, mask=np.isclose(skeleton_lee, 0)),
                 cmap='gray', origin='lower')
    ax[1].scatter(x, y, s=1, marker='.')
    ax[1].set(title="Lee", xlim=xlim, ylim=ylim)

    # 3rd figure -----------------------------------------------
    thinned = thin(image)
    ax[2].imshow(np.ma.masked_array(thinned > 0, mask=np.isclose(thinned, 0)),
                 cmap='gray', origin='lower')
    ax[2].scatter(x, y, s=1, marker='.')
    ax[2].set(title="using 'thin'", xlim=xlim, ylim=ylim)

    plt.tight_layout()


def lee_skeletonization(image: np.ndarray,start=0):
    """

    :param image: biarized image
           start: Starting point to complete the ordering of points. There are only two valid values 0, -1. 
    :return: the skeletonized image, and the vectors of x and y corrdinates
    """
    sk = skeletonize(image, method='lee')

    mask = sk > 0
    shape = image.shape
    xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
    x = xx[mask].ravel()
    y = yy[mask].ravel()
    
    p1 = np.array([[x[start],y[start]]]) # Get first point
    p = np.transpose(np.array([x,y])) # get all points 
    p = np.delete(p,start,axis=0) # remove the first point
    xnew , ynew = [] , [] # updated ordered list of points based on minimum distance from points 
    xnew.append(p1[0,0]) 
    ynew.append(p1[0,1])
    
    wrong_start = False # check if everything if fine. Might need to change the starting point if next neighbour is too far 
    length = 0.0
    for i in range(1,len(x)-1):
        dis = cdist(p1,p) # distance between point p1 and all the other points on the splines. p does not include the points which are already calculated
        xnew.append(p[dis.argmin(),0]) # adding the next closest point to the new list 
        ynew.append(p[dis.argmin(),1])
        if (dis.min() > 5): wrong_start = True
        p = np.delete(p,dis.argmin(),axis=0) # deleting the point already calculated 
        p1 = np.array([[xnew[-1],ynew[-1]]]) # Now find the nearest neighbour to this point and continue
        length+=dis.min()
                
    return sk, np.array(xnew), np.array(ynew) , wrong_start, length



