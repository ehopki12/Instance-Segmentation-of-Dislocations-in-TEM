import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import cv2
import PIL
import skimage.morphology as morphology
from pathlib import Path
from ultralytics.DislocationMaskProcessing.preprocessing import (MaskDataClass,
                                                     connected_components,
                                                     connected_components_cmap)
from ultralytics.DislocationMaskProcessing.skeletonization import (lee_skeletonization,
                                                       separate_overlapping_segments,
                                                       bounding_box)
from ultralytics.DislocationMaskProcessing.curvefitting import order_data_points, fit_spline


class MaskAnalysis(MaskDataClass):
    def __init__(self, img: [str, Path, np.ndarray], verbose=True):
        super().__init__(image=img, verbose=verbose)

        # data changed by various methods
        self.labels = None           # an image/mask with integers (colors) for different classes.
        #                            # self.labels has same shape as self.__image
        self.pixel_groups = []       # a list of image subsection each of which contains (ideally) one dislocation
        self.bboxes = []             # list of coordinates of each pixel_groups' sub-image (left, upper, right, lower)
        self.spline_coords = []      # list of all fitted splie coordinates


    def threshold(self, type='manual', threshold=128, block_size=11, offset=2):
        """Binarization of the image.

        :param type: - manual:         Global thresholding with given `threshold` value
                     - Otsu:           auto determination from analysing the bimodal distribution.
                                       Ignores the given threshold value.
                     - adaptive mean:  the threshold value is the mean of the neighbourhood area
                                       minus the constant "offset"
                     - adaptive gaussian: The threshold value is a gaussian-weighted sum of the neighbourhood values
                                       minus the constant "offset".
        :param threshold:              threshold value for manual thresholding
        :param block_size:             only for adaptive thresholds
                                       Size of pixel neighborhood that is used to calculate a threshold value for the
                                       pixel: 3, 5, 7, and so on.
        :param offset:                 only for adaptive thresholds
                                       Constant subtracted from the mean or weighted mean (see the details below).
                                       Normally, it is positive but may be zero or negative as well.
        """
        assert type in ['manual', 'Otsu', 'adaptive mean', 'adaptive gaussian']
        block_size = int(block_size)
        offset = int(offset)

        # Make sure that the image has correct type. Probably not needed any more...
        self.image = np.array(self.image, dtype=np.uint8)

        # We also assume that the background is 0 (black) and the foreground 255 (white). Since this is
        # opposite to what is CV assumes as default, we use cv2.THRESH_BINARY_INV to indicate that we are
        # inverting the data at the end of this function.

        if type == 'manual':
            thresh_type = cv2.THRESH_BINARY_INV  # if towards_zero else cv2.THRESH_BINARY
            self.image = cv2.threshold(self.image, threshold, 255, thresh_type)[1]

        elif type == 'Otsu':
            thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            self.image = cv2.threshold(self.image, 0, 255, thresh_type)[1]

        elif type == 'adaptive mean':
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, block_size, offset)
        elif type == 'adaptive gaussian':
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, block_size, offset)

        else:
            assert False, "You never should end up here..."

        self.invert()


    def extract_and_store_connected_components(self, min_size=20, max_size=500,
                                               show_bboxes=False, show_components=False, verbose=False):
        self.labels, self.pixel_groups, self.bboxes = connected_components(self.image,
                                                                           show_bboxes=show_bboxes,
                                                                           show_components=show_components,
                                                                           verbose=verbose,
                                                                           width_limits=(min_size, max_size),
                                                                           height_limits=(min_size, max_size),
                                                                           area_limits=(200, 50000))


    def get_coordinates_from_pixel_groups(self):
        """Return a list of pixel coordinates for the individual regions in self.labels"""
        assert self.pixel_groups != [], "the list self.pixel_groups need to be defined first"
        assert self.labels is not None, "the image array self.labels need to be defined first"

        coords = []
        for i in range(len(self.pixel_groups)):
            mask = self.labels == i
            shape = mask.shape
            xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
            x = xx[mask].ravel()
            y = yy[mask].ravel()
            coords.append(np.c_[x, y])
        return coords


    def split_overlapping_lines(self, min_size=20, max_size=500, threshold_pca_points=0.7, smoothing_window_size=35):
        """Use PCA to split nearby, partially connected lines

        This works only if lines are approximately parallel. No real intersections can be handled here! In case that
        there are multiple interesections that can not be resolved, the functions still continues.

        :param min_size: parameter for extract_and_store_connected_components
        :param max_size: parameter for extract_and_store_connected_components
        :param threshold_pca_points: how many PCA points must overlap with the original mask so that we assume that
                             the line created from PCA is fully overlapping with the mask (0..1)
        :param smoothing_window_size: size of the window for smoothing of the line data in PCA space
        """
        disloc_images = []
        disloc_bboxes = []
        for i, (pixel_group, bbox) in enumerate(zip(self.pixel_groups, self.bboxes)):

            # Use PCA to split lines that are nearby and therefore connected
            has_single_component, pixel_group_image, diff_img, original_points, pca_points = \
                separate_overlapping_segments(pixel_group_image=pixel_group,
                                              threshold_pca_points=threshold_pca_points,
                                              smoothing_window_size=smoothing_window_size)

            if has_single_component:
                # Nope, just a single line -- add it to th list of lines
                disloc_images.append(pixel_group)
                disloc_bboxes.append(bbox)
            else:
                # There is more than one dislocation line in pixel_group_image
                _, split_pixel_groups, _ = connected_components(pixel_group_image,
                                                                width_limits=(min_size, max_size),
                                                                height_limits=(min_size, max_size),
                                                                area_limits=(min_size ** 2, min_size * max_size))

                for split_pixel_group in split_pixel_groups:

                    # just a sanity check in case that the image has multiple intersections...
                    has_single_component2, img, _, _, _ =  \
                        separate_overlapping_segments(pixel_group_image=split_pixel_group,
                                                      threshold_pca_points=threshold_pca_points,
                                                      smoothing_window_size=smoothing_window_size)
                    if not has_single_component2:
                        print("there are more intersection in this image")
                    # assert has_single_component2, "there are more intersection in this image"

                    # if everything is good, we can now add the bunch of pixels to the list
                    disloc_images.append(split_pixel_group)
                    disloc_bboxes.append(bbox)

        self.pixel_groups = list(disloc_images)
        self.bboxes = list(disloc_bboxes)


    def subplot_all_individual_pixel_groups(self, title="", frame_on=False, padding=0):
        """Create one subplot for each separate pixel group/line

        :param frame_on: show frame/ticks or not
        :param padding: some extra pixel so that, e.g., thinned lines can be better seen
        """
        ncols = 4
        fig, axes = plt.subplots(nrows=len(self.pixel_groups) // ncols + 1, ncols=ncols,
                                 figsize=(15, 4 * (len(self.pixel_groups) // ncols + 1)))
        ax = axes.ravel()

        if frame_on:
            [a.set(aspect="equal") for a in ax]
        else:
            [a.set(xticks=[], yticks=[], frame_on=False, aspect="equal") for a in ax]

        if title != "":
            fig.suptitle(title, fontsize='xx-large', y=0.99)

        for i in range(len(self.pixel_groups), len(ax)):
           ax[i].remove()

        colors = connected_components_cmap.colors[1:]
        for i, (_ax, img) in enumerate(zip(ax, self.pixel_groups)):
            xlim, ylim = bounding_box(img)
            xlim = (xlim[0] - padding, xlim[1] + padding)
            ylim = (ylim[0] - padding, ylim[1] + padding)
            _ax.imshow(img, interpolation='None', cmap=ListedColormap([(0.3, 0.3, 0.3), list(colors[i % len(colors)])]))
            _ax.set(xlim=xlim, ylim=ylim[::-1])
            dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
            _ax.text(xlim[0] + 0.05*dx, 0.95*dy + ylim[0], f'#{i}',
                     va='center', ha='left', c='white', fontsize='x-large')

            # if show_splines and self.spline_coords:
            #     print("plotting splines")
            #     coords = self.spline_coords[i]
            #     #_ax.plot(*coords.T, c='1', lw=3)
            #     _ax.plot(*coords.T, c='0', lw=3, ls='--')

        plt.tight_layout()
        return fig, axes


    def do_line_thinning(self, update_bboxes=True):
        """Lee skeletonization is applied to each pixel_group as identified by `extract_and_store_connected_components`
        """
        assert self.pixel_groups is not None, \
            "`extract_and_store_connected_components` must be called prior to this function"

        self.labels = np.zeros_like(self.pixel_groups[0])
        new_pixel_groups = []
        bboxes = []
        for i, pixel_group in enumerate(self.pixel_groups, start=1):
            img, x, y = lee_skeletonization(pixel_group)
            new_pixel_groups.append(img)
            self.labels += i * img
            bboxes.append([x.min(), y.min(), x.max(), y.max()])
        self.labels = np.array(self.labels, dtype=np.uint8)
        self.pixel_groups = list(new_pixel_groups)
        if update_bboxes:
            self.bboxes = bboxes


    def fit_splines_to_pixel_groups(self, degree=3, smoothing=5., n_interp_points=50, n_skip=1):
        """For each sub-image of self.pixel_groups, fit a spline to the points

        :param degree: spline dregree
        :param smoothing: spline smoothing factor
        :param n_skip: only use every n_skip point
        """
        self.spline_coords = []
        for pixel_group in self.pixel_groups:
            mask = pixel_group > 0
            xx, yy = np.meshgrid(range(pixel_group.shape[1]), range(pixel_group.shape[0]))
            x = xx[mask].ravel()
            y = yy[mask].ravel()

            x, y = order_data_points(x, y)

            # skip some points but make sure that both end points are contained
            if not np.isclose(x[-1], x[::n_skip][-1]):
                x = np.concatenate((x[::n_skip], [x[-1]]))
                y = np.concatenate((y[::n_skip], [y[-1]]))

            fitted_points = fit_spline(x, y, degree=degree, smoothing=smoothing, n_interp_points=n_interp_points)
            self.spline_coords.append(fitted_points)


