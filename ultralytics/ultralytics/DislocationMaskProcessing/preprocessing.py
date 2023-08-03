import numpy as np
import scipy as scp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib import cm
import cv2
import PIL
from skimage import *
import skimage.io
import skimage.morphology as morphology
import matplotlib.pyplot as plt
from pathlib import Path

from cv2 import connectedComponentsWithStats

connected_components_cmap = ListedColormap([(0, 0, 0)] + [c for c in cm.get_cmap('tab20').colors], name='my colormap')
default_fig_size = (4.5, 4.5)
default_fig_wspace = 0.03

def connected_components(image: np.ndarray,
                         connectivity: int = 4,
                         width_limits: tuple = (5, 200),
                         height_limits: tuple = (5, 200),
                         area_limits: tuple = (200, 5000),
                         show_bboxes=False,
                         show_components=False,
                         verbose=False):
    """ Extract all groups of directly connected pixels.

    To get one particular pixel group:
    ```
    p = MaskPreprocessor(path)
    p.thresholding()
    labels, pixel_groups = p.connected_components()
    plt.imshow(pixel_groups[3])
    x0, y0, x1, y1 = bboxes[3]
    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
    ```
    :param image:           numpy array of ints with the image data
    :param connectivity:    4 or 8 (N, E, S, W or N, NE, E, ...)
    :param width_limits:    tuple with min/max value of horizontal pixel.
                            Pixel groups with width values outside this range are rejected.
    :param height_limits:   tuple with min/max value of vertical pixel.
                            Pixel groups with heights outside this range are rejected.
    :param show_bboxes:     figure showing the bounding boxes of the connected components
    :param show_components: figure showing the extracted pixel groups
    :param verbose:         allow some extra text output
    :returns: labels (an array with same shape as the image and pixel values according to the "labels",
                     i.e., the pixel group number)
              pixel_groups (a list of arrays where each array corresponds to one pixel group). Each pixel_group
                     image is typically smaller than the original image
              bboxes  (a list of coordinates (left, top, right, bottom) of the bounding box)
    """
    assert connectivity in [4, 8]
    # https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

    # apply connected component analysis to the thresholded image
    # connectivity = 4   # 4 or 8:
    # https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri
    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    # The cv2.connectedComponentsWithStats then returns a 4-tuple of:
    # - The total number of unique labels (i.e., number of total components) that were detected
    # - A mask named labels has the same spatial dimensions as our input thresh image. For each location in labels,
    #   we have an integer ID value that corresponds to the connected component where the pixel belongs. You’ll
    #   learn how to filter the labels matrix later in this section.
    # - stats: Statistics on each connected component, including the bounding box coordinates and area (in pixels).
    # - The centroids (i.e., center) (x, y)-coordinates of each connected component.

    if show_bboxes:
        ncols = 4
        nrows = numLabels // ncols + (numLabels % ncols > 0)
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex="col", sharey="row",
                                 figsize=(ncols * 3.5 + 0., nrows * 3.5))
        ax = axes.ravel()
        [a.set(xticklabels=[], xticks=[], yticklabels=[], yticks=[], facecolor='1', frame_on=False) for a in ax]

    # loop over the number of unique connected component labels
    mask = np.zeros(image.shape, dtype="uint8")
    pixel_groups = []
    bboxes = []
    for i in range(0, numLabels):
        # if this is the first component then we examine the *background* (typically we
        # would just ignore this component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)

        # print a status message update for the current connected component
        if verbose:
            print("[INFO] {}".format(text))

        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        if show_bboxes:
            cv_image = np.array(image / 255, dtype=np.float32)
            output = np.array(255 * cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 50), 8)
            cv2.circle(output, (int(cX), int(cY)), 20, (50, 150, 255), -1)
            ax[i].imshow(output)
            ax[i].text(50, 50, 'id =' + str(i), va='center', ha='left', c='yellow', fontsize='large')

        # ensure the width, height, and area are all neither too small
        # nor too big
        keep_width = width_limits[0] < w < width_limits[1]
        keep_height = height_limits[0] < h < height_limits[1]
        keep_area = area_limits[0] < area < area_limits[1]
        if all((keep_width, keep_height, keep_area)):
            # construct a mask for the current connected component and then take the bitwise OR with the mask
            if verbose:
                print("[INFO] keeping connected component '{}'".format(i))
            component_mask = (labels == i).astype("uint8") * i  # 255
            # mask = cv2.bitwise_or(mask, component_mask)
            mask += component_mask
            pixel_groups.append((labels == i).astype("uint8"))
            bboxes.append((x, y, x + w, y + h))

    if show_bboxes:
        for i in range(numLabels, ax.size):
            ax[i].remove()
        plt.tight_layout()

    if show_components:
        fig, ax = plt.subplots(ncols=3, figsize=(11.5, 5.5), gridspec_kw={'width_ratios': [1, 1, 0.05], 'wspace': 0.03})
        im0 = ax[0].imshow(image, cmap='gray')
        cmap = connected_components_cmap  # ListedColormap([(0, 0, 0)] + [c for c in cm.get_cmap('tab20').colors], name='my colormap')
        im1 = ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=cmap.N, interpolation='none')
        cb = plt.colorbar(im1, cax=ax[2], ticks=0.5 + np.arange(cmap.N), label='classes')
        cb.set_ticklabels(['backgr.'] + [f"#{l}" for l in range(cmap.N)])
        [a.set_frame_on(False) for a in ax[:2]]
        ax[0].set(xticklabels=[], xticks=[], yticklabels=[], yticks=[], title='input mask')
        ax[1].set(xticklabels=[], xticks=[], yticklabels=[], yticks=[], title='identified pixel groups')

    # return numLabels, labels, stats, centroids
    return labels, pixel_groups, bboxes



class MaskDataClass:
    """This class provides methods for handling the mask data and some conversion (CV2/np.array) functionality.

     Default value range is 0..255. For 0..1 there is the getter "cv_image".

    self._image holds the data and is an array of ints with values from 0.255
    """

    def __init__(self, image: [str, Path, np.ndarray], verbose=True):
        """

        :param image: can be a path to an image or the numpy array of the image itself
        """
        if isinstance(image, np.ndarray):
            mask = image.copy() if np.isclose(image.max(), 0) else  255 * image.copy()
        else:
            path = Path(str(image))
            assert path.exists(), f"The path '{path}' does not exist."

            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # or 0
            # mask = PIL.Image.open(path)
            # pil_image = mask.convert('RGB')
            # open_cv_image = np.array(pil_image)
            # _cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            # gray = cv2.cvtColor(_cv_image, cv2.COLOR_BGR2GRAY)
            # print(type(gray), gray.dtype)
            # print(gray)

        self.original_image = np.array(mask, dtype=np.float32)  # that's the equivalent to cv2.CV_32F
        self.__image = np.array(mask, dtype=np.float32)

        # some sanity checks ensuring that that bg=0, fg=255
        # (doesn't work for grayscale images, only for masks!)
        n, _ = np.histogram(self.__image, bins=[-0.5, 0.5, 254.5, 255.5])
        if not (n[0] == n[2] == 0):
            # we're using a binary mask...
            assert n[0] > 0, "no background pixel"
            assert n[2] > 0, "no foreground pixel (-->empty image or not 0..255)"
            if verbose and (n[0] < n[2]):
                print("[INFO]: Image has less background pixel (value=0, black) than foreground pixel (value=255, white) \n"
                      f"        --> Image might have to be inverted: freq. of value=0: {n[0]}, ... of value =255: {n[2]}")

    # Setter would not work as expected, e.g. self.image[:30, 3] = 0 does not work. To avoid confusion
    # or unexpected results we do not implement setter.
    @property
    def cv_image(self):
        # data are between 0 and 1
        return np.array(1 - self.image / 255, dtype=np.float32)

    @property
    def image(self):
        return np.array(self.__image, dtype=np.uint8)

    @image.setter
    def image(self, img):
        # Careful, a.image[3:4, 0] = 42 not working!
        self.__image = np.array(img, dtype=int)

    def set_image(self, image):
        self.__image = np.array(image, dtype=np.uint8)

    @staticmethod
    def from_cv_image_to_pil_array(cv_image):
        # convert the given cv image (0..1) to a "regular" numpy array with values 0..255
        return (1 - cv_image) * 255

    def set_image_data_from_cv_image(self, cv_image):
        self.__image = (1 - cv_image) * 255

    def show(self, title='', ax=None, show_colorbar=False, hist=True):
        """

        :param title:
        :param ax:
        :param show_colorbar:
        :param hist: show histogram
        :return:
        """
        if not ax:
            ncols = 2 if hist else 1
            figsize = (ncols * default_fig_size[0] + (ncols - 1) * default_fig_size[0] * default_fig_wspace,
                       default_fig_size[1])
            if show_colorbar and hist:
                fig, ax = plt.subplots(ncols=ncols, figsize=(figsize[0] + 3, figsize[1]),
                                       gridspec_kw={'wspace': 0.2, 'width_ratios': [1.4, 1]})
            else:
                fig, ax = plt.subplots(ncols=ncols, figsize=figsize, )#gridspec_kw={'wspace': default_fig_wspace})

        ax0, ax1 = ax if hist else (ax, None)

        im = ax0.imshow(self.__image, vmin=0, vmax=255, cmap='gray')
        ax0.set(title=title, xticks=[], yticks=[])

        if show_colorbar:
            divider = make_axes_locatable(ax0)  # Use axes divider to exactly match the color bar height with the 2D plot
            cax = divider.new_horizontal(size="6%", pad=0.2,
                                         pack_start=False)  # width, space True/False=left/right of ax1
            ax0.figure.add_axes(cax)
            plt.colorbar(im, cax=cax)


        if ax1:
            ax1.hist(self.__image.ravel(), bins=np.linspace(-0.5, 255.5, 32))
            ax1.set(title='histogram of gray values')

        if not show_colorbar:
            plt.tight_layout()


    def invert(self):
        # cv2.invert(self.__image/255, self.__image) * 255
        self.__image = 255 - self.__image


    def info(self):
        d = 0  # self.__image.max() - self.__image.min()
        bins = np.linspace(self.__image.min() - d / 20, self.__image.max() + d / 20, 10)
        n, _ = np.histogram(self.__image.ravel(), bins=bins)
        bc = bins[:-1] + 0.5 * np.diff(bins)
        print(f"self.__image:\n"
              f"- min/max = {self.__image.min()} / {self.__image.max()}\n"
              f"- frequency of values:\n" +
              "  bin center: " + (len(bc) * "{:8g} |").format(*bc) + "\n" +
              "  frequency:  " + (len(n) * "{:8g} |").format(*n) + "\n"
              )





class MaskPreprocessor:
    """The main data container and a collection of image processing functions for noise removal, gap closure etc

    Default value range is 0..255. For 0..1 there is the getter "cv_image".

    self._image holds the data and is an array of ints with values from 0.255
    """
    CMAP = 'gray'

    def __init__(self, image: [str, Path, np.ndarray], verbose=True):
        """

        :param image: can be a path to an image or the numpy array of the image itself
        """
        if isinstance(image, np.ndarray):
            mask = image.copy() if np.isclose(image.max(), 0) else  255 * image.copy()
        else:
            path = Path(str(image))
            assert path.exists()

            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # or 0
            # mask = PIL.Image.open(path)
            # pil_image = mask.convert('RGB')
            # open_cv_image = np.array(pil_image)
            # _cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            # gray = cv2.cvtColor(_cv_image, cv2.COLOR_BGR2GRAY)
            # print(type(gray), gray.dtype)
            # print(gray)

        self.original_image = np.array(mask, dtype=np.float32)  # that's the equivalent to cv2.CV_32F
        self.__image = np.array(mask, dtype=np.float32)

        # some sanity checks ensuring that that bg=0, fg=255
        n, _ = np.histogram(self.__image, bins=[-0.5, 0.5, 254.5, 255.5])
        assert n[0] > 0, "no background pixel"
        assert n[2] > 0, "no foreground pixel (-->empty image or not 0..255)"
        if verbose and (n[0] < n[2]):
            print("[INFO]: Image has less background pixel (value=0, black) than foreground pixel (value=255, white) \n"
                  f"        --> Image might have to be inverted: freq. of value=0: {n[0]}, ... of value =255: {n[2]}")

    @property
    def cv_image(self):
        # print("cv_image prop")
        # data are between 0 and 1
        return np.array(1 - self.image / 255, dtype=np.float32)

    # @cv_image.setter
    # def cv_image(self, cv_img):
    #     # Attention -- you can not do things like self.cv_image[:10, :50] = 42
    #     #print("cv_image setter")
    #     self.__image = np.array(1 - np.array(cv_img, dtype=float) * 255, dtype=int)

    @property
    def image(self):
        # print("image prop")
        return np.array(self.__image, dtype=np.uint8)

    # @image.setter
    # def image(self, img):
    #     # Careful, a.image[3:4, 0] = 42 not working!
    #     #print("image setter")
    #     self.__image = np.array(img, dtype=int)

    @staticmethod
    def from_cv_image_to_pil_array(cv_image):
        # convert the given cv image (0..1) to a "regular" numpy array with values 0..255
        return (1 - cv_image) * 255

    def set_image_data_from_cv_image(self, cv_image):
        self.__image = (1 - cv_image) * 255

    def show(self, title='', ax=None, show_colorbar=False):
        if not ax:
            fig, ax = plt.subplots()

        im = ax.imshow(self.__image, vmin=0, vmax=255, cmap=self.CMAP)
        ax.set(title=title, xticks=[], yticks=[])
        if show_colorbar:
            divider = make_axes_locatable(ax)  # Use axes divider to exactly match the color bar height with the 2D plot
            cax = divider.new_horizontal(size="6%", pad=0.2,
                                         pack_start=False)  # width, space True/False=left/right of ax1
            ax.figure.add_axes(cax)
            plt.colorbar(im, cax=cax)
        return im

    def invert(self):
        # cv2.invert(self.__image/255, self.__image) * 255
        self.__image = 255 - self.__image

    def thresholding(self, threshold=128, towards_zero=False):
        # change THRESH_BINARY_INV to THRESH_BINARY to avoid inverting
        # towards_zero=True --> 0..threshold will be mapped to 0, everything else to 255
        self.__image = np.array(self.__image.copy(), dtype=np.uint8)
        thresh_type = cv2.THRESH_BINARY_INV if towards_zero else cv2.THRESH_BINARY
        self.__image = cv2.threshold(self.__image, 0, 255, thresh_type | cv2.THRESH_OTSU)[1]

        # self.__image = (self.__image > threshold) * 255
        return

    def binary_closing(self, radius: int):
        d = morphology.disk(radius)
        img = morphology.binary_closing(self.cv_image, selem=d)
        self.set_image_data_from_cv_image(img)

    def info(self):
        d = 0  # self.__image.max() - self.__image.min()
        bins = np.linspace(self.__image.min() - d / 20, self.__image.max() + d / 20, 10)
        n, _ = np.histogram(self.__image.ravel(), bins=bins)
        bc = bins[:-1] + 0.5 * np.diff(bins)
        print(f"self.__image:\n"
              f"- min/max = {self.__image.min()} / {self.__image.max()}\n"
              f"- frequency of values:\n" +
              "  bin center: " + (len(bc) * "{:8g} |").format(*bc) + "\n" +
              "  frequency:  " + (len(n) * "{:8g} |").format(*n) + "\n"
              )

    def connected_components(self, connectivity: int = 4,
                             width_limits: tuple = (5, 200),
                             height_limits: tuple = (5, 200),
                             area_limits: tuple = (200, 5000),
                             show_bboxes=False,
                             show_components=False,
                             verbose=False):
        """ Extract all groups of directly connected pixels.

        To get one particular pixel group:
        ```
        p = MaskPreprocessor(path)
        p.thresholding()
        labels, pixel_groups = p.connected_components()
        plt.imshow(pixel_groups[3])
        x0, y0, x1, y1 = bboxes[3]
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
        ```

        :param connectivity:    4 or 8 (N, E, S, W or N, NE, E, ...)
        :param width_limits:    tuple with min/max value of horizontal pixel.
                                Pixel groups with width values outside this range are rejected.
        :param height_limits:   tuple with min/max value of vertical pixel.
                                Pixel groups with heights outside this range are rejected.
        :param show_bboxes:     figure showing the bounding boxes of the connected components
        :param show_components: figure showing the extracted pixel groups
        :param verbose:         allow some extra text output
        :returns: labels (an array with same shape as the image and pixel values according to the "labels",
                         i.e., the pixel group number)
                  pixel_groups (a list of arrays where each array corresponds to one pixel group). Each pixel_group
                         image is typically smaller than the original image
                  bboxes  (a list of coordinates (left, top, right, bottom) of the bounding box)
        """
        assert connectivity in [4, 8]
        # https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

        # apply connected component analysis to the thresholded image
        # connectivity = 4   # 4 or 8:
        # https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri
        output = cv2.connectedComponentsWithStats(self.image, connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # The cv2.connectedComponentsWithStats
        # then returns a 4-tuple of:
        # - The total number of unique labels (i.e., number of total components) that were detected
        # - A mask named labels
        #     has the same spatial dimensions as our input thresh
        #     image. For each location in labels, we have an integer ID value that corresponds to the connected
        #     component where the pixel belongs. You’ll learn how to filter the labels matrix later in this section.
        # - stats: Statistics on each connected component, including the bounding box coordinates and area (in pixels).
        # - The centroids (i.e., center) (x, y)-coordinates of each connected component.

        if show_bboxes:
            ncols = 5
            nrows = numLabels // ncols + (numLabels % ncols > 0)
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex="col", sharey="row",
                                     figsize=(ncols * 3.5 + 0., nrows * 3.5))
            ax = axes.ravel()

        # loop over the number of unique connected component labels
        mask = np.zeros(self.__image.shape, dtype="uint8")
        pixel_groups = []
        bboxes = []
        centers = []
        for i in range(0, numLabels):
            # if this is the first component then we examine the *background* (typically we
            # would just ignore this component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format(i + 1, numLabels)

            # print a status message update for the current connected component
            if verbose:
                print("[INFO] {}".format(text))

            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            if show_bboxes:
                output = np.array(255 * cv2.cvtColor(self.cv_image.copy(), cv2.COLOR_GRAY2RGB), dtype=np.uint8)
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 50), 5)
                cv2.circle(output, (int(cX), int(cY)), 10, (0, 0, 255), -1)
                ax[i].imshow(output)  # , vmin=0, vmax=255)
                ax[i].text(50, 50, 'id =' + str(i)+"  area = "+ str(area), va='center', ha='left')

            # ensure the width, height, and area are all neither too small
            # nor too big
            keep_width = width_limits[0] < w < width_limits[1]
            keep_height = height_limits[0] < h < height_limits[1]
            keep_area = area_limits[0] < area < area_limits[1]
            if keep_area:
                # construct a mask for the current connected component and
                # then take the bitwise OR with the mask
                if verbose:
                    print("[INFO] keeping connected component '{}'".format(i))
                component_mask = (labels == i).astype("uint8") * i  # 255
                # mask = cv2.bitwise_or(mask, component_mask)
                mask += component_mask
                pixel_groups.append((labels == i).astype("uint8"))
                bboxes.append((x, y, x + w, y + h))
                centers.append((cX, cY))
        if show_bboxes:
            for i in range(numLabels, ax.size):
                ax[i].remove()
            plt.tight_layout()

        if show_components:
            fig, ax = plt.subplots(ncols=3, figsize=(11.5, 5.5), gridspec_kw={'width_ratios': [1, 1, 0.05], 'wspace': 0.03})
            im0 = ax[0].imshow(self.__image, cmap=self.CMAP)
            cmap = ListedColormap([(0, 0, 0)] + [c for c in cm.get_cmap('tab20').colors], name='my colormap')
            im1 = ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=cmap.N, interpolation='none')
            cb = plt.colorbar(im1, cax=ax[2], ticks=0.5 + np.arange(cmap.N), label='classes')
            cb.set_ticklabels(['backgr.'] + [f"#{l}" for l in range(1, cmap.N)])
            [a.set_frame_on(False) for a in ax[:2]]
            ax[0].set(xticklabels=[], xticks=[], yticklabels=[], yticks=[], facecolor='1')
            ax[1].set(xticklabels=[], xticks=[], yticklabels=[], yticks=[], facecolor='1')

        # return numLabels, labels, stats, centroids
        return labels, pixel_groups, bboxes , centers
