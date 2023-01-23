# File: he.py
# Description: Histogram Equalisation (actually not Auto)
# Created: 2021/02/12 20:50
# Author: Tim


import numpy as np
import cv2
from matplotlib import pyplot as plt

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer


class HE(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_bins = self.params.num_bins

    def execute(self, data):
        y_image = data['y_image'].astype(np.uint16)
        img = cv2.cvtColor(y_image, cv2.COLOR_GRAY2BGR)
        #img = cv2.resize(img, (640,480))

        # just using numpy
        hist, bins = np.histogram(y_image, bins=self.num_bins, range=(0, self.cfg.saturation_values.sdr))

        # plt.figure()
        # plt.title("Grayscale Histogram")
        # plt.xlabel("grayscale value")
        # plt.ylabel("pixel count")
        #
        # plt.plot(hist)  # <- or here
        # plt.show()


        cdf = hist.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize
        y_image = np.interp(y_image.flatten(), bins[:-1], cdf).reshape(self.cfg.hardware.raw_height, self.cfg.hardware.raw_width).astype(np.uint8)

        # using open cv
        img_new = cv2.equalizeHist(data['y_image'])
        hist_new, bins_new = np.histogram(img_new, bins=256, range=(0, self.cfg.saturation_values.sdr))

        # horizontal_stack = np.hstack((img, img_new))
        #
        # cv2.imshow('', img_new)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        data['y_image'] = img_new