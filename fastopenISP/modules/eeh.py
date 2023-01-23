# File: eeh.py
# Description: Edge Enhancement
# Created: 2023/01/15 20:50
# Author: Brian Deegan


import numpy as np
import cv2
import matplotlib.pyplot as plt

from .basic_module import BasicModule, register_dependent_modules
from .helpers import generic_filter, gen_gaussian_kernel


@register_dependent_modules('csc')
class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        kernel = gen_gaussian_kernel(kernel_size=self.params.kernel_size, sigma=self.params.sigma)
        self.gaussian = (1024 * kernel / kernel.max()).astype(np.int32)  # x1024


    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        delta = y_image - generic_filter(y_image, self.gaussian)
        sign_map = np.sign(delta)
        abs_delta = np.abs(delta)

        enhanced_delta = (
                (abs_delta > self.params.flat_threshold) * (delta * (self.params.edge_gain/256.0))
        )

        enhanced_delta = np.clip(enhanced_delta, -self.params.delta_threshold, self.params.delta_threshold)
        eeh_y_image = np.clip(y_image + enhanced_delta, 0, self.cfg.saturation_values.sdr)

        data['y_image'] = eeh_y_image.astype(np.uint8)
        data['edge_map'] = delta
        plt.imshow(data['y_image'])