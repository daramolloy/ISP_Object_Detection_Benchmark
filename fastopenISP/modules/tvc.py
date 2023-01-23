# File: tvc.py
# Description: Total-Variation Chambolle
# Created: 22/02/2022
# Author: Hao Lin

import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pyplot as plt

from .basic_module import BasicModule


class TVC(BasicModule):

    """Perform total-variation denoising on n-dimensional images.

       Parameters
       ----------
       image : ndarray of ints, uints or floats
           Input data to be denoised. `image` can be of any numeric type,
           but it is cast into an ndarray of floats for the computation
           of the denoised image.
       weight : float, optional
           Denoising weight. The greater `weight`, the more denoising (at
           the expense of fidelity to `input`).
       eps : float, optional
           Relative difference of the value of the cost function that
           determines the stop criterion. The algorithm stops when:

               (E_(n-1) - E_n) < eps * E_0

       n_iter_max : int, optional
           Maximal number of iterations used for the optimization.
       multichannel : bool, optional
           Apply total-variation denoising separately for each channel. This
           option should be true for color images, otherwise the denoising is
           also applied in the channels dimension.

       Returns
       -------
       out : ndarray
           Denoised image."""


    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight = self.params.weight
        self.multichannel = self.params.multichannel
        self.eps = self.params.eps
        self.iterations = self.params.iterations

    def execute(self, data):
        img = data['rgb_image'].astype(np.uint8)
        # img = cv2.cvtColor(y_image, cv2.COLOR_GRAY2BGR)
        
        output = denoise_tv_chambolle(img, weight=self.weight, multichannel=self.multichannel, eps=self.eps, n_iter_max=self.iterations)
        
        data['rgb_image'] = (output*255).astype(np.uint16)
