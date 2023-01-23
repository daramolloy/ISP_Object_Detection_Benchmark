# File: wnf.py
# Description: Wavelet Noise Filtering
# Created: 22/02/2022
# Author: Hao Lin


import numpy as np
import cv2
from skimage.restoration import denoise_wavelet
from matplotlib import pyplot as plt

from .basic_module import BasicModule


class WNF(BasicModule):

    """Perform wavelet denoising on an image.

        Parameters
        ----------
        image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
            Input data to be denoised. `image` can be of any numeric type,
            but it is cast into an ndarray of floats for the computation
            of the denoised image.
        sigma : float or list, optional
            The noise standard deviation used when computing the wavelet detail
            coefficient threshold(s). When None (default), the noise standard
            deviation is estimated via the method in [2]_.
        wavelet : string, optional
            The type of wavelet to perform and can be any of the options
            ``pywt.wavelist`` outputs. The default is `'db1'`. For example,
            ``wavelet`` can be any of ``{'db2', 'haar', 'sym9'}`` and many more.
        mode : {'soft', 'hard'}, optional
            An optional argument to choose the type of denoising performed. It
            noted that choosing soft thresholding given additive noise finds the
            best approximation of the original image.
        wavelet_levels : int or None, optional
            The number of wavelet decomposition levels to use.  The default is
            three less than the maximum number of possible decomposition levels.
        multichannel : bool, optional
            Apply wavelet denoising separately for each channel (where channels
            correspond to the final axis of the array).
        convert2ycbcr : bool, optional
            If True and multichannel True, do the wavelet denoising in the YCbCr
            colorspace instead of the RGB color space. This typically results in
            better performance for RGB images.
        method : {'BayesShrink', 'VisuShrink'}, optional
            Thresholding method to be used. The currently supported methods are
            "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".
        rescale_sigma : bool, optional
            If False, no rescaling of the user-provided ``sigma`` will be
            performed. The default of ``True`` rescales sigma appropriately if the
            image is rescaled internally.

        Returns
        -------
        out : ndarray
            Denoised image."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rescale = self.params.rescale
        self.multichannel = self.params.multichannel
        self.ycbcr = self.params.ycbcr
        self.wavelet_levels = self.params.wavelet_levels
        self.sigma = self.params.sigma
        pass

    def execute(self, data):
        img = data['rgb_image'].astype(np.uint8)
        # img = cv2.cvtColor(y_image, cv2.COLOR_GRAY2BGR)

        output = denoise_wavelet(img, rescale_sigma=self.rescale, multichannel=self.multichannel, sigma=self.sigma,
                                 convert2ycbcr=self.ycbcr, wavelet_levels=self.wavelet_levels)

        data['rgb_image'] = (output*255).astype(np.uint16)