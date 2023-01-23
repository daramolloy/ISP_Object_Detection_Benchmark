# File: cfa.py
# Description: Color Filter Array Interpolation
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np
# import PySpin
import cv2 as cv

from .basic_module import BasicModule
from .helpers import pad, split_bayer, reconstruct_bayer, shift_array


class CFA(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.channel_indices_and_weights = (
            [[1, 3, 4, 5, 7], [-0.125, -0.125, 0.5, -0.125, -0.125]],
            [[1, 3, 4, 5, 7], [-0.1875, -0.1875, 0.75, -0.1875, -0.1875]],
            [[1, 3, 4, 5, 7], [0.0625, -0.125, 0.625, -0.125, 0.0625]],
            [[1, 3, 4, 5, 7], [-0.125, 0.0625, 0.625, 0.0625, -0.125]],
            [[3, 4], [0.25, 0.25]],
            [[1, 4], [0.25, 0.25]],
            [[4, 7], [0.25, 0.25]],
            [[4, 5], [0.25, 0.25]],
            [[4, 5], [0.5, 0.5]],
            [[1, 3], [0.5, 0.5]],
            [[1, 4], [0.5, 0.5]],
            [[4, 7], [0.5, 0.5]],
            [[3, 4], [0.5, 0.5]],
            [[0, 1, 3, 4], [0.25, 0.25, 0.25, 0.25]],
            [[4, 5, 7, 8], [0.25, 0.25, 0.25, 0.25]],
            [[1, 2, 4, 5], [-0.125, -0.125, -0.125, -0.125]],
            [[3, 4, 6, 7], [-0.125, -0.125, -0.125, -0.125]]
        )

        for i_and_w in self.channel_indices_and_weights:
            i_and_w[1] = [int(1024 * w) for w in i_and_w[1]]  # x1024

    def rotate_to_rggb(self, array):
        k = {'rggb': 0,
             'bggr': 2,
             'grbg': 1,
             'gbrg': 3}[self.cfg.hardware.bayer_pattern]
        return np.rot90(array, k=k)

    def rotate_from_rggb(self, array):
        k = {'rggb': 0,
             'bggr': 2,
             'grbg': 3,
             'gbrg': 1}[self.cfg.hardware.bayer_pattern]
        return np.rot90(array, k=k)

    @staticmethod
    def index_weighted_sum(arrays, indices, weights):
        result = 0
        for i, w in zip(indices, weights):
            result += w * arrays[i]

        return result
        
    def execute(self, data):
        if self.params.mode.lower() == 'bilinear':
            self.execute_bilinear(data)
        elif self.params.mode.lower() == 'malvar':
            self.execute_malvar(data)
        elif self.params.mode.lower() == 'nn':
            self.execute_nearest_neighbour(data)
        elif self.params.mode.lower() == 'edge':
            self.execute_edge_aware(data)
        # elif self.params.mode.lower() == 'hqlinear':
        #     self.execute_hq_linear(data)
        # elif self.params.mode.lower() == 'nnf':
        #     self.execute_nearest_neighbour_fast(data)
        # elif self.params.mode.lower() == 'nna':
        #     self.execute_nearest_neighbour_average(data)
        # elif self.params.mode.lower() == 'edgesense':
        #     self.execute_edge_sense(data)
        # elif self.params.mode.lower() == 'ipp':
        #     self.execute_IPP(data)
        # elif self.params.mode.lower() == 'dirfilter':
        #     self.execute_directional_filter(data)
        # elif self.params.mode.lower() == 'rigorous':
        #     self.execute_rigorous(data)
        # elif self.params.mode.lower() == 'weighteddirfilter':
        #     self.execute_weighted_directional_filter(data)
        else:
            raise NotImplementedError
        
    # def execute_weighted_directional_filter(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.WEIGHTED_DIRECTIONAL_FILTER)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_rigorous(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.RIGOROUS)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_directional_filter(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.DIRECTIONAL_FILTER)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_IPP(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.IPP)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_edge_sense(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.EDGE_SENSING)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_nearest_neighbour_average(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.NEAREST_NEIGHBOR_AVG)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_nearest_neighbour_fast(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.NEAREST_NEIGHBOR)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)
    #
    # def execute_hq_linear(self, data):
    #     width = data['bayer'].shape[0]
    #     height = data['bayer'].shape[1]
    #     image_bayer = PySpin.Image.Create(height,width,0,0,PySpin.PixelFormat_BayerRG8,data['bayer'].astype(np.uint8))
    #     rgb_image = image_bayer.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    #     data['rgb_image'] = rgb_image.GetNDArray().astype(np.uint16)

    def execute_bilinear(self, data):
        bayer = data['bayer'].astype(np.int32)
        bayer = self.rotate_to_rggb(bayer)  # RGGB pattern

        r, gr, gb, b = split_bayer(bayer, bayer_pattern='rggb')
        height, width = r.shape

        # R-channel
        padded_r = pad(r, pads=(0, 1, 0, 1))
        r_right = padded_r[:height, 1:1 + width]
        r_bottom = padded_r[1:1 + height, :width]
        r_br = padded_r[1:1 + height, 1:1 + width]

        r_on_gr = np.right_shift(r + r_right, 1)
        r_on_gb = np.right_shift(r + r_bottom, 1)
        r_on_b = np.right_shift(r + r_right + r_bottom + r_br, 2)

        # G-channel
        padded_gr = pad(gr, pads=(0, 1, 1, 0))
        padded_gb = pad(gb, pads=(1, 0, 0, 1))
        gr_left = padded_gr[:height, :width]
        gr_bottom = padded_gr[1:1 + height, 1:1 + width]
        gb_top = padded_gb[:height, :width]
        gb_right = padded_gb[1:1 + height, 1:1 + width]

        g_on_r = np.right_shift(gr_left + gr + gb_top + gb, 2)
        g_on_b = np.right_shift(gr + gr_bottom + gb+ gb_right, 2)

        # B-channel
        padded_b = pad(b, pads=(1, 0, 1, 0))
        b_top = padded_b[:height, 1:1 + width]
        b_left = padded_b[1:1 + height, :width]
        b_tl = padded_b[:height, :width]

        b_on_r = np.right_shift(b_tl + b_top + b_left + b, 2)
        b_on_gr = np.right_shift(b_top + b, 1)
        b_on_gb = np.right_shift(b_left + b, 1)

        # ---------------- Reconstruction ----------------
        rgb_image = np.dstack([
            reconstruct_bayer((r, r_on_gr, r_on_gb, r_on_b), bayer_pattern='rggb'),
            reconstruct_bayer((g_on_r, gr, gb, g_on_b), bayer_pattern='rggb'),
            reconstruct_bayer((b_on_r, b_on_gr, b_on_gb, b), bayer_pattern='rggb')
        ])
        rgb_image = self.rotate_from_rggb(rgb_image)
        rgb_image = np.clip(rgb_image, 0, self.cfg.saturation_values.hdr)

        data['rgb_image'] = rgb_image.astype(np.uint16)

    def execute_malvar(self, data):
        bayer = data['bayer'].astype(np.int32)
        bayer = self.rotate_to_rggb(bayer)  # RGGB pattern

        r, gr, gb, b = split_bayer(bayer, bayer_pattern='rggb')

        padded_bayer = pad(bayer, pads=2)
        padded_r, padded_gr, padded_gb, padded_b = split_bayer(padded_bayer, bayer_pattern='rggb')

        shifted_r = tuple(shift_array(padded_r, window_size=3))  # generator --> tuple
        shifted_gr = tuple(shift_array(padded_gr, window_size=3))
        shifted_gb = tuple(shift_array(padded_gb, window_size=3))
        shifted_b = tuple(shift_array(padded_b, window_size=3))

        # ---------------- R-plane ----------------
        g_on_r = np.right_shift(
            self.index_weighted_sum(shifted_r, *self.channel_indices_and_weights[0]) +
            self.index_weighted_sum(shifted_gr, *self.channel_indices_and_weights[4]) +
            self.index_weighted_sum(shifted_gb, *self.channel_indices_and_weights[5]), 10
        )
        b_on_r = np.right_shift(
            self.index_weighted_sum(shifted_r, *self.channel_indices_and_weights[1]) +
            self.index_weighted_sum(shifted_b, *self.channel_indices_and_weights[13]), 10
        )

        # ---------------- Gr-plane ----------------
        r_on_gr = np.right_shift(
            self.index_weighted_sum(shifted_r, *self.channel_indices_and_weights[8]) +
            self.index_weighted_sum(shifted_gr, *self.channel_indices_and_weights[2]) +
            self.index_weighted_sum(shifted_gb, *self.channel_indices_and_weights[15]), 10
        )
        b_on_gr = np.right_shift(
            self.index_weighted_sum(shifted_b, *self.channel_indices_and_weights[10]) +
            self.index_weighted_sum(shifted_gr, *self.channel_indices_and_weights[3]) +
            self.index_weighted_sum(shifted_gb, *self.channel_indices_and_weights[15]), 10
        )

        # ---------------- Gb-plane ----------------
        r_on_gb = np.right_shift(
            self.index_weighted_sum(shifted_r, *self.channel_indices_and_weights[11]) +
            self.index_weighted_sum(shifted_gr, *self.channel_indices_and_weights[16]) +
            self.index_weighted_sum(shifted_gb, *self.channel_indices_and_weights[3]), 10
        )
        b_on_gb = np.right_shift(
            self.index_weighted_sum(shifted_b, *self.channel_indices_and_weights[12]) +
            self.index_weighted_sum(shifted_gr, *self.channel_indices_and_weights[16]) +
            self.index_weighted_sum(shifted_gb, *self.channel_indices_and_weights[2]), 10
        )

        # ---------------- B-plane ----------------
        r_on_b = np.right_shift(
            self.index_weighted_sum(shifted_r, *self.channel_indices_and_weights[14]) +
            self.index_weighted_sum(shifted_b, *self.channel_indices_and_weights[1]), 10
        )
        g_on_b = np.right_shift(
            self.index_weighted_sum(shifted_gr, *self.channel_indices_and_weights[6]) +
            self.index_weighted_sum(shifted_gb, *self.channel_indices_and_weights[7]) +
            self.index_weighted_sum(shifted_b, *self.channel_indices_and_weights[0]), 10
        )

        # ---------------- Reconstruction ----------------
        rgb_image = np.dstack([
            reconstruct_bayer((r, r_on_gr, r_on_gb, r_on_b), bayer_pattern='rggb'),
            reconstruct_bayer((g_on_r, gr, gb, g_on_b), bayer_pattern='rggb'),
            reconstruct_bayer((b_on_r, b_on_gr, b_on_gb, b), bayer_pattern='rggb')
        ])
        rgb_image = self.rotate_from_rggb(rgb_image)
        rgb_image = np.clip(rgb_image, 0, self.cfg.saturation_values.hdr)

        data['rgb_image'] = rgb_image.astype(np.uint16)


    def execute_nearest_neighbour(self, data):
        bayer = data['bayer'].astype(np.int32)
        bayer = self.rotate_to_rggb(bayer)  # RGGB pattern

        height, width = bayer.shape

        rgb_img = np.zeros((height, width, 3), dtype=np.uint16)
        for i in range(0, int(height / 2)):
            for j in range(0, int(width / 2)):
                r3 = bayer[2 * i, 2 * j]
                g1 = bayer[2 * i, 2 * j + 1]
                g2 = bayer[2 * i + 1, 2 * j]
                b0 = bayer[2 * i + 1, 2 * j + 1]
                rgb_img[2 * i, 2 * j, :] = [r3, g1, b0]
                rgb_img[2 * i, 2 * j + 1, :] = [r3, g1, b0]
                rgb_img[2 * i + 1, 2 * j, :] = [r3, g2, b0]
                rgb_img[2 * i + 1, 2 * j + 1, :] = [r3, g2, b0]

        rgb_img = self.rotate_from_rggb(rgb_img)
        rgb_img = np.clip(rgb_img, 0, self.cfg.saturation_values.hdr)

        data['rgb_image'] = rgb_img.astype(np.uint16)

    def execute_edge_aware(self, data):
        bayer = data['bayer'].astype(np.uint16)
        bayer = self.rotate_to_rggb(bayer)  # RGGB pattern
        rgb_img = cv.cvtColor(bayer, cv.COLOR_BayerBG2RGB_EA)
        rgb_img = self.rotate_from_rggb(rgb_img)
        rgb_img = np.clip(rgb_img, 0, self.cfg.saturation_values.hdr)
        data['rgb_image'] = rgb_img.astype(np.uint16)