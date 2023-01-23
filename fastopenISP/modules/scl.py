# File: scl.py
# Description: Scaler
# Created: 2021/10/26 22:36
# Author: Qiu Jueqin (qiujueqin@gmail.com)


from functools import partial
import cv2
from PIL import Image
import numpy as np

from .basic_module import BasicModule


class SCL(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.resize = partial(
            # cv2.resize, dsize=(self.params.width, self.params.height), interpolation=cv2.INTER_LINEAR
        # )

    def execute(self, data):
        # if 'y_image' and 'cbcr_image' in data:
            # data['y_image'] = self.resize(data['y_image'])
            # data['cbcr_image'] = self.resize(data['cbcr_image'])
        # elif 'rgb_image' in data:
            # data['rgb_image'] = self.resize(data['rgb_image'])
        # else:
            # raise NotImplementedError('can not resize Bayer array')
            
        newWidth = self.cfg.scl.newWidth #1024 #int(self.cfg.hardware.raw_width / self.cfg.scl.downsample_factor)
        newHeight = self.cfg.scl.newHeight # 512 #int(self.cfg.hardware.raw_height / self.cfg.scl.downsample_factor)


        imgArr = data['rgb_image'].astype(np.uint8)
        img = Image.fromarray(imgArr,"RGB")
        data['rgb_image'] = np.array(img.resize((newWidth,newHeight))).astype(np.uint16)
        #print(data['rgb_image'].shape)
        #cv2.imshow("After Scale",data['rgb_image'].astype(np.uint8))
        #print(data['rgb_image'].shape)
        #cv2.waitKey(0)
