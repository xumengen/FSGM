# -*- encoding: utf-8 -*-
'''
@File    :   draw.py
@Time    :   2021/06/23 15:54:23
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import torch
import numpy as np
from PIL import Image


if __name__ == "__main__":
    # result = torch.load("./debug.pt").numpy()
    # img = Image.fromarray((result*255).astype(np.uint8))
    # img.save("debug.png")

    # result = torch.load("./debug1.pt").numpy()
    # img = Image.fromarray((result*255).astype(np.uint8))
    # img.save("debug1.png")

    img = Image.open("/home/mengen/PANet/data/Pascal/VOCdevkit/VOC2012/SegmentationClassAug/2007_000032.png")
    result = np.array(img)
    print(np.unique(img))

