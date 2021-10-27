import cv2
import onnxruntime
import numpy as np
import torch
import glob
from tqdm import tqdm
import os
import shutil
import time
import json


def imgTransform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    return img


def seg_DeepForward(img, ort_session):

    img_org = np.float32(img[0])
    img_transed = imgTransform(img_org)

    # add zeros if need
    x = torch.zeros((1, 3, 240, 240), requires_grad=False).cuda()
    x[0, :, :img_transed.shape[1], :img_transed.shape[2]] = img_transed

    t2 = time.perf_counter()
    time_record.preTime += (t2 - t1)

    # inference
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    outputs = ort_session.run(None, ort_inputs)[0]

    return outputs


if __name__ == "__main__":
    ort_session_seg = onnxruntime.InferenceSession('./segGray.onnx')
    # ort_session = (ort_session_seg, ort_session_cls)
    imglists = glob.glob('/home/zhenyuyi/project/29_scc_wuxi/test/testData/avi_out_0624_J_V_202109/3_301113762A&14458909-003/Defect/*.bmp')
    
    imglists.sort()
    print(len(imglists))
    assert len(imglists) > 0
    pbar = tqdm(total=len(imglists))
    index = 0
    time_record = timeRecord()


    tmpIndex = 0
    for eachlist in imglists:
        pbar.update(1)
        tmpIndex += 1

        imgName = eachlist.split('/')[-1]
        fileName = eachlist.split('/')[-2]
        dataName = eachlist.split('/')[-3]
        # print(eachlist)
        img = cv2.imread(eachlist, 1)

    print('ng num:', index)
