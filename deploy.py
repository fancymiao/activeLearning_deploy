import cv2
import onnxruntime
from tqdm import tqdm
import os
from t29_deploy import seg_DeepForward
from active_mothod import RandomSelector


# record time comsuming
class timeRecord(object):
    def __init__(self) -> None:
        super().__init__()
        self.preTime = 0
        self.deepTime = 0
        self.postTime = 0
        self.visTime = 0
        self.account = 0

    def resultShow(self):
        print('preProcess time comsuming: {}'.format(self.preTime / self.account))
        print('deepModule time comsuming: {}'.format(self.deepTime / self.account))
        print('postProcess time comsuming: {}'.format(self.postTime / self.account))
        print('visulization all time: {}'.format(self.visTime))


if __name__ == "__main__":
    ort_session_seg = onnxruntime.InferenceSession('./segGray.onnx')
    imglists = []
    for root, dirs, files in imglists:
        for each_file in files:
            wholePath = os.path.join(root, files)
            imglists.append(wholePath)
    
    imglists.sort()
    print(len(imglists))
    assert len(imglists) > 0
    pbar = tqdm(total=len(imglists))
    index = 0
    time_record = timeRecord()
    tmpIndex = 0

    al_module = RandomSelector()
    for each_list in imglists:
        pbar.update(1)
        tmpIndex += 1
        
        # single image
        img = cv2.imread(each_list)
        output = seg_DeepForward(img.copy(), ort_session_seg)
        al_module.addFeature(each_list, output)
        # print(eachlist)
    
    k = 50 # the choosen image to be labeled
    al_module.selectNextBatch(k)

    print('ng num:', index)
