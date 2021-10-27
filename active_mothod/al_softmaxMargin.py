import numpy as np


class SoftMaxMarginSelector(object):
    '''
        基于不确定度方法
        随机选择，通常作为baseline
        过低最大置信度可以被ignore
        分割中 得分计算通过整图所有max confidence位置的confidence之和作为 该图的置信度
    '''

    def __init__(self, confidence_ignore=0.0):
        super().__init__()
        self.confidence_ignore = confidence_ignore
        self.method = 'margin'
        self.reset()

    def reset(self):
        self.remaining_group = []
        self.next_batch_group = []

    def __confidenMarginScore(self, feature):
        # we assume the shape of feature is NxHxW
        pred_max_confidence = feature.max(axis=0) # H W
        ndx = np.indices(feature.shape)
        pred_second_confidence = feature[feature.argsort(0), ndx[1], ndx[2]][-2]
        margin = pred_max_confidence - pred_second_confidence
        score = margin[pred_max_confidence > self.confidence_ignore]
        score = np.mean(score)
        return score

    def addFeature(self, imgPath, feature):
        if self.method == 'margin':
            score = self.__confidenMarginScore(feature)
        self.remaining_group.append((imgPath, score))
    
    def selectNextBatch(self, k):
        self.remaining_group.sort(key=lambda x: x[1])
        for each_index in range(len(self.remaining_group)):
            if each_index >= k:
                break
            self.next_batch_group.append(self.remaining_group[each_index][0])
        return self.next_batch_group


if __name__ == '__main__':
    # test = SoftMaxMarginSelector(confidence_ignore=0.2)
    # test.addFeature('a', np.random.rand(2, 2, 2))
    # test.addFeature('b', np.random.rand(2, 2, 2))
    # test.addFeature('c', np.random.rand(2, 2, 2))

    # print(test.selectNextBatch(1))
    np.random.seed(0)
    test = np.random.rand(2, 2, 2)
    
    ndx = np.indices(test.shape)

    print(test.argsort(0))
    secondScore = test[test.argsort(0), ndx[1], ndx[2]]
    
