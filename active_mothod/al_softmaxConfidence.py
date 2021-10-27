import numpy as np


class SoftMaxConfidenceSelector(object):
    '''
        基于不确定度方法
        随机选择，通常作为baseline
        过低最大置信度可以被ignore
        分割中 得分的计算通过整图所有max confidence位置的confidence之和作为 该图的置信度
    '''

    def __init__(self, confidence_ignore=0.0):
        super().__init__()
        self.confidence_ignore = confidence_ignore
        self.method = 'confidence'
        self.reset()

    def reset(self):
        self.remaining_group = []
        self.next_batch_group = []

    def __confidenScore(self, feature):
        # we assume the shape of feature is NxHxW
        pred_max_confidence = feature.max(axis=0) # H W
        score = pred_max_confidence[pred_max_confidence > self.confidence_ignore]
        score = np.mean(score)
        return score

    def addFeature(self, imgPath, feature):
        if self.method == 'confidence':
            score = self.__confidenScore(feature)
        self.remaining_group.append((imgPath, score))
    
    def selectNextBatch(self, k):
        self.remaining_group.sort(key=lambda x: x[1])
        for each_index in range(len(self.remaining_group)):
            if each_index >= k:
                break
            self.next_batch_group.append(self.remaining_group[each_index][0])
        return self.next_batch_group


if __name__ == '__main__':
    test = SoftMaxConfidenceSelector(confidence_ignore=0.2)
    test.addFeature('a', np.random.rand(2, 2, 2))
    test.addFeature('b', np.random.rand(2, 2, 2))
    test.addFeature('c', np.random.rand(2, 2, 2))

    print(test.selectNextBatch(1))
