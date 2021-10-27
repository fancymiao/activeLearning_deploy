import numpy as np


class SoftMaxConfidenceSelector(object):
    '''
        基于不确定度方法
        随机选择，通常作为baseline
        过低最大置信度可以被ignore
        得分通过 逐channel的softmax熵之和 得出
    '''

    def __init__(self, confidence_ignore=0.0):
        super().__init__()
        self.confidence_ignore = confidence_ignore
        self.method = 'entropy'
        self.reset()

    def reset(self):
        self.remaining_group = []
        self.next_batch_group = []

    def __entropyScore(self, feature):
        # we assume the shape of feature is NxHxW
        entropy_map = np.zeros((feature.shape[1], feature.shape[1]))
        for each_index in range(feature.shape[0]):
            now_feature = feature[each_index, :, :]
            entropy_map -= now_feature * np.log(now_feature)
        
        pred_max_confidence = feature.max(axis=0) # H W
        score = entropy_map[pred_max_confidence > self.confidence_ignore]
        score = np.mean(score)
        return score

    def addFeature(self, imgPath, feature):
        if self.method == 'entropy':
            score = self.__entropyScore(feature)
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
