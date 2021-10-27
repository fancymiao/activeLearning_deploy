import numpy.random as rd


class RandomSelector(object):
    '''
        随机选择，通常作为baseline
        初始化seed可选择。
    '''

    def __init__(self, seed=0):
        super().__init__()
        rd.seed(seed)
        self.reset()

    def reset(self):
        self.remaining_group = []
        self.next_batch_group = []

    def addFeature(self, imgPath, feature):
        self.remaining_group.append((imgPath, feature))
    
    def selectNextBatch(self, k):
        rd.shuffle(self.remaining_group)
        for each_index in range(len(self.remaining_group)):
            if each_index >= k:
                break
            self.next_batch_group.append(self.remaining_group[each_index][0])
        return self.next_batch_group


if __name__ == '__main__':
    test = RandomSelector()
    test.addFeature('a', 'a')
    test.addFeature('b', 'a')
    test.addFeature('c', 'a')

    print(test.selectNextBatch(1))
