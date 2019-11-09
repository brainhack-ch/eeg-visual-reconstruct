import numpy as np
import collections
from torch.utils import data


class SiameseSampler(data.Sampler):
    def __init__(self, data_source, batch_size, n_batches):
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        return Iterator(self.data_source, self.batch_size, self.n_batches)

    def __len__(self):
        return self.n_batches


class Iterator(collections.Iterator):
    def __init__(self, data_source, batch_size, n_batches):
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            self.current_batch += 1
            data_classes = [int(i) for i in self.data_source.classes]

            classes = np.array(
                [np.random.choice(
                    data_classes, size=2, replace=False
                ) for i in range(self.batch_size // 2)] +
                [[np.random.choice(
                    data_classes, size=1, replace=True
                )] * 2 for i in range(self.batch_size // 2)])

            class_inds = {int(i): [] for i in data_classes}
            for i, sample in enumerate(self.data_source.samples):
                class_inds[sample[1]].append(i)

            def sample_from_class(i):
                return np.random.choice(class_inds[i])

            classes = list(map(sample_from_class, classes.flatten('A')))
            return classes
        raise StopIteration()
