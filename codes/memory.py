import numpy as np
import random
import warnings
from collections import namedtuple


Experience = namedtuple('Experience', 'items_1a,items_2a,items_1b,items_2b,user_emb_1,user_emb_2')


class RingBuffer(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class CycleMemory(object):

    def __init__(self, memory_size):
        self.limit = memory_size  # memory size

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.items_1a = RingBuffer(self.limit)
        self.items_2a = RingBuffer(self.limit)
        self.items_1b = RingBuffer(self.limit)
        self.items_2b = RingBuffer(self.limit)
        self.user_emb_1 = RingBuffer(self.limit)
        self.user_emb_2 = RingBuffer(self.limit)
        self.rng = np.random.RandomState(123)

    def append(self, items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2):
        self.items_1a.append(items_1a)
        self.items_2a.append(items_2a)
        self.items_1b.append(items_1b)
        self.items_2b.append(items_2b)
        self.user_emb_1.append(user_emb_1)
        self.user_emb_2.append(user_emb_2)

    def sample_batch_indexes(self, low, high, size):
        if high - low >= size:  # enough data
            try:
                r = range(low, high)
            except NameError:
                r = range(low, high)
            batch_idxs = self.rng.choice(r, size, replace=False)
        else:
            warnings.warn("not enough data")
            batch_idxs = self.rng.random_integers(low, high - 1, size=size)

        assert len(batch_idxs) == size
        return batch_idxs

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = self.sample_batch_indexes(0, len(self.items_1a)-1, batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < len(self.items_1a)
        assert len(batch_idxs) == batch_size

        # create experience
        experiences = []
        for idx in batch_idxs:
            items_1a = self.items_1a[idx - 1]
            items_2a = self.items_2a[idx - 1]
            items_1b = self.items_1b[idx - 1]
            items_2b = self.items_2b[idx - 1]
            user_emb_1 = self.user_emb_1[idx - 1]
            user_emb_2 = self.user_emb_2[idx - 1]

            experiences.append(Experience(items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2))

        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size):
        items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2 = [], [], [], [], [], []

        experience = self.sample(batch_size, batch_idxs=None)

        for e in experience:
            items_1a.append(e.items_1a)
            items_2a.append(e.items_2a)
            items_1b.append(e.items_1b)
            items_2b.append(e.items_2b)
            user_emb_1.append(e.user_emb_1)
            user_emb_2.append(e.user_emb_2)

        items_1a = np.array(items_1a)
        items_2a = np.array(items_2a)
        items_1b = np.array(items_1b)
        items_2b = np.array(items_2b)
        user_emb_1 = np.array(user_emb_1)
        user_emb_2 = np.array(user_emb_2)  # TODO change must!

        return items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2

