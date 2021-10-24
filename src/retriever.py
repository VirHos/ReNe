import numpy as np

from utils import batch


class Retriever:
    def __init__(self, index, encoder, output_idx_storage: np.array, use_last=16):
        self.encoder = encoder
        self.index = index
        self.output_idx_storage = output_idx_storage
        self.use_last = use_last

    def __call__(self, user_input, n=16):
        cut_user_input = user_input[-self.use_last:]
        user_emb = self.encoder(cut_user_input)
        mean_user_emb = np.mean(user_emb, axis=0).reshape((1, -1))

        dist, idx = self.index.search(mean_user_emb, n)
        return self.output_idx_storage[idx[0]]
