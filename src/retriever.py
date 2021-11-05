import numpy as np


class Retriever:
    def __init__(self, index, encoder, output_idx_storage: np.array, use_last=5):
        self.encoder = encoder
        self.index = index
        self.output_idx_storage = output_idx_storage

    def __call__(self, user_input, n=5):
        cut_user_input = user_input[-self.use_last :]
        slice = []
        if cut_user_input:
            user_emb = self.encoder(cut_user_input)
            dist, idx = self.index.search(user_emb, n)
            slice = idx[0]
        return self.output_idx_storage[slice]
