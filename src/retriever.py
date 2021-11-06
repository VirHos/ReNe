import numpy as np
from faiss import IndexFlatIP
from utils import timer

class Retriever:
    def __init__(self, index: IndexFlatIP, encoder, output_idx_storage: np.array, use_last=5):
        self.encoder = encoder
        self.index = index
        self.output_idx_storage = output_idx_storage
        self.use_last = use_last

    def __call__(self, user_input, n=16):
        cut_user_input = user_input[-self.use_last :]
        slice = []
        if cut_user_input:
            user_emb = self.encoder(cut_user_input)
            dist, idx = self.index.search(user_emb, n)
            slice = idx[0]
        return self.output_idx_storage[slice]

    def add_news_to_index(self, meta_str, news_idx):
        meta_emb = self.encoder.labse_embed(meta_str)
        self.output_idx_storage = np.append(self.output_idx_storage, news_idx)
        self.index.add(meta_emb)
        return 'ok'
