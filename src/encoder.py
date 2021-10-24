from typing import List

import numpy as np

from utils import batch


class SimpleEncoder:
    def __init__(self, texts: List[str], embs: np.array):
        self.texts = texts
        self.embs = embs
        self.cache = dict(zip(texts, embs))

    def __call__(self, texts):
        return np.array([self.cache[t] for t in texts])
