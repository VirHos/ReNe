#simple encoder class for batch encoding

import numpy as np
import faiss
from utils import batch

class SimpleEncoder:
  def __init__(self, texts, embs):
    self.texts = texts
    self.embs = embs
    self.cache = dict(zip(texts, embs))

  def __call__(self, texts):
      return np.array([self.cache[t] for t in texts])
