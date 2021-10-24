# cache all question representations in encoder object
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from utils import batch


class CacheEncoder:
    def __init__(
        self,
        base_encoder,
        sents=None,
        vectors=None,
        dim=(768,),
        dtype="float32",
        preprocessor=None,
        bsize=128,
    ):
        self.cached_vectors = vectors
        self.dim = dim
        self.stoid = {}
        self.encoder = base_encoder
        self._dtype = dtype
        self.bsize = bsize
        self.preprocessor = preprocessor
        self.check_cache(sents, vectors)

    def check_cache(self, sents, vectors):
        if sents is not None:
            self.stoid = {s: i for i, s in enumerate(sents)}
            self.cached_vectors = vectors

    def encode_new(self, texts, verbose):
        bgen = batch(texts, n=self.bsize)

        encoded = []
        if verbose :
            bgen = tqdm(bgen, total=len(texts)//self.bsize)
        for bt in bgen:
            enc = self.encoder(bt)
            encoded.append(enc)
        mat = np.vstack(encoded).astype(self._dtype)
        return mat

    def __call__(self, texts, verbose=False):
        if self.preprocessor:
            texts = list(map(self.preprocessor, texts))
        new_cache_np = np.zeros((len(texts), *list(self.dim)), self._dtype)
        new_samples = []
        new_sample_indices = []
        new_cache_indices_list = []
        old_cache_indices_list = []
        for index, question in enumerate(texts):
            if question in self.stoid:
                old_cache_indices_list.append(self.stoid[question])
                new_cache_indices_list.append(index)
            else:
                new_samples.append(question)
                new_sample_indices.append(index)

        new_cache_indices_np = np.array(new_cache_indices_list, dtype="int")
        old_cache_indices_np = np.array(old_cache_indices_list, dtype="int")

        if self.cached_vectors is not None:
            new_cache_np[new_cache_indices_np] = self.cached_vectors[
                old_cache_indices_np
            ]
        if len(new_samples) > 0:
            new_sample_indices = np.array(new_sample_indices)
            new_vectorized_samples = self.encode_new(new_samples, verbose)
            new_cache_np[new_sample_indices] = new_vectorized_samples

        return new_cache_np

class Encoder:
  def __init__(self, tok, model):
    self.tok = tok
    self.model = model

  def embed(self, texts):
    t = self.tok.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = self.model(**t)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


  def __call__(self, texts, bs=128):
    embs = []
    for b in batch(texts, bs):
      embs.append(self.embed(b))
    return torch.vstack(embs).numpy()

class SimpleEncoder:
    def __init__(self, texts: List[str], embs: np.array):
        self.texts = texts
        self.embs = embs
        self.cache = dict(zip(texts, embs))

    def __call__(self, texts):
        return np.array([self.cache[t] for t in texts])
