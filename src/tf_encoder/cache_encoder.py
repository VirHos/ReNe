# cache all question representations in encoder object
import os

import numpy as np
from tensorflow.keras.utils import Progbar

from tf_encoder.frozen_batch import FrozenBatchedTFModel, FrozenBert
from utils import batch, load_graph, yaml_load


def get_nlu_executor(config, bsize=128):
    vocab = os.path.join(config["vocab"])
    model_path = os.path.join(config["model_path"])
    nlu_graph = load_graph(model_path)
    nlu_executor = FrozenBert(nlu_graph, vocab_path=vocab, **config["nlu_config"])
    return nlu_executor


def get_state_encoder(config, nlu_executor):
    cfg = yaml_load(os.path.join(config["static_path"], config["scu_config"]))
    cfg["batch_size"] = 96
    model_path = os.path.join(config["static_path"], cfg["model_path"])
    scu_graph = load_graph(model_path)
    scu_executor = FrozenBatchedTFModel(scu_graph, **cfg)

    se = StackedEncoder(nlu_executor, scu_executor)
    #     se = CacheEncoder(se)
    return se


class StackedEncoder:
    def __init__(self, nlu, scu):
        self.nlu = nlu
        self.scu = scu

    def __call__(self, list_of_ctx, verbose=False, ctx_size=6):
        flatten = np.hstack(list_of_ctx)
        nlu_emb = self.nlu(flatten, verbose)
        nlu_emb_reshaped = nlu_emb  # .reshape((1, nlu_emb.shape[0], nlu_emb.shape[1]))
        states = self.scu(nlu_emb_reshaped, verbose)
        return states[: len(list_of_ctx)]


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

        bar = Progbar(len(texts))
        encoded = []
        for bt in bgen:

            enc = self.encoder(bt)
            encoded.append(enc)
            if verbose:
                bar.add(len(bt))
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
