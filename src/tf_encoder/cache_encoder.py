# cache all question representations in encoder object
import os

import numpy as np
from tensorflow.keras.utils import Progbar

from tf_encoder.frozen_labse import LaBSE, build_ctx_model
from tf_encoder.text_preprocessing import FullTokenizer
from utils import batch, load_graph, timer, yaml_load


def get_nlu_executor(config):
    vocab = os.path.join(config["vocab"])
    model_path = os.path.join(config["model_path"])
    seq_len = config.get("seq_len", 24)
    tokenizer = FullTokenizer(vocab_file=vocab, do_lower_case=True)
    nlu_executor = LaBSE(model_path=model_path, tokenizer=tokenizer, seq_len=seq_len)
    return nlu_executor


def get_state_encoder(config, nlu_executor):
    ctx_model = build_ctx_model()
    ctx_model.load_weights(config["ctx_model"])
    se = StackedEncoder(nlu_executor, ctx_model)
    return se


class StackedEncoder:
    def __init__(self, labse, ctx_model):
        self.labse = labse
        self.ctx_model = ctx_model

    def __call__(self, sents):
        emb = self.labse(sents)
        reshaped = emb.reshape((1, emb.shape[0], emb.shape[1]))
        user_emb = self.ctx_model.predict(reshaped)
        return user_emb

    def labse_embed(self, texts):
        if type(texts) != list:
            texts = [texts]
        return self.labse(texts)


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

    def update_cache_sample(self, text, emb):
        new_cache = np.vstack((self.cached_vectors, emb))
        self.cached_vectors = new_cache
        self.stoid[text] = len(self.cached_vectors) - 1

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
