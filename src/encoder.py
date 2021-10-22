class Recommender:

    def __init__(self, retriever, encoder, news, complex_filter):
        self.retriever = retriever
        self.news = news
        self.encoder = encoder
        self.complex_filter = complex_filter

    def get_news(self, read_news):
        embedding = self.encoder(read_news)
        news = self.retriever(embedding)
        filtered_news = self.complex_filter(news)
        return filtered_news

class Retriever:
    
    def __init__(self, index, ques):
        self.ques = ques
        self.index = index
    
    def __call__(self, emb, n=3):
      dist, idx = self.index.search(emb, n)
      ques = self.ques[idx[0]]
      return ques

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

#simple encoder class for batch encoding

import numpy as np
import faiss

class Encoder:
  def __init__(self, tok, model):
    self.tok = tok
    self.model = model

  def encode(self, texts):
      return np.random.uniform(size=(len(texts),768)).astype('float32')
    # t = self.tok.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    # with torch.no_grad():
    #     model_output = self.model(**{k: v.to(model.device) for k, v in t.items()})
    # embeddings = model_output.pooler_output
    # embeddings = torch.nn.functional.normalize(embeddings)
    # return embeddings


  def __call__(self, texts, bs=128):
    embs = []
    for b in batch(texts, bs):
      embs.append(self.encode(b))
    return np.vstack(embs)


def build_faiss_index(all_embs, gpu=False):
    index = faiss.IndexFlatIP(all_embs.shape[1])
    # make it into a gpu index
    if gpu:
      res = faiss.StandardGpuResources()
      index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(all_embs)
    index.add(all_embs)
    return index
