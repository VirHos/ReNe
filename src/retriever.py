from utils import batch

class Retriever:
    
    def __init__(self, index, encoder, output_idx_storage):
        self.encoder = encoder
        self.index = index
        self.output_idx_storage = output_idx_storage
    
    def __call__(self, user_input, n=3):
      user_emb = self.encoder(user_input)

      dist, idx = self.index.search(user_emb, n)
      return self.output_idx_storage[idx[0]]