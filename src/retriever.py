from utils import batch

class Retriever:
    
    def __init__(self, index, encoder):
        self.encoder = encoder
        self.index = index
    
    def __call__(self, user_input, n=3):
      user_emb = self.encoder(user_input)

      dist, idx = self.index.search(user_emb, n)
      return idx[0]