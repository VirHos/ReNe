import random

import sentencepiece as spm


class VocabBuilder:
    def __init__(
        self,
        path,
        model_prefix="tokenizer",
        voc_size=3200,
        num_placeholders=256,
        subsample_size=12800000,
        voc_fname="vocab.txt",
    ):
        self.path = path
        self.model_prefix = model_prefix
        self.voc_size = voc_size
        self.num_placeholders = num_placeholders
        self.subsample_size = subsample_size
        self.voc_fname = voc_fname

    def read_sentencepiece_vocab(self, filepath):
        voc = []
        with open(filepath, encoding="utf-8") as fi:
            for line in fi:
                voc.append(line.split("\t")[0])
            # skip the first <unk> token
        voc = voc[1:]
        return voc

    def parse_sentencepiece_token(self, token):
        if token.startswith("▁"):
            return token[1:]
        else:
            return "##" + token

    def write_to_file(self, filename, bert_vocab):
        with open(filename, "w") as fo:
            for token in bert_vocab:
                fo.write(token + "\n")

    def build_vocab(self):
        spm_command = (
            "--input={} --model_prefix={} "
            "--vocab_size={} --input_sentence_size={} "
            "--shuffle_input_sentence=true "
            "--bos_id=-1 --eos_id=-1"
        ).format(
            self.path,
            self.model_prefix,
            self.voc_size - self.num_placeholders,
            self.subsample_size,
        )
        spm.SentencePieceTrainer.Train(spm_command)
        snt_vocab = self.read_sentencepiece_vocab("{}.vocab".format(self.model_prefix))
        print("Learnt vocab size: {}".format(len(snt_vocab)))
        print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))
        bert_vocab = list(map(self.parse_sentencepiece_token, snt_vocab))
        ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        bert_vocab = ctrl_symbols + bert_vocab
        bert_vocab += [
            "[UNUSED_{}]".format(i) for i in range(self.voc_size - len(bert_vocab))
        ]
        self.write_to_file(self.voc_fname, bert_vocab)
