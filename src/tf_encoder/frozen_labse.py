import logging

import numpy as np
import tensorflow as tf


def build_proto_config(cpu_cores=4, allow_growth=True, use_gpu=True):
    config = tf.ConfigProto(device_count={"GPU": 1 if use_gpu else 0})
    config.gpu_options.allow_growth = allow_growth
    return config


def get_configurable_session_and_graph(config=None):
    if config is None:
        config = build_proto_config()

    graph = tf.compat.v1.Graph()
    sess = tf.compat.v1.Session(config=config, graph=graph)

    return {"session": sess, "graph": graph}


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def _get_masks(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        raise IndexError("Token lengath more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [0] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _trim_input(text, max_seq_length, tokenizer):

    t = tokenizer.tokenize(text)

    if len(t) > max_seq_length - 2:
        t = t[0 : (max_seq_length - 2)]

    return t


def _get_ids(tokens, tokenizer, max_seq_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _convert_to_bert_inputs(text, tokenizer, max_seq_length):
    # stacked token
    stoken = ["[CLS]"] + text + ["[SEP]"]
    input_ids = _get_ids(
        tokens=stoken, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    input_masks = _get_masks(tokens=stoken, max_seq_length=max_seq_length)
    input_segments = _get_segments(tokens=stoken, max_seq_length=max_seq_length)

    return [input_ids, input_masks, input_segments]


def compute_input_array(texts, tokenizer, max_seq_length):
    input_ids, input_masks, input_segments = [], [], []
    for text in texts:
        t = text
        t = _trim_input(t, max_seq_length, tokenizer)
        ids, masks, segments = _convert_to_bert_inputs(t, tokenizer, max_seq_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [
        np.array(input_ids, dtype=np.int32),
        np.array(input_masks, dtype=np.int32),
        np.array(input_segments, dtype=np.int32),
    ]


class LaBSE:
    def __init__(self, model_path, tokenizer, seq_len=24):
        self.max_seq_length = seq_len
        self.tokenizer = tokenizer
        self.restored_graph = self._load_graph(model_path)
        self.sess = tf.compat.v1.Session(graph=self.restored_graph)
        # первый запуск всегда долгий поэтому в конструкторе
        self.x_id, self.x_mask, self.x_seg, self.y1 = self.restore_model(
            self.restored_graph
        )
        _ = self.predict(
            [[0] * self.max_seq_length],
            [[0] * self.max_seq_length],
            [[0] * self.max_seq_length],
        )

    def _load_graph(self, frozen_graph_filename):
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.compat.v1.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
        return graph

    def restore_model(self, restored_graph):

        graph_ops = restored_graph.get_operations()
        input_op, output_op = [
            graph_ops[0].name,
            graph_ops[1].name,
            graph_ops[2].name,
        ], [graph_ops[-1].name]

        [x_id, x_mask, x_seg] = [
            restored_graph.get_tensor_by_name(input_op[i] + ":0")
            for i, k in enumerate(input_op)
        ]
        y1 = restored_graph.get_tensor_by_name(output_op[0] + ":0")
        return x_id, x_mask, x_seg, y1

    def predict(self, in_ids, in_masks, in_segs):
        return self.sess.run(
            self.y1,
            feed_dict={self.x_id: in_ids, self.x_mask: in_masks, self.x_seg: in_segs},
        )

    def get_embs(self, texts):

        if type(texts) != list:
            texts = [texts]

        in_ids, in_masks, in_segs = compute_input_array(
            texts, self.tokenizer, self.max_seq_length
        )

        outs = self.predict(in_ids, in_masks, in_segs)
        return {"sentence_embs": outs}

    def __call__(self, texts):
        out_dict = self.get_embs(texts)
        return out_dict["sentence_embs"]


def build_ctx_model(ctx_len=5, dim=768):

    inp_emb = tf.keras.layers.Input(
        shape=(ctx_len, dim), dtype=tf.float32, name="nlu_input"
    )

    d1 = tf.keras.layers.Dense(1024, activation="tanh", name="d1")
    d2 = tf.keras.layers.Dense(768, activation="tanh", name="d2")

    enc_q = tf.reshape(inp_emb, (-1, ctx_len, 768))
    enc_q = tf.keras.layers.GlobalAveragePooling1D()(enc_q)
    enc_q = d1(enc_q)
    enc_q = d2(enc_q)

    final_state_encoded = tf.identity(enc_q, "state_output")
    t_model = tf.keras.models.Model(inputs=[inp_emb], outputs=[final_state_encoded])

    return t_model
