import json
import logging
import pickle
import time
from contextlib import contextmanager

import faiss
import tensorflow as tf
import yaml


@contextmanager
def timer(name, disable=False):
    """Simple timer as context manager."""

    start = time.time()
    yield
    if not disable:
        logging.info(f"[{name}] done in {(time.time() - start)*1000:.1f} ms")


def pickle_load(path):
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    return pkl


def pickle_dump(o, fpath: str):
    with open(fpath, "wb") as fi:
        pickle.dump(o, fi, protocol=pickle.HIGHEST_PROTOCOL)
    return 1


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def json_load(fp):
    with open(fp, "r") as f:
        js = json.loads(f.read())
    return js


def yaml_load(fpath: str):
    with open(fpath, "r") as f:
        yml = yaml.safe_load(f)
    return yml


def build_faiss_index(all_embs, gpu=False):
    index = faiss.IndexFlatIP(all_embs.shape[1])
    # make it into a gpu index
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(all_embs)
    index.add(all_embs)
    return index


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def get_meta_str(di):
    tags_str, sphere_str = "", ""
    if di["tags"]:
        tags = [t["title"] for t in di["tags"]]
        tags_str = "".join(["просмотренные теги:", ", ".join(tags).lower()])
    if di["spheres"]:
        spheres = [t["title"] for t in di["spheres"]]
        sphere_str = "".join(["просмотренные сферы:", ", ".join(spheres).lower()])

    meta_str = ". ".join(list(filter(None, [tags_str, sphere_str])))
    return meta_str
