from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from tf_encoder.frozen import FrozenTFModel
from utils import batch
from wp.core import build_preprocessor


class FrozenBatchedTFModel(FrozenTFModel):
    def __init__(self, model_graph: tf.compat.v1.GraphDef, **config):
        super().__init__(model_graph, **config)

        self.config = config

    def __call__(self, input_arrays: List[np.array], verbose=0):

        input_arrays = self.check_input_arrays(input_arrays)
        input_sizes = [arr.shape[0] for arr in input_arrays]
        assert (
            len(set(input_sizes)) == 1
        ), f"All input arrays must have equal first shapes, got {set(input_sizes)}"
        assert len(input_sizes) == len(
            self._input_tensors
        ), f"Passed input is incompatible, expect {len(self._input_tensors)} arrays, got {len(input_sizes)}"

        predictions = []
        stacked_predictions = []
        input_len = input_sizes[0]
        input_idx = np.arange(0, input_len)
        batched_idxs = batch(input_idx, self.config.get("batch_size", 16))
        bar = Progbar(input_len)
        method_for_output_tensors = self.get_method_for_output_tensors(input_arrays)

        for batch_ix in batched_idxs:
            batch_arrs = [arr[batch_ix] for arr in input_arrays]
            feed_dict = {
                tensor: mat for tensor, mat in zip(self._input_tensors, batch_arrs)
            }
            batch_pred = self.session.run(self._output_tensors, feed_dict=feed_dict)
            predictions.append(method_for_output_tensors(batch_pred))
            if verbose:
                bar.add(batch_ix.shape[0])

        if len(predictions):
            out_buf_size = len(predictions[0])
            out_mat_sizes = [arr.shape[1:] for arr in predictions[0]]
            stacked_predictions = []
            for i in range(out_buf_size):
                stack_shape = (input_len,) + out_mat_sizes[i]
                stack_array = np.zeros(stack_shape, dtype="float32")
                ptr = 0
                for arrs in predictions:
                    cur_arr = arrs[i]
                    cur_len = cur_arr.shape[0]
                    stack_array[ptr : ptr + cur_len] = cur_arr
                    ptr += cur_len
                stacked_predictions.append(stack_array)

        return self.check_output_arrays(stacked_predictions)

    @staticmethod
    def check_input_arrays(input_arrays):
        input_type = type(input_arrays)
        allowed_types = {list, tuple}
        string_types = {str, np.str_}

        if input_type not in allowed_types:
            if input_type is np.ndarray:
                input_arrays = [input_arrays]
            elif input_type in string_types:
                input_arrays = [np.atleast_1d(input_arrays)]
            else:
                raise ValueError(f"Unsupported input format [{type(input_arrays)}]")
        else:
            input_dtypes = [type(arr) for arr in input_arrays]
            if len(set(input_dtypes)) == 1 and input_dtypes[0] in string_types:
                input_arrays = [np.array(input_arrays)]
            else:
                if not all([i_type is np.ndarray for i_type in input_dtypes]):
                    raise ValueError(f"Unsupported input format [{input_dtypes[:5]}]")
        return input_arrays

    @staticmethod
    def check_output_arrays(output_arrays):
        if isinstance(output_arrays, list) and len(output_arrays) == 1:
            output_arrays = output_arrays[0]
        return output_arrays


class FrozenBert(FrozenBatchedTFModel):
    def __init__(self, model_graph: tf.compat.v1.GraphDef, **config) -> None:
        super().__init__(model_graph, **config)

        self.config = config
        self.seq_len = config["seq_len"]
        self.preprocessor = config.get("preprocessor")
        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor()
        self.dim = self.get_output_shape()[-1][-1]
        self.cur_query_arrays = None

    def _build_preprocessor(self):
        preprocessor = build_preprocessor(
            self.config["vocab_path"],
            self.config["seq_len"],
            self.config["do_lower_case"],
        )
        return preprocessor

    def __call__(self, input_arrays: List[np.array], verbose=False) -> np.ndarray:
        input_arrays = self.check_input_arrays(input_arrays)
        input_strings = input_arrays[0]
        query = np.atleast_1d(input_strings)
        self.all_query_preprocessing = self.preprocessor(query)
        query_bert_arrays = self.all_query_preprocessing
        if self.config.get("return_tokenization_map", False):
            query_bert_arrays = self.all_query_preprocessing[0]
        return super().__call__(query_bert_arrays, verbose=verbose)
