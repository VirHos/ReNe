import logging
from typing import List

import numpy as np
import tensorflow as tf


class FrozenTFModel:
    LIMIT_DATA_COUNT_WITHOUT_COPING_TENSOR_DATA = 16000

    def __init__(self, model_graph: tf.compat.v1.GraphDef, **config):

        self._model_graph = model_graph

        self._name: str = config.get("name", "FrozenTFModel")

        input_tensor_names = config.get("input_tensors_names")
        output_tensor_names = config.get("output_tensors_names")

        graph_ops = self._model_graph.get_operations()
        if not input_tensor_names:
            amount_of_inputs = config.get("amount_of_inputs", 1)
            input_tensor_names = [self._get_tensor_name(op) for op in graph_ops[:amount_of_inputs]]
        if not output_tensor_names:
            output_tensor_names = [self._get_tensor_name(graph_ops[-1])]

        self.logger = logging.getLogger(__name__ + self._name)
        self._build(input_tensors_names=input_tensor_names,
                    output_tensors_names=output_tensor_names)

        self.logger.info("Model " + self._name + " is initialized.")

    def _build(self, input_tensors_names: List[str],
               output_tensors_names: List[str]) -> None:

        self._input_tensors = []
        self._output_tensors = []

        for input_tensor_name in input_tensors_names:
            self._input_tensors.append(self._model_graph.get_tensor_by_name(input_tensor_name))

        for output_tensor_name in output_tensors_names:
            self._output_tensors.append(self._model_graph.get_tensor_by_name(output_tensor_name))

        self.session = tf.compat.v1.Session(graph=self._model_graph)

    def __call__(self, input_arrays: List[np.array]):

        input_dict = {}

        for inp_t, inp_arr in zip(self._input_tensors, input_arrays):
            input_dict[inp_t] = inp_arr

        pred = self.session.run(self._output_tensors, input_dict)

        return pred

    @staticmethod
    def _get_tensor_name(op) -> str:
        return op.name + ':0'

    @staticmethod
    def __get_shape(tensor_list: List[tf.Tensor]) -> List[tf.TensorShape]:
        shapes_list = []
        for el in tensor_list:
            shapes_list.append(el.shape.as_list())
        return shapes_list

    def get_output_shape(self) -> List[tf.TensorShape]:
        return self.__get_shape(self._output_tensors)

    def get_input_shape(self) -> List[tf.TensorShape]:
        return self.__get_shape(self._input_tensors)

    @staticmethod
    def get_method_for_output_tensors(input_data: List[np.ndarray]):
        """
        Add case for large count input data with coping array for release shared memory
        """
        max_len = max((arr.shape[0] for arr in input_data))
        if max_len <= FrozenTFModel.LIMIT_DATA_COUNT_WITHOUT_COPING_TENSOR_DATA:
            return lambda x: x
        else:
            return lambda x: [np.array(el, dtype=el.dtype) for el in x]

