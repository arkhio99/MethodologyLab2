import numpy as np
import pandas as pd


class HierarchyAnalysisMatrix:
    _matrix_features = None
    _dict_matrix_examples = None
    _features = None
    _examples = None
    _rand_concord_table = [None, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

    def __init__(self, features: list, examples: list):
        self._features = features
        self._examples = examples
        self._matrix_features = pd.DataFrame(data=np.eye(len(features)),
                                             columns=features,
                                             index=features)

        self._dict_matrix_examples = {}
        for feature in features:
            self._dict_matrix_examples[feature] = pd.DataFrame(data=np.eye(len(examples)),
                                                               columns=examples,
                                                               index=examples)

    def set_value_feature(self, dominant_feature: str, recessive_feature: str, value: float):
        self._matrix_features.at[recessive_feature, dominant_feature] = value
        self._matrix_features.at[dominant_feature, recessive_feature] = 1 / value

    def get_value_feature(self, dominant_feature: str, recessive_feature: str):
        return self._matrix_features.at[recessive_feature, dominant_feature]

    def set_value_example_by_feature(self, feature: str, dominant_example: str, recessive_example: str, value: float):
        self._dict_matrix_examples[feature].at[recessive_example, dominant_example] = value
        self._dict_matrix_examples[feature].at[dominant_example, recessive_example] = 1 / value

    def get_value_example_by_feature(self, feature: str, dominant_example: str, recessive_example: str):
        return self._dict_matrix_examples[feature].at[recessive_example, dominant_example]

    def get_features(self):
        return self._features

    def get_examples(self):
        return self._examples

    def get_feature_priority_vector(self):
        return HierarchyAnalysisMatrix._get_priority_vector(self._matrix_features)

    def get_example_priority_vector(self, feature: str):
        return HierarchyAnalysisMatrix._get_priority_vector(self._dict_matrix_examples[feature])

    def get_feature_concord_data(self):
        return HierarchyAnalysisMatrix._get_concord_data(self._matrix_features)

    def get_example_concord_data(self, feature):
        return HierarchyAnalysisMatrix._get_concord_data(self._dict_matrix_examples[feature])

    def calculate_result_priorities_for_examples(self):
        feature_priorities = self.get_feature_priority_vector()
        features = self.get_features()
        examples = self.get_examples()

        result = []
        for feature in features:
            prior_vector = self.get_example_priority_vector(feature)
            result.append(prior_vector)

        result = np.array(result)
        result = result.transpose()

        result_matrix = pd.DataFrame(result, index=examples, columns=feature_priorities)
        result_priority_vector = []
        for example in examples:
            val = 0
            for p in feature_priorities:
                val += result_matrix.at[example, p] * p

            result_priority_vector.append(val)

        result_matrix["Priority"] = result_priority_vector

        return result_matrix

    @staticmethod
    def _get_priority_vector(matrix: pd.DataFrame):
        f_count = len(matrix)
        priority_vector = np.empty(f_count)
        for i in range(0, f_count):
            req_list = matrix.iloc[i]
            priority_vector[i] = pow(np.prod(req_list), 1 / len(req_list))

        norm_priority_vector = priority_vector / np.sum(priority_vector)

        return norm_priority_vector

    @staticmethod
    def _get_concord_data(matrix: pd.DataFrame):
        eigen_value = HierarchyAnalysisMatrix._get_eigen_value(matrix)
        n = matrix.shape[0]
        concord_index = (eigen_value - n) / (n - 1)
        concord_estimate = concord_index / HierarchyAnalysisMatrix._rand_concord_table[n] * 100
        return pd.Series([eigen_value, concord_index, concord_estimate],
                         index=['EigenValue', 'ConcordIndex', 'ConcordIndex'])

    @staticmethod
    def _get_eigen_value(matrix: pd.DataFrame):
        prior_vector = HierarchyAnalysisMatrix._get_priority_vector(matrix)
        val = 0
        i = 0
        for feature in matrix.columns:
            val += np.sum(matrix[feature]) * prior_vector[i]
            i += 1

        return val
