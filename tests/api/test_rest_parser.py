import unittest

import numpy as np

from lightly.openapi_generated.swagger_client import ApiClient, ScoresApi, ActiveLearningScoreCreateRequest, \
    SamplingMethod
from lightly.openapi_generated.swagger_client.rest import ApiException


class TestRestParser(unittest.TestCase):

    @unittest.skip("This test only shows the error, it does not ensure it is solved.")
    def test_parse_active_learning_scores(self):
        score_value_tuple = (
            np.random.normal(0, 1, size=(999,)).astype(np.float32),
            np.random.normal(0, 1, size=(999,)).astype(np.float64),
            [12.0] * 999
        )
        api_client = ApiClient()
        self.scores_api = ScoresApi(api_client)
        for i, score_values in enumerate(score_value_tuple):
            with self.subTest(i=i, msg=str(type(score_values))):
                body = ActiveLearningScoreCreateRequest(score_type=SamplingMethod.CORESET, scores=list(score_values))
                if isinstance(score_values[0], float):
                    with self.assertRaises(ApiException):
                        self.scores_api.create_or_update_active_learning_score_by_tag_id(
                            body, dataset_id="dataset_id_xyz", tag_id="tag_id_xyz")
                else:
                    with self.assertRaises(AttributeError):
                        self.scores_api.create_or_update_active_learning_score_by_tag_id(
                            body, dataset_id="dataset_id_xyz", tag_id="tag_id_xyz")

