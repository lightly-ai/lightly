import unittest
import torch

import lightly


class TestNestedImports(unittest.TestCase):

    def test_nested_imports(self):
        # active learning
        lightly.active_learning.agents.agent.ActiveLearningAgent
        lightly.active_learning.agents.ActiveLearningAgent
        lightly.active_learning.config.selection_config.SelectionConfig
        lightly.active_learning.config.SelectionConfig
        lightly.active_learning.scorers.classification.ScorerClassification
        lightly.active_learning.scorers.ScorerClassification
        lightly.active_learning.scorers.detection.ScorerObjectDetection
        lightly.active_learning.scorers.ScorerObjectDetection
        lightly.active_learning.utils.bounding_box.BoundingBox
        lightly.active_learning.utils.BoundingBox
        lightly.active_learning.utils.object_detection_output.ObjectDetectionOutput
        lightly.active_learning.utils.ObjectDetectionOutput

        # api imports
        lightly.api.api_workflow_client.ApiWorkflowClient
        lightly.api.ApiWorkflowClient
        lightly.api.bitmask.BitMask

        # data imports
        lightly.data.LightlyDataset
        lightly.data.dataset.LightlyDataset
        lightly.data.BaseCollateFunction
        lightly.data.collate.BaseCollateFunction
        lightly.data.ImageCollateFunction
        lightly.data.collate.ImageCollateFunction
        lightly.data.MoCoCollateFunction
        lightly.data.collate.MoCoCollateFunction
        lightly.data.SimCLRCollateFunction
        lightly.data.collate.SimCLRCollateFunction
        lightly.data.imagenet_normalize
        lightly.data.collate.imagenet_normalize

        # embedding imports
        lightly.embedding.BaseEmbedding
        lightly.embedding._base.BaseEmbedding
        lightly.embedding.SelfSupervisedEmbedding
        lightly.embedding.embedding.SelfSupervisedEmbedding

        # loss imports
        lightly.loss.NTXentLoss
        lightly.loss.ntx_ent_loss.NTXentLoss
        lightly.loss.SymNegCosineSimilarityLoss
        lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss
        lightly.loss.memory_bank.MemoryBankModule
        lightly.loss.regularizer.CO2Regularizer
        lightly.loss.regularizer.co2.CO2Regularizer

        # models imports
        lightly.models.ResNetGenerator
        lightly.models.resnet.ResNetGenerator
        lightly.models.SimCLR
        lightly.models.simclr.SimCLR
        lightly.models.MoCo
        lightly.models.moco.MoCo
        lightly.models.SimSiam
        lightly.models.simsiam.SimSiam
        lightly.models.ZOO
        lightly.models.zoo.ZOO
        lightly.models.checkpoints
        lightly.models.zoo.checkpoints
        lightly.models.batchnorm.get_norm_layer

        # transforms imports
        lightly.transforms.GaussianBlur
        lightly.transforms.gaussian_blur.GaussianBlur
        lightly.transforms.RandomRotate
        lightly.transforms.rotation.RandomRotate

        # core imports
        lightly.train_model_and_embed_images
        lightly.core.train_model_and_embed_images
        lightly.train_embedding_model
        lightly.core.train_embedding_model
        lightly.embed_images
        lightly.core.embed_images