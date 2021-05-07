import unittest
import torch

import lightly


class TestSemiNestedImports(unittest.TestCase):

    def test_seminested_imports(self):
        from lightly import active_learning
        # active learning (commented out don't work)
        active_learning.agents.ActiveLearningAgent
        active_learning.config.SamplerConfig
        active_learning.scorers.ScorerClassification
        active_learning.scorers.ScorerObjectDetection

        # api imports
        from lightly import api
        api.api_workflow_client.ApiWorkflowClient
        api.bitmask.BitMask

        # data imports
        from lightly import data
        data.LightlyDataset
        data.dataset.LightlyDataset
        data.BaseCollateFunction
        data.collate.BaseCollateFunction
        data.ImageCollateFunction
        data.collate.ImageCollateFunction
        data.MoCoCollateFunction
        data.collate.MoCoCollateFunction
        data.SimCLRCollateFunction
        data.collate.SimCLRCollateFunction
        data.imagenet_normalize
        data.collate.imagenet_normalize

        # embedding imports
        from lightly import embedding
        embedding.BaseEmbedding
        embedding._base.BaseEmbedding
        embedding.SelfSupervisedEmbedding
        embedding.embedding.SelfSupervisedEmbedding

        # loss imports
        from lightly import loss
        loss.NTXentLoss
        loss.ntx_ent_loss.NTXentLoss
        loss.SymNegCosineSimilarityLoss
        loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss
        loss.memory_bank.MemoryBankModule

        from lightly.loss import regularizer
        regularizer.CO2Regularizer
        regularizer.co2.CO2Regularizer

        # models imports
        from lightly import models
        models.ResNetGenerator
        models.resnet.ResNetGenerator
        models.SimCLR
        models.simclr.SimCLR
        models.MoCo
        models.moco.MoCo
        models.SimSiam
        models.simsiam.SimSiam
        models.ZOO
        models.zoo.ZOO
        models.checkpoints
        models.zoo.checkpoints
        models.batchnorm.get_norm_layer

        # transforms imports
        from lightly import transforms
        transforms.GaussianBlur
        transforms.gaussian_blur.GaussianBlur
        transforms.RandomRotate
        transforms.rotation.RandomRotate

        # utils imports
        from lightly import utils
        utils.save_embeddings
        utils.io.save_embeddings
        utils.load_embeddings
        utils.io.load_embeddings
        utils.load_embeddings_as_dict
        utils.io.load_embeddings_as_dict
        utils.fit_pca
        utils.embeddings_2d.fit_pca
