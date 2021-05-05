import unittest
import torch

import lightly


class TestFromImports(unittest.TestCase):

    def test_from_imports(self):
        # active learning (commented out don't work)
        from lightly.active_learning.config.sampler_config import SamplerConfig
        from lightly.active_learning.agents.agent import ActiveLearningAgent
        from lightly.active_learning.scorers.classification import ScorerClassification

        # api imports
        from lightly.api.api_workflow_client import ApiWorkflowClient
        from lightly.api.bitmask import BitMask

        # data imports
        from lightly.data import LightlyDataset
        from lightly.data.dataset  import LightlyDataset
        from lightly.data  import BaseCollateFunction
        from lightly.data.collate  import BaseCollateFunction
        from lightly.data import ImageCollateFunction
        from lightly.data.collate import ImageCollateFunction
        from lightly.data import MoCoCollateFunction
        from lightly.data.collate import MoCoCollateFunction
        from lightly.data import SimCLRCollateFunction
        from lightly.data.collate import SimCLRCollateFunction
        from lightly.data import imagenet_normalize
        from lightly.data.collate import imagenet_normalize

        # embedding imports
        from lightly.embedding import BaseEmbedding
        from lightly.embedding._base import BaseEmbedding
        from lightly.embedding import SelfSupervisedEmbedding
        from lightly.embedding.embedding import SelfSupervisedEmbedding

        # loss imports
        from lightly.loss import NTXentLoss
        from lightly.loss.ntx_ent_loss import NTXentLoss
        from lightly.loss import SymNegCosineSimilarityLoss
        from lightly.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss
        from lightly.loss.memory_bank import MemoryBankModule
        from lightly.loss.regularizer import CO2Regularizer
        from lightly.loss.regularizer.co2 import CO2Regularizer

        # models imports
        from lightly.models import ResNetGenerator
        from lightly.models.resnet import ResNetGenerator
        from lightly.models import SimCLR
        from lightly.models.simclr import SimCLR
        from lightly.models import MoCo
        from lightly.models.moco import MoCo
        from lightly.models import SimSiam
        from lightly.models.simsiam import SimSiam
        from lightly.models import ZOO
        from lightly.models.zoo import ZOO
        from lightly.models import checkpoints
        from lightly.models.zoo import checkpoints
        from lightly.models.batchnorm import get_norm_layer

        # transforms imports
        from lightly.transforms import GaussianBlur
        from lightly.transforms.gaussian_blur import GaussianBlur
        from lightly.transforms import RandomRotate
        from lightly.transforms.rotation import RandomRotate

        # utils imports
        from lightly.utils import save_embeddings
        from lightly.utils.io import save_embeddings
        from lightly.utils import load_embeddings
        from lightly.utils.io import load_embeddings
        from lightly.utils import load_embeddings_as_dict
        from lightly.utils.io import load_embeddings_as_dict
        from lightly.utils import fit_pca
        from lightly.utils.embeddings_2d import fit_pca

        # core imports
        from lightly import train_model_and_embed_images
        from lightly import train_embedding_model
        from lightly import embed_images