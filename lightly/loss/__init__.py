"""The lightly.loss package provides loss functions for self-supervised learning. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.loss.barlow_twins_loss import BarlowTwinsLoss
from lightly.loss.dcl_loss import DCLLoss, DCLWLoss
from lightly.loss.dino_loss import DINOLoss
from lightly.loss.msn_loss import MSNLoss
from lightly.loss.negative_cosine_similarity import NegativeCosineSimilarity
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.loss.swav_loss import SwaVLoss
from lightly.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss
from lightly.loss.vicreg_loss import VICRegLoss
