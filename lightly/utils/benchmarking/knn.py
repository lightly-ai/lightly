import torch
from torch import Tensor

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb


def knn_predict(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> Tensor:
    """Run kNN predictions on features based on a feature bank.

    This method is commonly used to monitor the performance of self-supervised
    learning methods. The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions, where B is the
            batch size and D is the feature dimension.
        feature_bank:
            Tensor of shape (D, N) representing a database of features used for kNN,
            where N is the number of stored feature vectors.
        feature_labels:
            Tensor with shape (N,) containing labels for the corresponding
            feature vectors in the feature_bank.
        num_classes:
            Number of classes (e.g., `10` for CIFAR-10).
        knn_k:
            Number of k nearest neighbors used for kNN.
        knn_t:
            Temperature parameter to reweight similarities for kNN.

    Returns:
        Tensor of shape (B, num_classes) with the predicted class indices sorted
        by probability in descending order for each sample. The first index 
        corresponds to the most probable class. To get the top-1 prediction, 
        you can access `pred_labels[:, 0]`.

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend normalizing the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
        >>> # top-1 prediction
        >>> top1_pred = pred_labels[:, 0]
    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
