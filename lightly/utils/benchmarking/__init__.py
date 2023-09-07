from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    from lightly.utils.benchmarking.benchmark_module import BenchmarkModule
    from lightly.utils.benchmarking.knn import knn_predict
    from lightly.utils.benchmarking.knn_classifier import KNNClassifier
    from lightly.utils.benchmarking.linear_classifier import LinearClassifier
    from lightly.utils.benchmarking.metric_callback import MetricCallback
    from lightly.utils.benchmarking.online_linear_classifier import (
        OnlineLinearClassifier,
    )
