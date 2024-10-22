import typing as tp
from abc import ABC


class EvalMetric(ABC):
    __eval_name: str = ""

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    @property
    def eval_name(self) -> str:
        return self.__eval_name


class KNNEvalMetric(EvalMetric):
    __eval_name = "knn_eval"


class LinearEvalMetric(EvalMetric):
    __eval_name = "linear_eval"


class FinetuneEvalMetric(EvalMetric):
    __eval_name = "finetune_eval"


EvalMetrics = tp.List[EvalMetric]


def eval_metrics_to_markdown(metrics: EvalMetrics) -> str:
    lines = []
    header = f"| Eval Name | Metric Name | Value |"
    lines.append(header)
    for metric in metrics:
        line = f"| {metric.eval_name} | {metric.name} | {metric.value} |"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "EvalMetric",
    "KNNEvalMetric",
    "LinearEvalMetric",
    "FinetuneEvalMetric",
    "EvalMetrics",
    "eval_metrics_to_markdown",
]
