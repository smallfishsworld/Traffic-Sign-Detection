"""Evaluation: COCO mAP with Open Images verified-label filtering and hierarchy expansion
(group-of boxes scored as COCO crowd regions), with bootstrap intervals."""

from tsd.eval.metrics import (
    ImagePredictions,
    ImageTruth,
    MatchCache,
    Summary,
    bootstrap,
    build_cache,
    summarize,
)
from tsd.eval.protocol import ClassMap, EvalImage, EvalSet, map_classes

__all__ = [
    "ClassMap",
    "EvalImage",
    "EvalSet",
    "ImagePredictions",
    "ImageTruth",
    "MatchCache",
    "Summary",
    "bootstrap",
    "build_cache",
    "map_classes",
    "summarize",
]
