"""
Multi-label Classification Models: Binary Relevance and Classifier Chains.
"""

from .binary_relevance import (
    BinaryRelevanceClassifier,
    BinaryRelevanceLogisticRegression,
    BinaryRelevanceMLP
)
from .classifier_chain import ClassifierChainClassifier
from .ensemble_classifier_chain import EnsembleClassifierChainClassifier
from .mlc_pa import MLCPartialAbstentionClassifier, MLCPAClassifier
from .gsi_mlc_pa import GSIMLCPartialAbstentionClassifier, GSIMLCPAClassifier
from .gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from .registry import (
    LOGISTIC_MLP_MODEL_IDS,
    MATCHED_MODEL_IDS,
    canonical_registered_model_id,
    create_registered_model,
    get_model_spec,
    model_family,
)
try:
    from .pytorch_mlp import (
        MultiLabelMLPClassifier,
        FastPyTorchBinaryMLP,
        get_default_device
    )
except ImportError:  # PyTorch is an optional acceleration dependency.
    MultiLabelMLPClassifier = None
    FastPyTorchBinaryMLP = None

    def get_default_device():
        return "cpu"

__all__ = [
    "BinaryRelevanceClassifier",
    "BinaryRelevanceLogisticRegression",
    "BinaryRelevanceMLP",
    "ClassifierChainClassifier",
    "EnsembleClassifierChainClassifier",
    "MLCPartialAbstentionClassifier",
    "MLCPAClassifier",
    "GSIMLCPartialAbstentionClassifier",
    "GSIMLCPAClassifier",
    "GSIMLCPAv6_3_3Classifier",
    "MATCHED_MODEL_IDS",
    "LOGISTIC_MLP_MODEL_IDS",
    "canonical_registered_model_id",
    "create_registered_model",
    "get_model_spec",
    "model_family",
    "MultiLabelMLPClassifier",
    "FastPyTorchBinaryMLP"
]
