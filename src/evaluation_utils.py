from typing import List, Dict

from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_multiclass(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Compute basic metrics for multi-class classification.
    Returns overall accuracy and macro-averaged F1.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


def print_classification_summary(y_true: List[int], y_pred: List[int], target_names=None) -> None:
    """Pretty-print sklearn classification report."""
    print(classification_report(y_true, y_pred, digits=4, target_names=target_names))
