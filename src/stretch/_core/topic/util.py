"""Small TOPIC helpers required by the STRETCH search wrapper.

The original evaluation utility also imported evaluation metrics from ``cdt``.
Those imports are deliberately excluded because neither helper below needs them.
"""


def is_insignificant(gain: float, alpha: float = 0.05) -> bool:
    """Return whether an information gain fails the legacy significance rule."""

    return gain < 0 or 2 ** (-gain) > alpha


def compare_adj(true_adj, estimated_adj):
    """Compute the legacy directional edge counts and F1 score."""

    true_positive = 0
    false_positive = 0
    false_negative = 0
    for source in estimated_adj:
        for target in estimated_adj[source]:
            if target in true_adj[source]:
                true_positive += 1
            else:
                false_positive += 1
    for source in true_adj:
        for target in true_adj[source]:
            if target not in estimated_adj[source]:
                false_negative += 1
    denominator = true_positive + 0.5 * (false_positive + false_negative)
    f1 = true_positive / denominator if denominator > 0 else 1
    return {
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }
