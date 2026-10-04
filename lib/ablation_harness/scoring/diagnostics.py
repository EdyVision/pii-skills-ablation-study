"""Offline diagnostics for saved runs; does not change the original F1 scorer.

Localization stages use independent maximum-cardinality one-to-one matchings.
Their count ratios are not per-entity conditional retention probabilities.
"""
from collections import Counter, defaultdict
import math
import statistics


def _integer(value):
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return 0


def _bounds(entity):
    return _integer(entity.get("start", 0)), _integer(entity.get("end", 0))


def _valid_span(entity):
    start, end = _bounds(entity)
    return start >= 0 and end > start


def _iou(a, b):
    if not _valid_span(a) or not _valid_span(b):
        return 0.0
    a0, a1 = _bounds(a)
    b0, b1 = _bounds(b)
    intersection = max(0, min(a1, b1) - max(a0, b0))
    return intersection / ((a1 - a0) + (b1 - b0) - intersection)


def matched_count(predictions, gold, normalize, mode):
    """Maximum number of one-to-one matches under one criterion.

    A prediction and a gold instance can each receive credit at most once.
    Augmenting paths avoid dependence on prediction/gold ordering.
    """
    if mode not in {"type", "overlap", "span", "strict"}:
        raise ValueError(f"Unknown matching mode: {mode}")
    predictions = [p for p in predictions if isinstance(p, dict) and "type" in p]
    edges = []
    for p in predictions:
        matches = []
        for j, g in enumerate(gold):
            if normalize(p.get("type", "")) != normalize(g.get("type", "")):
                continue
            if mode == "type":
                matches.append(j)
            elif _valid_span(p) and _valid_span(g):
                overlap = _iou(p, g)
                if ((mode == "overlap" and overlap > 0)
                    or (mode == "span" and overlap >= 0.5)
                    or (mode == "strict" and _bounds(p) == _bounds(g))):
                    matches.append(j)
        edges.append(matches)
    owners = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return len(owners)


def validate_records(records):
    """Reject duplicate model-condition-example observations."""
    seen = set()
    for row in records:
        key = (row["model"], row["condition"], row["sample_id"])
        if key in seen:
            raise ValueError(f"Duplicate model-condition-example: {key}")
        seen.add(key)


def localization_diagnostics(records, condition, normalize):
    totals = Counter(gold=0, type=0, overlap=0, span=0, strict=0)
    for row in records:
        if row["condition"] != condition:
            continue
        gold = [g for g in (row.get("ground_truth") or [])
                if isinstance(g, dict) and _valid_span(g)]
        totals["gold"] += len(gold)
        for mode in ("type", "overlap", "span", "strict"):
            totals[mode] += matched_count(row.get("predictions") or [], gold, normalize, mode)
    n = totals["gold"]
    pct = lambda count: 100 * count / n if n else float("nan")
    return {"gold": n, **{f"{k}_matches": totals[k] for k in ("type", "overlap", "span", "strict")},
            "type %": pct(totals["type"]), "any overlap %": pct(totals["overlap"]),
            "IoU≥0.5 %": pct(totals["span"]), "exact %": pct(totals["strict"]),
            "overlap/type count ratio %": 100 * totals["overlap"] / totals["type"] if totals["type"] else float("nan"),
            "exact/overlap count ratio %": 100 * totals["strict"] / totals["overlap"] if totals["overlap"] else float("nan")}


def prediction_volumes(records, conditions, normalize):
    """Count both predictions and gold once per selected evaluation row."""
    predicted, support = Counter(), Counter()
    n_rows = 0
    for row in records:
        if row["condition"] not in conditions:
            continue
        n_rows += 1
        for entity in row.get("predictions") or []:
            if isinstance(entity, dict) and "type" in entity:
                predicted[normalize(entity["type"])] += 1
        for entity in row.get("ground_truth") or []:
            if isinstance(entity, dict) and "type" in entity:
                support[normalize(entity["type"])] += 1
    return [{"type": label, "predicted": predicted[label], "support": support[label],
             "evaluation_rows": n_rows,
             "ratio": predicted[label] / support[label] if support[label] else None}
            for label in sorted(predicted.keys() | support.keys(), key=lambda k: (-predicted[k], k))]


def paired_dz(differences):
    """Paired Cohen's d_z, using the sample SD of paired differences."""
    values = list(differences)
    if len(values) < 2:
        return float("nan")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    if sd == 0:
        return 0.0 if mean == 0 else math.copysign(float("inf"), mean)
    return mean / sd


def sample_cluster_ci(observations, confidence=0.95):
    """t interval across per-example mean differences.

    Input is (sample_id, condition, difference). Every example must have the
    same conditions. Average within an example before computing uncertainty,
    so repeated conditions are not treated as independent examples.
    """
    from scipy import stats
    grouped = defaultdict(dict)
    for sample_id, condition, difference in observations:
        if condition in grouped[sample_id]:
            raise ValueError("Duplicate sample-condition difference")
        if not math.isfinite(difference):
            raise ValueError("Non-finite paired difference")
        grouped[sample_id][condition] = difference
    if not grouped:
        raise ValueError("No paired observations")
    sets = {frozenset(v) for v in grouped.values()}
    if len(sets) != 1:
        raise ValueError("Unequal condition coverage across examples")
    values = [statistics.mean(v.values()) for v in grouped.values()]
    n, mean = len(values), statistics.mean(values)
    margin = (float(stats.t.ppf((1 + confidence) / 2, n - 1))
              * statistics.stdev(values) / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": mean, "low": mean - margin, "high": mean + margin,
            "margin": margin, "n_samples": n, "n_pairs": sum(map(len, grouped.values()))}
