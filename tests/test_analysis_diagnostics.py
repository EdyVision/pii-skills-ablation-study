import math
import pytest
from ablation_harness.scoring.diagnostics import (
    matched_count, localization_diagnostics, prediction_volumes,
    paired_dz, sample_cluster_ci, validate_records,
)

def entity(start, end, kind="PERSON"):
    return {"type": kind, "start": start, "end": end}

def row(predictions, gold, sample_id="a", condition="with_tools", model="m"):
    return dict(predictions=predictions, ground_truth=gold, sample_id=sample_id,
                condition=condition, model=model)

def test_repeated_type_perfect_predictions_get_full_recall():
    gold = [entity(0, 3), entity(10, 13)]
    result = localization_diagnostics([row(gold, gold)], "with_tools", str)
    assert all(result[key] == 100 for key in ["type %", "any overlap %", "IoU≥0.5 %", "exact %"])

def test_later_same_type_entity_can_match():
    gold = [entity(0, 3), entity(10, 13)]
    assert matched_count([gold[1]], gold, str, "strict") == 1

def test_one_prediction_cannot_cover_two_gold_entities():
    assert matched_count([entity(0, 15)], [entity(0, 3), entity(10, 13)], str, "overlap") == 1

def test_duplicate_predictions_cannot_reuse_gold():
    gold = [entity(0, 3)]
    assert matched_count(gold * 2, gold, str, "strict") == 1

def test_augmenting_path_and_order_invariance():
    gold = [entity(0, 5), entity(5, 10)]
    predictions = [entity(0, 10), entity(0, 5)]
    for pp in [predictions, predictions[::-1]]:
        for gg in [gold, gold[::-1]]:
            assert matched_count(pp, gg, str, "span") == 2

def test_invalid_spans_receive_no_localization_credit():
    for bad in [entity(-1, 5), entity(5, 5), entity(6, 2)]:
        assert matched_count([bad], [entity(0, 5)], str, "overlap") == 0

def test_type_only_gold_is_excluded_from_localization_denominator():
    r = row([entity(0, 3)], [entity(0, 3), {"type": "PERSON"}])
    assert localization_diagnostics([r], "with_tools", str)["gold"] == 1

def test_volume_denominators_repeat_gold_for_every_selected_evaluation():
    e = [entity(0, 3)]
    records = [row(e, e, model=m, condition=c) for m in ["m1", "m2"] for c in ["with_tools", "with_skills"]]
    records.append(row(e * 20, e, condition="zero_shot"))
    result = prediction_volumes(records, ["with_tools", "with_skills"], str)[0]
    assert result["predicted"] == result["support"] == 4
    assert result["ratio"] == 1

def test_cluster_interval_not_narrowed_by_duplicate_conditions():
    once = [(i, "a", float(i)) for i in range(8)]
    repeated = [(i, c, float(i)) for i in range(8) for c in ["a", "b", "c", "d"]]
    assert sample_cluster_ci(once)["margin"] == pytest.approx(sample_cluster_ci(repeated)["margin"])
    assert sample_cluster_ci(repeated)["n_samples"] == 8

def test_cluster_requires_balanced_conditions():
    with pytest.raises(ValueError, match="Unequal"):
        sample_cluster_ci([("a", "x", 1), ("a", "y", 1), ("b", "x", 1)])

def test_paired_effect_size_uses_sample_sd():
    assert paired_dz([1, 2, 3]) == 2
    assert paired_dz([0, 0]) == 0
    assert math.isinf(paired_dz([1, 1]))

def test_duplicate_result_keys_fail_explicitly():
    r = row([], [])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_records([r, r])
