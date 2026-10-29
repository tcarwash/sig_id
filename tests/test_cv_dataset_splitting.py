from collections import Counter

from sigid.cv_dataset import _splits


def test_splits_assign_oversized_lineage_to_train_without_breaking_group_quotas():
    groups = {"dominant", *(f"small-{index}" for index in range(9))}
    group_sizes = {group: 1 for group in groups}
    group_sizes["dominant"] = 2007

    assignment = _splits(groups, seed=4192526, group_sizes=group_sizes)

    assert assignment["dominant"] == "train"
    assert Counter(assignment.values()) == {"train": 4, "test": 3, "validation": 3}
    assert assignment == _splits(groups, seed=4192526, group_sizes=group_sizes)
