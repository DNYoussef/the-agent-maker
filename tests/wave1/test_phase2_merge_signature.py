"""
Wave 1 - Class A: Phase2 EvoMerge merger signature unification.

Every merger must accept the single canonical signature
merge(models: List[nn.Module]) -> nn.Module. Before the fix, ties/dare/
frankenmerge/dfs_merge required (target, refs) positional args and raised
TypeError when the pipeline called merge([a, b]).
"""

import torch
import torch.nn as nn


def _tiny_linear(seed):
    torch.manual_seed(seed)
    return nn.Linear(8, 8)


def test_all_merge_classes_accept_a_two_model_list():
    from src.phase2_evomerge.merge.dare_merge import DAREMerge
    from src.phase2_evomerge.merge.dfs_merge import DFSMerge
    from src.phase2_evomerge.merge.frankenmerge import FrankenMerge
    from src.phase2_evomerge.merge.linear_merge import LinearMerge
    from src.phase2_evomerge.merge.slerp_merge import SLERPMerge
    from src.phase2_evomerge.merge.ties_merge import TIESMerge

    mergers = [
        SLERPMerge(),
        LinearMerge(),
        TIESMerge(),
        DAREMerge(),
        FrankenMerge(),
        DFSMerge(),
    ]
    for merger in mergers:
        merged = merger.merge([_tiny_linear(1), _tiny_linear(2)])
        assert isinstance(merged, nn.Module), f"{type(merger).__name__} did not return a model"


def test_merge_techniques_apply_combo_still_works():
    """MergeTechniques.apply_combo must keep working after signature unification."""
    from src.phase2_evomerge.merge import MergeTechniques

    techniques = MergeTechniques()
    models = [_tiny_linear(1), _tiny_linear(2), _tiny_linear(3)]
    for combo in range(8):
        result = techniques.apply_combo(models, combo_id=combo)
        assert isinstance(result, nn.Module)


def test_too_few_models_raises():
    from src.phase2_evomerge.merge.ties_merge import TIESMerge

    try:
        TIESMerge().merge([_tiny_linear(1)])
    except ValueError:
        return
    raise AssertionError("expected ValueError for single-model list")
