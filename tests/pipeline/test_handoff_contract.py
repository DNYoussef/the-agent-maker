"""E0 gate - CRUCIBLE: the pipeline handoff contract.

The phase-audit synthesis found the orchestrator threads the model in-memory but
does NOT carry a tokenizer, so phases 5-8 each fabricate their own gpt2 tokenizer
(tokenizer drift). E0 establishes the contract: a PhaseResult carries BOTH model and
tokenizer, and the orchestrator injects the prior phase's tokenizer into the next
controller (controller.input_tokenizer). This gate locks that contract; E2 then makes
the downstream phases consume input_tokenizer instead of hardcoding gpt2.
"""

from cross_phase.orchestrator.base_controller import PhaseController, PhaseResult


def _result(name, model, tokenizer=None):
    return PhaseResult(
        success=True,
        phase_name=name,
        model=model,
        metrics={},
        duration=0.0,
        artifacts={},
        config={},
        tokenizer=tokenizer,
    )


def test_phaseresult_carries_tokenizer():
    tok = object()
    r = _result("phase1", model=object(), tokenizer=tok)
    assert r.tokenizer is tok
    # default is None (non-breaking for existing constructions)
    r2 = PhaseResult(
        success=True, phase_name="p", model=None, metrics={}, duration=0.0, artifacts={}, config={}
    )
    assert r2.tokenizer is None


def test_controller_exposes_input_tokenizer_slot():
    class _C(PhaseController):
        def execute(self, input_models=None):
            return _result("c", object())

        def validate_input(self, input_models=None):
            return True

        def validate_output(self, result):
            return True

    c = _C(config={}, session_id="t")
    assert c.input_tokenizer is None


class _Rec(PhaseController):
    """Records what model list + tokenizer the orchestrator injected before execute."""

    def __init__(self, name, out_model, out_tokenizer=None):
        super().__init__(config={}, session_id="t")
        self._name = name
        self._out = _result(name, out_model, out_tokenizer)
        self.seen_models = "UNSET"
        self.seen_tokenizer = "UNSET"

    def execute(self, input_models=None):
        self.seen_models = input_models
        self.seen_tokenizer = self.input_tokenizer
        return self._out

    def validate_input(self, input_models=None):
        return True

    def validate_output(self, result):
        return True


def test_run_full_pipeline_threads_tokenizer_to_all_downstream_phases(tmp_path, monkeypatch):
    # Drive the REAL run_full_pipeline loop (not a hand-mirrored carry): phase 1 emits a
    # tokenizer; assert every downstream phase (2-8) received it via input_tokenizer, and
    # that a phase emitting tokenizer=None does NOT wipe the carried one (last-non-None).
    from cross_phase.orchestrator.pipeline import PipelineOrchestrator

    cfg = {"registry": {"db_path": str(tmp_path / "reg.db")}, "wandb": {"enabled": False}}
    orch = PipelineOrchestrator(cfg, session_id="t")

    t1 = object()
    controllers = {1: _Rec("phase1", out_model=object(), out_tokenizer=t1)}
    for n in range(2, 9):
        controllers[n] = _Rec(f"phase{n}", out_model=object())  # emit tokenizer=None
    monkeypatch.setattr(orch, "_get_phase_controller", lambda n: controllers[n])
    monkeypatch.setattr(orch, "_register_phase_model", lambda *a, **k: None)
    monkeypatch.setattr(orch.registry, "update_session_progress", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_log_phase_to_wandb", lambda *a, **k: None)

    orch.run_full_pipeline()

    assert controllers[1].seen_tokenizer is None, "phase 1 should start with no tokenizer"
    for n in range(2, 9):
        assert controllers[n].seen_tokenizer is t1, f"phase {n} lost the threaded tokenizer"
    assert controllers[2].seen_models == [controllers[1]._out.model], "model not threaded 1->2"


def test_run_single_phase_accepts_input_tokenizer(tmp_path, monkeypatch):
    from cross_phase.orchestrator.pipeline import PipelineOrchestrator

    cfg = {"registry": {"db_path": str(tmp_path / "reg.db")}, "wandb": {"enabled": False}}
    orch = PipelineOrchestrator(cfg, session_id="t")
    tok = object()
    c = _Rec("phase4", out_model=object())
    monkeypatch.setattr(orch, "_get_phase_controller", lambda n: c)
    monkeypatch.setattr(orch, "_register_phase_model", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_log_phase_to_wandb", lambda *a, **k: None)
    orch.run_single_phase(4, input_models=[object()], input_tokenizer=tok)
    assert c.seen_tokenizer is tok, "run_single_phase did not thread input_tokenizer"
