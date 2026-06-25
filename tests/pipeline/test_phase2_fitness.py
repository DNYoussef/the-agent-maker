"""E3 gate - CRUCIBLE: Phase 2 fitness is honest and the evolution is well-formed.

Synthesis HIGH bugs: proxy fitness includes random speed (champion selection partly
random); population shrinks each generation while metrics report the nominal size;
use_real_fitness silently falls back to proxy when no tokenizer; champion/metrics can
disagree. These gate the deterministic, constant-population, fail-loud behavior.
"""

import pytest
import torch.nn as nn

from phase2_evomerge.phase2_pipeline import EvolutionConfig, Phase2Pipeline


def _models(n=3):
    return [nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)) for _ in range(n)]


def _cfg(**kw):
    base = dict(population_size=6, num_generations=2, use_real_fitness=False)
    base.update(kw)
    return EvolutionConfig(**base)


def test_proxy_fitness_is_deterministic():
    p = Phase2Pipeline(_cfg())
    m = _models(1)[0]
    a = p._quick_fitness(m)["composite"]
    b = p._quick_fitness(m)["composite"]
    assert a == b, "proxy fitness must be deterministic (no random term)"


def test_population_size_stays_constant():
    p = Phase2Pipeline(_cfg())
    p.run(_models(3))
    assert (
        len(p.population) == p.config.population_size
    ), f"population shrank to {len(p.population)} (expected {p.config.population_size})"
    assert p.metrics["population_size"] == len(p.population), "metrics report a phantom size"


def test_real_fitness_without_tokenizer_fails_loud():
    p = Phase2Pipeline(_cfg(use_real_fitness=True))
    with pytest.raises(ValueError):
        p.run(_models(3), tokenizer=None)  # must NOT silently fall back to proxy


def test_champion_matches_reported_fitness():
    p = Phase2Pipeline(_cfg())
    champion = p.run(_models(3))
    assert champion is not None
    # the reported final/champion fitness must equal the champion the run returned
    assert p.metrics["final_fitness"] == pytest.approx(p.best_fitness)


def test_rerun_resets_per_run_state():
    # A reused pipeline must not report/return a prior run's champion (Codex E3 #1).
    p = Phase2Pipeline(_cfg())
    p.run(_models(3))
    p.run(_models(3))
    assert len(p.fitness_history) == p.config.num_generations, "state leaked across runs"


def test_use_cmaes_fails_loud():
    # use_cmaes was theater (printed CMA-ES, ran standard evolution) - must fail loud now.
    with pytest.raises(NotImplementedError):
        Phase2Pipeline(_cfg(use_cmaes=True)).run(_models(3))
