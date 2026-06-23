"""Phase 2 real-fitness path: correct answer parsing + reachable from the standard
evolutionary loop (not only the hybrid path)."""


def test_extract_numeric_answer_handles_thousands_separators():
    # Bug: "#### 1,000" parsed as 1 (regex stopped at the comma), undercounting
    # correct answers and corrupting the real-fitness signal.
    from phase2_evomerge.fitness.benchmarks import extract_numeric_answer

    assert extract_numeric_answer("#### 1,000") == 1000.0
    assert extract_numeric_answer("The answer is 12,345") == 12345.0
    assert extract_numeric_answer("#### 42") == 42.0
    assert extract_numeric_answer("3.5") == 3.5
    assert extract_numeric_answer("result: 1,234,567") == 1234567.0


def test_evaluate_population_honors_use_real_fitness():
    # Bug: _evaluate_population always used the proxy _quick_fitness, ignoring
    # use_real_fitness (only the hybrid path honored it).
    import inspect
    from phase2_evomerge.phase2_pipeline import Phase2Pipeline

    src = inspect.getsource(Phase2Pipeline._evaluate_population)
    assert "use_real_fitness" in src and "_create_real_fitness_fn" in src, (
        "standard evaluation path must honor use_real_fitness"
    )
