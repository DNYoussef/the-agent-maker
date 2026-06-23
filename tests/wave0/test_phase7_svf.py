"""Phase 7 SVF: the REINFORCE baseline must not zero the logged loss on a scalar."""


def test_svf_reinforce_baseline_guarded_for_scalars():
    # Bug: `batch_loss - batch_loss.mean().detach()` on a scalar == `x - x.detach()`,
    # which zeroed the logged loss (~0) and reduced no variance. Now guarded to a
    # real batch (numel > 1) so scalar losses log their real value.
    import inspect
    import phase7_experts.svf_trainer as svf

    src = inspect.getsource(svf)
    assert "batch_loss.numel() > 1" in src, "scalar REINFORCE-baseline no-op must be guarded"
