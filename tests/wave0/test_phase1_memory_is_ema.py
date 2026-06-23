"""Phase 1 memory honesty: the 'Titans' long-term memory is actually a fixed EMA
recurrence, not the surprise-gradient neural memory of Behrouz et al. (2024).
This characterizes the real mechanism so the label can't drift back to a claim
the code doesn't implement."""

import torch

from phase1_cognate.model.components.memory import LongTermMemory


def test_memory_is_a_fixed_ema_recurrence_not_surprise_gradient():
    torch.manual_seed(0)
    d_model, d_mem, decay = 6, 4, 0.9
    mem = LongTermMemory(d_model, d_mem, decay=decay)
    mem.reset_memory()
    mem.eval()

    x = torch.randn(1, 3, d_model)
    with torch.no_grad():
        out = mem(x)
        # Reconstruct the claimed EMA recurrence independently:
        compressed = mem.w_down(x)
        m = torch.zeros(1, 1, d_mem)
        expected_steps = []
        for t in range(x.shape[1]):
            m = decay * m + (1 - decay) * compressed[:, t : t + 1, :]
            expected_steps.append(m)
        expected = mem.w_up(torch.cat(expected_steps, dim=1))

    # Output matches the pure EMA -> there is no test-time gradient/surprise update.
    assert torch.allclose(out, expected, atol=1e-5), "memory must be the plain EMA recurrence"

    # And the module exposes no surprise/gradient-memory machinery.
    assert not hasattr(mem, "surprise") and not hasattr(mem, "momentum")
