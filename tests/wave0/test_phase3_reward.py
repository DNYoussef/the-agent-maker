"""Phase 3 reward: accuracy must use next-token alignment (logits[:-1] vs
labels[1:]) to match the model's shifted cross-entropy. Comparing unshifted
argmax to labels rewarded the wrong offset."""

import torch


def test_compute_reward_uses_next_token_alignment():
    # Imported via the `src.` package path because step2_rl uses `..cross_phase`
    # relative imports that only resolve under that package root.
    from src.phase3_quietstar.step2_rl import REINFORCETrainer

    labels = torch.tensor([[5, 7, 9]])
    # With-thoughts predicts the NEXT token correctly: pos0->7 (=labels[1]),
    # pos1->9 (=labels[2]). Under correct shift this is 100% accurate; unshifted
    # it scores 0 (pos0 pred 7 vs labels[0]=5, etc.).
    logits_with = torch.zeros(1, 3, 10)
    logits_with[0, 0, 7] = 9.0
    logits_with[0, 1, 9] = 9.0
    logits_with[0, 2, 0] = 9.0
    # Without-thoughts predicts token 0 everywhere -> 0% accurate either way.
    logits_without = torch.zeros(1, 3, 10)
    logits_without[0, :, 0] = 9.0

    # compute_reward does not use self.
    reward = REINFORCETrainer.compute_reward(None, logits_with, logits_without, labels)
    assert reward.tolist() == [1.0], (
        "with-thoughts predicts the next tokens correctly -> reward 1.0 under "
        "next-token alignment (0.0 with the old unshifted comparison)"
    )


def test_compute_gae_does_not_leak_across_batch_samples():
    # Bug: the GAE recursion ran over the batch dimension, so one sample's
    # reward changed another sample's advantage. Each batch entry is an
    # independent single-step episode; advantages must be per-sample.
    from types import SimpleNamespace

    from src.phase3_quietstar.step2_rl import REINFORCETrainer

    stub = SimpleNamespace(config=SimpleNamespace(rl=SimpleNamespace(gamma=0.99, gae_lambda=0.95)))
    values = torch.zeros(2)
    next_values = torch.zeros(2)
    dones = torch.zeros(2)

    adv_a = REINFORCETrainer.compute_gae(stub, torch.tensor([1.0, 0.0]), values, next_values, dones)
    adv_b = REINFORCETrainer.compute_gae(stub, torch.tensor([1.0, 5.0]), values, next_values, dones)
    # Only sample 1's reward changed; sample 0's advantage must be unchanged.
    assert adv_a[0].item() == adv_b[0].item(), "advantage[0] must not depend on sample 1's reward"
