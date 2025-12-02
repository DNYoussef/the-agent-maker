"""
Unit tests for Prompt Baking System
Tests KL divergence baking, half-baking, and prompt management
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.prompt_baking.baker import PromptBakingConfig
from cross_phase.prompt_baking.prompts import PhasePrompts, PromptManager


class TestPhasePrompts:
    """Test prompt templates"""

    def test_phase3_cot_prompt(self):
        """Test Phase 3 CoT reasoning prompt exists"""
        prompt = PhasePrompts.PHASE3_COT_REASONING

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "reasoning" in prompt.lower()

    def test_phase5_eudaimonia_prompt(self):
        """Test Phase 5 eudaimonia prompt exists"""
        prompt = PhasePrompts.PHASE5_EUDAIMONIA

        assert isinstance(prompt, str)
        assert "virtue" in prompt.lower() or "rule" in prompt.lower()

    def test_phase5_tool_use_prompt(self):
        """Test Phase 5 tool use prompt exists"""
        prompt = PhasePrompts.PHASE5_TOOL_USE

        assert isinstance(prompt, str)
        assert "tool" in prompt.lower()

    def test_phase6_personas_exist(self):
        """Test Phase 6 personas are defined"""
        personas = PhasePrompts.PHASE6_PERSONAS

        assert isinstance(personas, dict)
        assert len(personas) == 9

        expected_personas = [
            "reasoning_specialist",
            "creative_thinker",
            "technical_expert",
            "educator",
            "code_specialist",
            "data_analyst",
            "researcher",
            "debugger",
            "system_designer",
        ]

        for persona in expected_personas:
            assert persona in personas
            assert isinstance(personas[persona], str)
            assert len(personas[persona]) > 0

    def test_phase6_swe_bench_prompt(self):
        """Test Phase 6 SWE-Bench prompt exists"""
        prompt = PhasePrompts.PHASE6_SWE_BENCH

        assert isinstance(prompt, str)
        assert "engineer" in prompt.lower() or "github" in prompt.lower()


class TestPromptManager:
    """Test PromptManager"""

    def test_get_phase3_prompts(self):
        """Test getting Phase 3 prompts"""
        prompts = PromptManager.get_phase3_prompts()

        assert isinstance(prompts, list)
        assert len(prompts) == 1
        assert prompts[0] == PhasePrompts.PHASE3_COT_REASONING

    def test_get_phase5_prompts(self):
        """Test getting Phase 5 prompts"""
        prompts = PromptManager.get_phase5_prompts()

        assert isinstance(prompts, list)
        assert len(prompts) == 2
        assert PhasePrompts.PHASE5_EUDAIMONIA in prompts
        assert PhasePrompts.PHASE5_TOOL_USE in prompts

    def test_get_phase6_prompts(self):
        """Test getting Phase 6 prompts"""
        prompts = PromptManager.get_phase6_prompts()

        assert isinstance(prompts, list)
        # 9 personas + 1 SWE-Bench = 10 prompts
        assert len(prompts) == 10
        assert PhasePrompts.PHASE6_SWE_BENCH in prompts

    def test_get_all_prompts(self):
        """Test getting all prompts"""
        all_prompts = PromptManager.get_all_prompts()

        assert isinstance(all_prompts, dict)
        assert "phase3" in all_prompts
        assert "phase5" in all_prompts
        assert "phase6" in all_prompts

    def test_phase6_prompts_include_personas(self):
        """Test Phase 6 prompts include all personas"""
        prompts = PromptManager.get_phase6_prompts()

        for persona_prompt in PhasePrompts.PHASE6_PERSONAS.values():
            assert persona_prompt in prompts


class TestPromptBakingConfig:
    """Test PromptBakingConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PromptBakingConfig()

        assert config.lora_r == 16
        assert config.num_epochs == 3
        assert config.learning_rate == 1e-4

    def test_half_baking_factor(self):
        """Test half-baking factor"""
        config = PromptBakingConfig()

        assert config.half_baking_factor == 0.5
        assert 0.0 < config.half_baking_factor <= 1.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = PromptBakingConfig(lora_r=32, num_epochs=5, learning_rate=5e-5)

        assert config.lora_r == 32
        assert config.num_epochs == 5
        assert config.learning_rate == 5e-5
