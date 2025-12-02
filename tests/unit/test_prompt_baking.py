"""
Unit tests for Prompt Baking System
Tests KL divergence baking, half-baking, and prompt management
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.prompt_baking.prompts import PhasePrompts, PromptManager
from cross_phase.prompt_baking.baker import PromptBakingConfig


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
            "analytical_debugger",
            "rapid_prototyper",
            "detail_oriented_reviewer",
            "systems_architect",
            "user_advocate",
            "optimizer",
            "documentation_expert"
        ]

        for persona in expected_personas:
            assert persona in personas
            assert isinstance(personas[persona], str)
            assert len(personas[persona]) > 0

    def test_phase6_tool_use_prompt(self):
        """Test Phase 6 tool use prompt exists"""
        prompt = PhasePrompts.PHASE6_TOOL_USE

        assert isinstance(prompt, str)
        assert "tool" in prompt.lower()


class TestPromptManager:
    """Test PromptManager"""

    def test_get_phase3_prompt(self):
        """Test getting Phase 3 prompt"""
        manager = PromptManager()
        prompt = manager.get_prompt(phase=3, prompt_type="cot_reasoning")

        assert prompt == PhasePrompts.PHASE3_COT_REASONING

    def test_get_phase5_prompts(self):
        """Test getting Phase 5 prompts"""
        manager = PromptManager()

        eudaimonia = manager.get_prompt(5, "eudaimonia")
        tool_use = manager.get_prompt(5, "tool_use")

        assert eudaimonia == PhasePrompts.PHASE5_EUDAIMONIA
        assert tool_use == PhasePrompts.PHASE5_TOOL_USE

    def test_get_phase6_persona(self):
        """Test getting Phase 6 persona"""
        manager = PromptManager()

        persona = manager.get_prompt(6, "reasoning_specialist")

        assert persona == PhasePrompts.PHASE6_PERSONAS["reasoning_specialist"]

    def test_get_phase6_tool_prompt(self):
        """Test getting Phase 6 tool prompt"""
        manager = PromptManager()

        prompt = manager.get_prompt(6, "tool_use")

        assert prompt == PhasePrompts.PHASE6_TOOL_USE

    def test_list_available_prompts(self):
        """Test listing available prompts"""
        manager = PromptManager()

        prompts = manager.list_available_prompts()

        assert isinstance(prompts, dict)
        assert 3 in prompts
        assert 5 in prompts
        assert 6 in prompts

    def test_invalid_phase(self):
        """Test invalid phase raises error"""
        manager = PromptManager()

        with pytest.raises(ValueError):
            manager.get_prompt(99, "invalid")


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
        config = PromptBakingConfig(
            lora_r=32,
            num_epochs=5,
            learning_rate=5e-5
        )

        assert config.lora_r == 32
        assert config.num_epochs == 5
        assert config.learning_rate == 5e-5
