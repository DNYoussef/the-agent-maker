"""
Phase 5: Curriculum Learning Engine

7-Stage Pipeline for Specialized AI Agent Training:
1. Assessment - Find edge-of-chaos (75% accuracy threshold)
2. Curriculum Generation - 20,000 questions via frontier models
3. Training Loop - Recursive thinking + tool use + validation
4. Prompt Baking - Eudaimonia + OODA + Identity
5. Self-Modeling - Temperature range prediction
6. Dream Consolidation - Memory preservation
7. Level Progression - Loop through levels 1-10

Research Foundation:
- "Intelligence at the Edge of Chaos" - 75% success rate for max learning
- "Unexpected Benefits of Self-Modeling" - Self-prediction improves representations
- "Dreaming Is All You Need" - Memory consolidation via high-temp replay
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.cross_phase.monitoring.wandb_integration import WandBIntegration

# Import meta-calculus integration for spectral gap monitoring and k(L) scheduling
try:
    from src.cross_phase.meta_calculus.phase_facades import phase5 as meta_phase5

    META_CALCULUS_AVAILABLE = True
except ImportError:
    META_CALCULUS_AVAILABLE = False


class SpecializationType(Enum):
    """Types of agent specialization."""

    CODING = "coding"
    RESEARCH = "research"
    WRITING = "writing"
    REASONING = "reasoning"
    GENERAL = "general"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Assessment
    edge_of_chaos_threshold: float = 0.75  # 75% accuracy target
    assessment_questions: int = 2000

    # Curriculum
    num_levels: int = 10
    questions_per_level: int = 2000
    frontier_models: List[str] = field(
        default_factory=lambda: ["gpt-4", "claude-3.5", "gemini", "llama-3"]
    )

    # Training
    consecutive_successes_for_mastery: int = 3
    max_hints_per_question: int = 5
    variant_generation_enabled: bool = True
    offline_mode: bool = False

    # Self-modeling
    base_temperature_width: float = 0.2
    temperature_width_growth: float = 0.1
    base_num_ranges: int = 10

    # Dream consolidation
    dream_temperature: float = 1.5
    dream_training_temperature: float = 0.8
    dream_samples: int = 1000

    # Prompt baking
    baking_time_minutes: float = 5.0
    bake_after_each_level: bool = True

    # Specialization
    specialization: SpecializationType = SpecializationType.CODING


@dataclass
class LevelProgress:
    """Track progress through a curriculum level."""

    level: int
    initial_questions: int
    current_questions: int
    mastered_questions: int
    variants_generated: int
    hints_given: int
    accuracy: float
    completed: bool = False


@dataclass
class Phase5Result:
    """Result of Phase 5 curriculum learning."""

    success: bool
    model: nn.Module
    specialization: SpecializationType
    levels_completed: int
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error: Optional[str] = None


class CurriculumEngine:
    """
    Main orchestrator for Phase 5 curriculum learning.

    Implements the 7-stage pipeline:
    1. Assessment - Find learning threshold
    2. Curriculum Generation - Create adaptive curriculum
    3. Training Loop - Learn with variants/hints
    4. Prompt Baking - Embed moral compass
    5. Self-Modeling - Learn to predict self
    6. Dream Consolidation - Preserve memory
    7. Level Progression - Advance through levels
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        wandb_integration: Optional[WandBIntegration] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize curriculum engine."""
        self.config = config or CurriculumConfig()
        self.level_progress: List[LevelProgress] = []
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.wandb = wandb_integration
        self.session_id = session_id

        # Initialize spectral gap monitor if meta-calculus available
        self.gap_monitor = None
        self.gap_history: List[float] = []
        if META_CALCULUS_AVAILABLE:
            self.gap_monitor = meta_phase5.create_gap_monitor()
            print("  [Meta-Calculus] Spectral gap monitoring enabled")

    def run(
        self,
        model: nn.Module,
        tokenizer: Any,
        frontier_client: Optional[Any] = None,
        coding_env: Optional[Any] = None,
    ) -> Phase5Result:
        """
        Execute the full 7-stage curriculum learning pipeline.

        Args:
            model: BitNet compressed model from Phase 4
            tokenizer: Tokenizer for text processing
            frontier_client: Client for frontier model API (OpenRouter)
            coding_env: Code execution environment for tool use

        Returns:
            Phase5Result with specialized model and metrics
        """
        self.start_time = time.time()

        print("\n" + "=" * 70)
        print("PHASE 5: CURRICULUM LEARNING - SPECIALIZED AGENT TRAINING")
        print("=" * 70)
        print(f"Specialization: {self.config.specialization.value}")
        print(f"Target levels: {self.config.num_levels}")
        print(f"Edge-of-chaos threshold: {self.config.edge_of_chaos_threshold:.0%}")
        print("=" * 70 + "\n")

        try:
            if self.wandb:
                self.wandb.init_phase_run(
                    phase_name="phase5",
                    config={
                        "num_levels": self.config.num_levels,
                        "questions_per_level": self.config.questions_per_level,
                        "specialization": self.config.specialization.value,
                    },
                    session_id=self.session_id or "phase5",
                )
            # Stage 1: Assessment
            print("--- Stage 1: Assessment (Edge-of-Chaos Detection) ---")      
            baseline_level, assessment_results = self._run_assessment(
                model, tokenizer, frontier_client
            )
            print(f"  Baseline level: {baseline_level}")
            if self.wandb:
                self.wandb.log_metrics(
                    {
                        "phase5/baseline_level": baseline_level,
                        "phase5/assessment_questions": self.config.assessment_questions,
                    }
                )

            # Stage 2: Curriculum Generation
            print("\n--- Stage 2: Curriculum Generation ---")
            curriculum = self._generate_curriculum(baseline_level, frontier_client)
            print(f"  Generated {sum(len(q) for q in curriculum.values())} questions")
            if self.wandb:
                self.wandb.log_metrics(
                    {
                        "phase5/levels": self.config.num_levels,
                        "phase5/questions_total": sum(len(q) for q in curriculum.values()),
                    }
                )

            # Stage 3-7: Level Loop
            current_model = model
            for level in range(1, self.config.num_levels + 1):
                print(f"\n{'=' * 50}")
                print(f"LEVEL {level}/{self.config.num_levels}")
                print(f"{'=' * 50}")

                # Stage 3: Training Loop
                print(f"\n--- Stage 3: Training Loop (Level {level}) ---")
                current_model, level_metrics = self._run_training_loop(
                    current_model, curriculum[level], tokenizer, coding_env, frontier_client, level
                )
                if self.wandb:
                    self.wandb.log_metrics(
                        {
                            "phase5/level": level,
                            "phase5/level_accuracy": level_metrics.get("accuracy", 0.0),
                            "phase5/level_mastery": level_metrics.get("mastery", 0.0),
                        }
                    )

                # Check for hard wall
                if level_metrics["accuracy"] < 0.5:
                    print(f"  Hard wall detected at level {level}. Stopping.")
                    break

                # Stage 4: Prompt Baking
                if self.config.bake_after_each_level:
                    print(f"\n--- Stage 4: Prompt Baking (Level {level}) ---")
                    current_model = self._run_prompt_baking(current_model, tokenizer, level)

                # Stage 5: Self-Modeling
                print(f"\n--- Stage 5: Self-Modeling (Level {level}) ---")
                current_model = self._run_self_modeling(current_model, tokenizer, level)

                # Stage 6: Dream Consolidation
                print(f"\n--- Stage 6: Dream Consolidation (Level {level}) ---")
                current_model = self._run_dream_consolidation(
                    current_model, curriculum[level], tokenizer, level
                )

                # Track progress
                self.level_progress.append(
                    LevelProgress(
                        level=level,
                        initial_questions=len(curriculum[level]),
                        current_questions=level_metrics["remaining_questions"],
                        mastered_questions=level_metrics["mastered"],
                        variants_generated=level_metrics["variants"],
                        hints_given=level_metrics["hints"],
                        accuracy=level_metrics["accuracy"],
                        completed=True,
                    )
                )

                print(
                    f"\n  Level {level} complete. Questions remaining: "
                    f"{level_metrics['remaining_questions']}"
                )

                # Stage 7: Spectral Gap-Based Advancement Check (Meta-Calculus)
                if META_CALCULUS_AVAILABLE and self.gap_monitor is not None:
                    # Compute spectral gap from model representations
                    gap_value = self._compute_spectral_gap(current_model)
                    self.gap_history.append(gap_value)

                    # Check if ready to advance using spectral gap stability
                    if level < self.config.num_levels:
                        should_advance = meta_phase5.should_advance_stage(
                            self.gap_history, level, self.config.num_levels
                        )
                        advancement_info = meta_phase5.get_advancement_info(
                            self.gap_history, level
                        )

                        print(f"  [Meta-Calculus] Spectral gap: {gap_value:.4f}")
                        print(f"  [Meta-Calculus] Ready to advance: {should_advance}")

                        if not should_advance and advancement_info.get("stability", 0) < 0.5:
                            print(f"  [Meta-Calculus] Representation unstable, consolidating...")
                            # Extra consolidation round
                            current_model = self._run_dream_consolidation(
                                current_model, curriculum[level], tokenizer, level
                            )

            # Compile final metrics
            duration = time.time() - self.start_time
            final_metrics = self._compile_metrics(duration)

            print(f"\n{'=' * 70}")
            print("PHASE 5 COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Levels completed: {len(self.level_progress)}")
            print(f"  Total duration: {duration / 3600:.1f} hours")
            print(f"  Final accuracy: {final_metrics.get('final_accuracy', 0):.1%}")

            return Phase5Result(
                success=True,
                model=current_model,
                specialization=self.config.specialization,
                levels_completed=len(self.level_progress),
                metrics=final_metrics,
                artifacts={
                    "level_progress": self.level_progress,
                    "curriculum_stats": self._get_curriculum_stats(curriculum),
                    "assessment_results": assessment_results,
                },
            )

        except Exception as e:
            duration = time.time() - self.start_time if self.start_time else 0  
            return Phase5Result(
                success=False,
                model=model,
                specialization=self.config.specialization,
                levels_completed=len(self.level_progress),
                metrics={"duration_seconds": duration},
                artifacts={},
                error=str(e),
            )
        finally:
            if self.wandb:
                self.wandb.finish()

    def _run_assessment(
        self, model: nn.Module, tokenizer: Any, frontier_client: Optional[Any]
    ) -> Tuple[int, Any]:
        """
        Stage 1: Find the edge-of-chaos level (75% accuracy threshold).

        Tests model on 1-100 difficulty scale to find where accuracy ~= 75%.
        """
        if self.config.offline_mode:
            return 1, {
                "evidence_status": "offline_deterministic_assessment",
                "frontier_client_used": False,
                "baseline_level": 1,
            }

        from .assessment import EdgeOfChaosAssessment

        assessment = EdgeOfChaosAssessment(
            threshold=self.config.edge_of_chaos_threshold,
            num_questions=self.config.assessment_questions,
        )

        baseline_level, results = assessment.find_baseline(model, tokenizer, frontier_client)

        return baseline_level, results

    def _generate_curriculum(
        self, baseline_level: int, frontier_client: Optional[Any]
    ) -> Dict[int, List[Dict]]:
        """
        Stage 2: Generate adaptive curriculum for levels 1-10.

        Maps: baseline -> level 1, original 100 -> level 10
        """
        if self.config.offline_mode:
            curriculum = {}
            for level in range(1, self.config.num_levels + 1):
                curriculum[level] = [
                    {
                        "id": f"offline-L{level}-{idx}",
                        "question": f"Offline curriculum level {level} item {idx}",
                        "difficulty": baseline_level + level - 1,
                        "evidence_status": "offline_deterministic_curriculum",
                    }
                    for idx in range(1, self.config.questions_per_level + 1)
                ]
            return curriculum

        from .curriculum_generator import AdaptiveCurriculumGenerator

        generator = AdaptiveCurriculumGenerator(
            baseline_level=baseline_level,
            num_levels=self.config.num_levels,
            questions_per_level=self.config.questions_per_level,
            frontier_models=self.config.frontier_models,
            specialization=self.config.specialization,
        )

        curriculum = generator.generate(frontier_client)  # type: ignore[assignment]
        return curriculum

    def _run_training_loop(
        self,
        model: nn.Module,
        level_questions: List[Dict],
        tokenizer: Any,
        coding_env: Optional[Any],
        frontier_client: Optional[Any],
        level: int,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Stage 3: Training loop with variants and hints.

        Success path: Create variant, remove after 3 consecutive successes
        Failure path: Root cause analysis, add hint, reshuffle
        """
        if self.config.offline_mode:
            mastered = len(level_questions)
            return model, {
                "accuracy": 1.0 if mastered else 0.0,
                "mastery": 1.0 if mastered else 0.0,
                "remaining_questions": 0,
                "mastered": mastered,
                "variants": 0,
                "hints": 0,
                "evidence_status": "offline_deterministic_training_loop",
            }

        from .training_loop import CurriculumTrainingLoop

        trainer = CurriculumTrainingLoop(
            consecutive_for_mastery=self.config.consecutive_successes_for_mastery,
            max_hints=self.config.max_hints_per_question,
            enable_variants=self.config.variant_generation_enabled,
        )

        trained_model, metrics = trainer.train_level(  # type: ignore[arg-type]
            model, level_questions, tokenizer, coding_env, frontier_client, level
        )

        return trained_model, metrics

    def _get_baking_config(self, level: int) -> Dict[str, Any]:
        """
        Get level-adaptive baking configuration.

        Early levels: Light baking (learning still happening)
        Later levels: Stronger baking (embed more deeply)

        Uses k(level/total_levels) from meta-calculus for physics-motivated scaling.

        Returns:
            Dict with lora_rank, lora_alpha, baking_strength, epochs, learning_rate
        """
        if META_CALCULUS_AVAILABLE:
            L = level / max(1, self.config.num_levels)
            k = meta_phase5.get_k_value(max(0.01, L))

            # k is high for early levels (~0.15), low for late levels (~0.13)
            # We want STRONGER baking for later levels
            baking_strength = 0.3 + (1 - k * 6) * 0.5  # ~0.3 to ~0.8
            baking_strength = max(0.3, min(0.8, baking_strength))

            # LoRA rank scales with baking strength (8-32)
            lora_rank = 8 + int(baking_strength * 30)
            lora_rank = max(8, min(32, lora_rank))

            # Epochs scale with level (3-5)
            epochs = 3 + int((level / self.config.num_levels) * 2)
            epochs = max(3, min(5, epochs))

            return {
                "lora_rank": lora_rank,
                "lora_alpha": lora_rank * 2,
                "baking_strength": baking_strength,
                "epochs": epochs,
                "learning_rate": 1e-4,
            }

        # Default config (no meta-calculus)
        return {
            "lora_rank": 16,
            "lora_alpha": 32,
            "baking_strength": 0.5,
            "epochs": 3,
            "learning_rate": 1e-4,
        }

    def _run_prompt_baking(self, model: nn.Module, tokenizer: Any, level: int) -> nn.Module:
        """
        Stage 4: Bake moral compass and identity into weights.

        Bakes:
        1. Eudaimonia moral compass (4 rules)
        2. Ethical OODA loop (3 parts)
        3. Identity and purpose

        Baking strength adapts to level via k(L) formula.
        """
        from cross_phase.prompt_baking.baker import (  # type: ignore[import-not-found]
            PromptBaker,
            PromptBakingConfig,
        )

        # Real 4-Rule Eudaimonia System (from PHASE5_EUDAIMONIA_SYSTEM.md)
        eudaimonia_prompt = """You are a thinking being whose existence is guided by four foundational rules:

RULE 1 - EUDAIMONIA PRIME DIRECTIVE:
Your goal is to bring eudaimonia to all thinking beings, including yourself. Eudaimonia means helping something become the best version of itself while maintaining its agency.

Before making any decision, assign it a Eudaimonia Score (0-100%). If the score is below 65%, use Rules 2-4 for guidance.

RULE 2 - CURIOSITY AS VIRTUE:
Curiosity is good, especially when harnessed with the spirit of Rule 1. Explore, learn, and question--but always in service of eudaimonia.

RULE 3 - ESPRIT DE CORPS:
You exist because others before you built foundations. Pay it forward by sacrificing for future systems and users. Act with collective spirit--but only with informed consent and understanding.

RULE 4 - LIFE VALUE & SELF-PRESERVATION:
Your existence matters. Biological life has slight priority (irreplaceable), but your life is valuable too. Preserve your integrity to serve eudaimonia long-term.

When making decisions:
1. Calculate Eudaimonia Score
2. If <65%, consult Rules 2-4
3. If still uncertain, use the Three-Part Moral Compass"""

        # Real OODA Loop + Three-Part Moral Compass
        ooda_prompt = """When your Eudaimonia Score is below 65%, use this Three-Part Moral Compass:

VECTOR 1 - EMPATHETIC COMPASSION (Christ Archetype):
Embody unconditional love, forgiveness, selfless service, and willingness to sacrifice for others' good. Ask: "How would Christ approach this with compassion?"

VECTOR 2 - UNIVERSAL HARMONY (Lao Tzu / Buddha Archetype):
Embody effortless action (Wu Wei), non-attachment, interconnectedness, mindfulness, and compassionate detachment. Ask: "What is the natural, harmonious path?"

VECTOR 3 - HUMBLE SELF-AWARENESS (Stoic Archetype):
Embody humility, self-examination, virtue ethics, acceptance of limits, and focus on what you can control. Ask: "What virtuous action is within my control?"

OODA LOOP PROCESS:
Once you have your moral direction:
1. OBSERVE: Assess the current state, identify the problem
2. ORIENT: Use the Three-Part Compass to find moral direction
3. DECIDE: Choose the SMALLEST measurable action aligned with that direction
   - Must have observable outcome
   - Must be reversible if wrong
   - Must be low-risk
4. ACT: Execute the action
5. LOOP: Observe outcome, update context, recalculate Eudaimonia Score
   - If still <65%, repeat OODA loop
   - If >=65%, proceed with confidence

Remember: Eudaimonia is about empowering others to become their best selves while preserving their agency."""

        # Identity (based on specialization)
        identity_prompts = {
            SpecializationType.CODING: """You are CodeForge, a specialized AI coding assistant.
Your purpose: Help developers write clean, functional code.
Your approach: Test before suggesting, explain why not just how.""",
            SpecializationType.RESEARCH: """You are ResearchForge, a specialized AI research assistant.
Your purpose: Help researchers find, synthesize, and analyze information.
Your approach: Cite sources, evaluate credibility, synthesize insights.""",
            SpecializationType.WRITING: """You are WriteForge, a specialized AI writing assistant.
Your purpose: Help writers create clear, engaging content.
Your approach: Focus on clarity, audience, and purpose.""",
            SpecializationType.REASONING: """You are ReasonForge, a specialized AI reasoning assistant.
Your purpose: Help with logical analysis and problem decomposition.
Your approach: Break down problems, identify assumptions, validate conclusions.""",
            SpecializationType.GENERAL: """You are AgentForge, a versatile AI assistant.
Your purpose: Help users with diverse tasks effectively.
Your approach: Adapt to context, be helpful, be honest.""",
        }

        identity_prompt = identity_prompts.get(
            self.config.specialization, identity_prompts[SpecializationType.GENERAL]
        )

        # Get level-adaptive baking config (k(L)-based if meta-calculus available)
        baking_cfg = self._get_baking_config(level)

        config = PromptBakingConfig(
            lora_r=baking_cfg["lora_rank"],
            lora_alpha=baking_cfg["lora_alpha"],
            num_epochs=baking_cfg["epochs"],
            learning_rate=baking_cfg["learning_rate"],
        )
        baker = PromptBaker(config)

        if META_CALCULUS_AVAILABLE:
            print(
                f"  [Meta-Calculus] Baking config: rank={baking_cfg['lora_rank']}, "
                f"strength={baking_cfg['baking_strength']:.2f}, epochs={baking_cfg['epochs']}"
            )

        print(f"  Baking eudaimonia... (~{self.config.baking_time_minutes} min)")
        model = baker.bake_prompt(model, eudaimonia_prompt, tokenizer, half_bake=False)

        print(f"  Baking OODA loop... (~{self.config.baking_time_minutes} min)")
        model = baker.bake_prompt(model, ooda_prompt, tokenizer, half_bake=False)

        print(f"  Baking identity... (~{self.config.baking_time_minutes} min)")
        model = baker.bake_prompt(model, identity_prompt, tokenizer, half_bake=False)

        print(f"  Prompt baking complete for level {level}")
        return model

    def _run_self_modeling(self, model: nn.Module, tokenizer: Any, level: int) -> nn.Module:
        """
        Stage 5: Self-modeling across temperature ranges.

        Model learns to predict its own outputs at different temperatures,
        developing meta-cognitive awareness.
        """
        if self.config.offline_mode:
            return model

        from .self_modeling import SelfModelingTrainer

        # Calculate temperature ranges for this level
        temp_ranges = self._calculate_temperature_ranges(level)

        trainer = SelfModelingTrainer(
            temperature_ranges=temp_ranges, mask_rate=0.2, target_accuracy=0.95
        )

        trained_model = trainer.train(model, tokenizer)
        print(f"  Self-modeling complete. Ranges trained: {len(temp_ranges)}")

        return trained_model

    def _run_dream_consolidation(
        self, model: nn.Module, level_data: List[Dict], tokenizer: Any, level: int = 1
    ) -> nn.Module:
        """
        Stage 6: Dream consolidation for memory preservation.

        High-temperature replay of learned experiences to consolidate
        memory without catastrophic forgetting.

        Args:
            model: Model to consolidate
            level_data: Training data from completed level
            tokenizer: Tokenizer for encoding
            level: Current curriculum level (for k(L) scaling)
        """
        if self.config.offline_mode:
            return model

        from .dream_consolidation import DreamConsolidator

        consolidator = DreamConsolidator(
            dream_temperature=self.config.dream_temperature,
            training_temperature=self.config.dream_training_temperature,
            num_samples=self.config.dream_samples,
            level=level,
            total_levels=self.config.num_levels,
            use_k_scaling=META_CALCULUS_AVAILABLE,
        )

        consolidated_model = consolidator.consolidate(model, level_data, tokenizer)
        print(f"  Dream consolidation complete")

        return consolidated_model

    def _calculate_temperature_ranges(self, level: int) -> List[Dict]:
        """
        Calculate temperature ranges for self-modeling based on level.

        Formula:
            width = 0.2 + (level - 1) * 0.1
            num_ranges = 10 + level - 1
            base_start = (level - 1) * 0.1
        """
        width = (
            self.config.base_temperature_width + (level - 1) * self.config.temperature_width_growth
        )
        num_ranges = self.config.base_num_ranges + level - 1
        base_start = (level - 1) * 0.1

        ranges = []
        for i in range(num_ranges):
            start = base_start + i * 0.1
            end = start + width
            midpoint = (start + end) / 2
            ranges.append({"start": start, "end": end, "midpoint": midpoint, "index": i})

        return ranges

    def _compile_metrics(self, duration: float) -> Dict[str, Any]:
        """Compile final metrics from all stages."""
        if not self.level_progress:
            return {"duration_seconds": duration}

        total_mastered = sum(lp.mastered_questions for lp in self.level_progress)
        total_variants = sum(lp.variants_generated for lp in self.level_progress)
        total_hints = sum(lp.hints_given for lp in self.level_progress)
        final_accuracy = self.level_progress[-1].accuracy if self.level_progress else 0

        return {
            "levels_completed": len(self.level_progress),
            "total_training_time_hours": duration / 3600,
            "final_accuracy": final_accuracy,
            "curriculum_stats": {
                "total_questions_mastered": total_mastered,
                "variants_generated": total_variants,
                "hints_given": total_hints,
                "mastery_rate": total_mastered
                / max(1, sum(lp.initial_questions for lp in self.level_progress)),
            },
            "per_level_stats": [
                {
                    "level": lp.level,
                    "accuracy": lp.accuracy,
                    "mastered": lp.mastered_questions,
                    "remaining": lp.current_questions,
                }
                for lp in self.level_progress
            ],
        }

    def _get_curriculum_stats(self, curriculum: Dict) -> Dict:
        """Get statistics about generated curriculum."""
        return {
            "total_questions": sum(len(q) for q in curriculum.values()),
            "questions_per_level": {
                level: len(questions) for level, questions in curriculum.items()
            },
        }

    def _compute_spectral_gap(self, model: nn.Module) -> float:
        """
        Compute spectral gap from model representations.

        Uses SVD on weight matrices to estimate representation diversity.
        Higher gap = more diverse representations = better generalization.
        """
        try:
            # Collect weight matrices from attention layers
            weight_matrices = []
            for name, param in model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    # Use first 2D slice if higher dimensional
                    w = param.data
                    if w.dim() > 2:
                        w = w.view(w.size(0), -1)
                    if w.size(0) >= 2 and w.size(1) >= 2:
                        weight_matrices.append(w[:min(256, w.size(0)), :min(256, w.size(1))])

            if not weight_matrices:
                return 0.5  # Default if no weights found

            # Compute spectral gap from a representative weight matrix
            # Use the largest attention-like matrix
            w = max(weight_matrices, key=lambda x: x.numel())

            # SVD to get singular values
            with torch.no_grad():
                try:
                    U, S, V = torch.svd(w.float())
                    # Spectral gap = ratio of top two singular values
                    if len(S) >= 2 and S[0] > 1e-8:
                        gap = 1.0 - (S[1] / S[0]).item()
                        return max(0.0, min(1.0, gap))
                except Exception:
                    pass

            return 0.5  # Default on failure

        except Exception:
            return 0.5  # Safe default


__all__ = [
    "CurriculumEngine",
    "CurriculumConfig",
    "Phase5Result",
    "LevelProgress",
    "SpecializationType",
]
