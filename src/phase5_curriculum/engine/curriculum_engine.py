"""
Phase 5: Curriculum Learning Engine

Main orchestrator for curriculum learning pipeline.

7-Stage Pipeline:
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

from typing import Dict, List, Optional, Any
import time

import torch.nn as nn

from .config import CurriculumConfig, SpecializationType
from .progress import LevelProgress
from .result import Phase5Result


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

    def __init__(self, config: Optional[CurriculumConfig] = None):
        """Initialize curriculum engine."""
        self.config = config or CurriculumConfig()
        self.level_progress: List[LevelProgress] = []
        self.metrics = {}
        self.start_time = None

    def run(
        self,
        model: nn.Module,
        tokenizer: Any,
        frontier_client: Optional[Any] = None,
        coding_env: Optional[Any] = None
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
            # Stage 1: Assessment
            print("--- Stage 1: Assessment (Edge-of-Chaos Detection) ---")
            baseline_level, assessment_results = self._run_assessment(
                model, tokenizer, frontier_client
            )
            print(f"  Baseline level: {baseline_level}")

            # Stage 2: Curriculum Generation
            print("\n--- Stage 2: Curriculum Generation ---")
            curriculum = self._generate_curriculum(
                baseline_level, frontier_client
            )
            print(f"  Generated {sum(len(q) for q in curriculum.values())} questions")

            # Stage 3-7: Level Loop
            current_model = model
            for level in range(1, self.config.num_levels + 1):
                print(f"\n{'=' * 50}")
                print(f"LEVEL {level}/{self.config.num_levels}")
                print(f"{'=' * 50}")

                # Stage 3: Training Loop
                print(f"\n--- Stage 3: Training Loop (Level {level}) ---")
                current_model, level_metrics = self._run_training_loop(
                    current_model, curriculum[level], tokenizer,
                    coding_env, frontier_client, level
                )

                # Check for hard wall
                if level_metrics['accuracy'] < 0.5:
                    print(f"  Hard wall detected at level {level}. Stopping.")
                    break

                # Stage 4: Prompt Baking
                if self.config.bake_after_each_level:
                    print(f"\n--- Stage 4: Prompt Baking (Level {level}) ---")
                    current_model = self._run_prompt_baking(
                        current_model, tokenizer, level
                    )

                # Stage 5: Self-Modeling
                print(f"\n--- Stage 5: Self-Modeling (Level {level}) ---")
                current_model = self._run_self_modeling(
                    current_model, tokenizer, level
                )

                # Stage 6: Dream Consolidation
                print(f"\n--- Stage 6: Dream Consolidation (Level {level}) ---")
                current_model = self._run_dream_consolidation(
                    current_model, curriculum[level], tokenizer
                )

                # Track progress
                self.level_progress.append(LevelProgress(
                    level=level,
                    initial_questions=len(curriculum[level]),
                    current_questions=level_metrics['remaining_questions'],
                    mastered_questions=level_metrics['mastered'],
                    variants_generated=level_metrics['variants'],
                    hints_given=level_metrics['hints'],
                    accuracy=level_metrics['accuracy'],
                    completed=True
                ))

                print(f"\n  Level {level} complete. Questions remaining: "
                      f"{level_metrics['remaining_questions']}")

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
                    'level_progress': self.level_progress,
                    'curriculum_stats': self._get_curriculum_stats(curriculum),
                    'assessment_results': assessment_results
                }
            )

        except Exception as e:
            duration = time.time() - self.start_time if self.start_time else 0
            return Phase5Result(
                success=False,
                model=model,
                specialization=self.config.specialization,
                levels_completed=len(self.level_progress),
                metrics={'duration_seconds': duration},
                artifacts={},
                error=str(e)
            )

    def _run_assessment(
        self,
        model: nn.Module,
        tokenizer: Any,
        frontier_client: Optional[Any]
    ) -> tuple:
        """
        Stage 1: Find the edge-of-chaos level (75% accuracy threshold).

        Tests model on 1-100 difficulty scale to find where accuracy ~= 75%.
        """
        from ..assessment import EdgeOfChaosAssessment

        assessment = EdgeOfChaosAssessment(
            threshold=self.config.edge_of_chaos_threshold,
            num_questions=self.config.assessment_questions
        )

        baseline_level, results = assessment.find_baseline(
            model, tokenizer, frontier_client
        )

        return baseline_level, results

    def _generate_curriculum(
        self,
        baseline_level: int,
        frontier_client: Optional[Any]
    ) -> Dict[int, List[Dict]]:
        """
        Stage 2: Generate adaptive curriculum for levels 1-10.

        Maps: baseline -> level 1, original 100 -> level 10
        """
        from ..curriculum_generator import AdaptiveCurriculumGenerator

        generator = AdaptiveCurriculumGenerator(
            baseline_level=baseline_level,
            num_levels=self.config.num_levels,
            questions_per_level=self.config.questions_per_level,
            frontier_models=self.config.frontier_models,
            specialization=self.config.specialization
        )

        curriculum = generator.generate(frontier_client)
        return curriculum

    def _run_training_loop(
        self,
        model: nn.Module,
        level_questions: List[Dict],
        tokenizer: Any,
        coding_env: Optional[Any],
        frontier_client: Optional[Any],
        level: int
    ) -> tuple:
        """
        Stage 3: Training loop with variants and hints.

        Success path: Create variant, remove after 3 consecutive successes
        Failure path: Root cause analysis, add hint, reshuffle
        """
        from ..training_loop import CurriculumTrainingLoop

        trainer = CurriculumTrainingLoop(
            consecutive_for_mastery=self.config.consecutive_successes_for_mastery,
            max_hints=self.config.max_hints_per_question,
            enable_variants=self.config.variant_generation_enabled
        )

        trained_model, metrics = trainer.train_level(
            model, level_questions, tokenizer, coding_env, frontier_client, level
        )

        return trained_model, metrics

    def _run_prompt_baking(
        self,
        model: nn.Module,
        tokenizer: Any,
        level: int
    ) -> nn.Module:
        """
        Stage 4: Bake moral compass and identity into weights.

        Bakes:
        1. Eudaimonia moral compass (4 rules)
        2. Ethical OODA loop (3 parts)
        3. Identity and purpose
        """
        from cross_phase.prompt_baking.baker import PromptBaker, PromptBakingConfig

        # Eudaimonia rules (placeholder - user defines)
        eudaimonia_prompt = """You are guided by these four principles:
1. Seek to understand before acting
2. Consider consequences for all affected parties
3. Be truthful and transparent in your reasoning
4. Continuously learn and improve

When making decisions, always consider these principles first."""

        # OODA loop
        ooda_prompt = """Before taking action, follow this process:
1. OBSERVE: Gather information about the situation and context
2. ORIENT: Align your understanding with your principles and knowledge
3. DECIDE: Choose the best course of action transparently
4. ACT: Execute only after completing steps 1-3."""

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
Your approach: Adapt to context, be helpful, be honest."""
        }

        identity_prompt = identity_prompts.get(
            self.config.specialization,
            identity_prompts[SpecializationType.GENERAL]
        )

        # Sequential baking
        config = PromptBakingConfig(
            lora_r=16,
            lora_alpha=32,
            num_epochs=3,
            learning_rate=1e-4
        )
        baker = PromptBaker(config)

        print(f"  Baking eudaimonia... (~{self.config.baking_time_minutes} min)")
        model = baker.bake_prompt(model, eudaimonia_prompt, tokenizer, half_bake=False)

        print(f"  Baking OODA loop... (~{self.config.baking_time_minutes} min)")
        model = baker.bake_prompt(model, ooda_prompt, tokenizer, half_bake=False)

        print(f"  Baking identity... (~{self.config.baking_time_minutes} min)")
        model = baker.bake_prompt(model, identity_prompt, tokenizer, half_bake=False)

        print(f"  Prompt baking complete for level {level}")
        return model

    def _run_self_modeling(
        self,
        model: nn.Module,
        tokenizer: Any,
        level: int
    ) -> nn.Module:
        """
        Stage 5: Self-modeling across temperature ranges.

        Model learns to predict its own outputs at different temperatures,
        developing meta-cognitive awareness.
        """
        from ..self_modeling import SelfModelingTrainer

        # Calculate temperature ranges for this level
        temp_ranges = self._calculate_temperature_ranges(level)

        trainer = SelfModelingTrainer(
            temperature_ranges=temp_ranges,
            mask_rate=0.2,
            target_accuracy=0.95
        )

        trained_model = trainer.train(model, tokenizer)
        print(f"  Self-modeling complete. Ranges trained: {len(temp_ranges)}")

        return trained_model

    def _run_dream_consolidation(
        self,
        model: nn.Module,
        level_data: List[Dict],
        tokenizer: Any
    ) -> nn.Module:
        """
        Stage 6: Dream consolidation for memory preservation.

        High-temperature replay of learned experiences to consolidate
        memory without catastrophic forgetting.
        """
        from ..dream_consolidation import DreamConsolidator

        consolidator = DreamConsolidator(
            dream_temperature=self.config.dream_temperature,
            training_temperature=self.config.dream_training_temperature,
            num_samples=self.config.dream_samples
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
        width = self.config.base_temperature_width + (level - 1) * self.config.temperature_width_growth
        num_ranges = self.config.base_num_ranges + level - 1
        base_start = (level - 1) * 0.1

        ranges = []
        for i in range(num_ranges):
            start = base_start + i * 0.1
            end = start + width
            midpoint = (start + end) / 2
            ranges.append({
                'start': start,
                'end': end,
                'midpoint': midpoint,
                'index': i
            })

        return ranges

    def _compile_metrics(self, duration: float) -> Dict[str, Any]:
        """Compile final metrics from all stages."""
        if not self.level_progress:
            return {'duration_seconds': duration}

        total_mastered = sum(lp.mastered_questions for lp in self.level_progress)
        total_variants = sum(lp.variants_generated for lp in self.level_progress)
        total_hints = sum(lp.hints_given for lp in self.level_progress)
        final_accuracy = self.level_progress[-1].accuracy if self.level_progress else 0

        return {
            'levels_completed': len(self.level_progress),
            'total_training_time_hours': duration / 3600,
            'final_accuracy': final_accuracy,
            'curriculum_stats': {
                'total_questions_mastered': total_mastered,
                'variants_generated': total_variants,
                'hints_given': total_hints,
                'mastery_rate': total_mastered / max(1, sum(
                    lp.initial_questions for lp in self.level_progress
                ))
            },
            'per_level_stats': [
                {
                    'level': lp.level,
                    'accuracy': lp.accuracy,
                    'mastered': lp.mastered_questions,
                    'remaining': lp.current_questions
                }
                for lp in self.level_progress
            ]
        }

    def _get_curriculum_stats(self, curriculum: Dict) -> Dict:
        """Get statistics about generated curriculum."""
        return {
            'total_questions': sum(len(q) for q in curriculum.values()),
            'questions_per_level': {
                level: len(questions)
                for level, questions in curriculum.items()
            }
        }
