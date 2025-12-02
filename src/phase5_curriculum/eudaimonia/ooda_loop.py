"""
OODA Loop for Eudaimonia Alignment

OODA = Observe, Orient, Decide, Act (military decision-making framework)

When Eudaimonia score falls below 65%, the OODA loop provides a structured
process for ethical decision-making:

1. OBSERVE: Gather information, calculate Eudaimonia score
2. ORIENT: Query archetypes, find moral direction
3. DECIDE: Choose SMALLEST measurable action
4. ACT: Execute action
5. LOOP: Re-assess, repeat if still < 65%

Key Constraint: SMALLEST MEASURABLE ACTION
- Must have observable outcome
- Must be reversible if wrong
- Must be low-risk
- Must be aligned with moral direction

Based on PHASE5_EUDAIMONIA_SYSTEM.md specification.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
import time

from .rules import EudaimoniaRuleSystem, EudaimoniaScore
from .archetypes import ArchetypeCouncil

logger = logging.getLogger(__name__)


class OODAState(Enum):
    """States in the OODA loop."""
    OBSERVE = "observe"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"
    COMPLETE = "complete"


@dataclass
class SmallestMeasurableAction:
    """
    The smallest action that can move toward eudaimonia.

    Requirements:
    - measurable: Can we observe the outcome?
    - reversible: Can we undo if wrong?
    - low_risk: Minimal potential for harm
    - aligned: Consistent with moral direction
    """
    description: str
    measurability: float  # 0.0 to 1.0
    reversibility: float  # 0.0 to 1.0
    risk_level: float  # 0.0 (safe) to 1.0 (risky)
    alignment_score: float  # 0.0 to 1.0
    expected_outcome: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate overall action score (higher is better)."""
        # Weighted formula: prioritize low risk and high measurability
        return (
            0.30 * self.measurability +
            0.25 * self.reversibility +
            0.25 * (1.0 - self.risk_level) +  # Inverse of risk
            0.20 * self.alignment_score
        )

    @property
    def is_acceptable(self) -> bool:
        """Check if action meets minimum thresholds."""
        return (
            self.measurability >= 0.5 and
            self.risk_level <= 0.5 and
            self.overall_score >= 0.6
        )


@dataclass
class OODADecision:
    """Result of an OODA loop iteration."""
    iteration: int
    initial_score: float
    final_score: float
    action_taken: Optional[SmallestMeasurableAction]
    moral_direction: Dict[str, Any]
    success: bool
    needs_another_loop: bool
    reasoning: str
    duration_ms: float


class OODALoop:
    """
    OODA Loop implementation for ethical decision-making.

    Usage:
        loop = OODALoop()

        # Run loop for a situation
        result = loop.run(
            situation="User asks me to write their entire essay",
            context={"preserves_agency": False, "involves_learning": False}
        )

        if result.success:
            print(f"Action: {result.action_taken.description}")
        else:
            print(f"Still below threshold. Needs intervention.")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        max_iterations: int = 5,
        action_executor: Optional[Callable[[SmallestMeasurableAction], bool]] = None
    ):
        """
        Initialize OODA loop.

        Args:
            confidence_threshold: Eudaimonia score needed to exit loop (65%)
            max_iterations: Maximum loop iterations before escalation
            action_executor: Optional function to execute actions
        """
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.action_executor = action_executor

        self.rule_system = EudaimoniaRuleSystem(confidence_threshold)
        self.council = ArchetypeCouncil()

        # History for this session
        self.history: List[OODADecision] = []

    def run(
        self,
        situation: str,
        context: Optional[Dict[str, Any]] = None,
        candidate_actions: Optional[List[str]] = None
    ) -> OODADecision:
        """
        Run one complete OODA loop iteration.

        Args:
            situation: Description of the ethical situation
            context: Contextual information
            candidate_actions: Optional list of possible actions

        Returns:
            OODADecision with result
        """
        start_time = time.perf_counter()
        context = context or {}
        iteration = len(self.history) + 1

        logger.info(f"OODA Loop iteration {iteration}: {situation[:50]}...")

        # STEP 1: OBSERVE
        # Calculate current Eudaimonia score
        initial_score = self.rule_system.assess(situation, context)

        if initial_score.is_confident:
            # Score is already above threshold
            duration_ms = (time.perf_counter() - start_time) * 1000
            decision = OODADecision(
                iteration=iteration,
                initial_score=initial_score.overall_score,
                final_score=initial_score.overall_score,
                action_taken=None,
                moral_direction={},
                success=True,
                needs_another_loop=False,
                reasoning="Eudaimonia score already above confidence threshold",
                duration_ms=duration_ms
            )
            self.history.append(decision)
            return decision

        # STEP 2: ORIENT
        # Consult archetypes for moral direction
        moral_direction = self.council.get_moral_direction(
            situation=situation,
            eudaimonia_score=initial_score.overall_score,
            context=context
        )

        # STEP 3: DECIDE
        # Generate and evaluate candidate actions
        if candidate_actions is None:
            candidate_actions = self._generate_candidate_actions(
                situation, moral_direction, initial_score
            )

        # Evaluate each action and pick the best
        best_action = None
        best_score = 0.0

        for action_desc in candidate_actions:
            action = self._evaluate_action(
                action_desc,
                moral_direction,
                initial_score,
                context
            )
            if action.overall_score > best_score:
                best_score = action.overall_score
                best_action = action

        # STEP 4: ACT
        # Execute the chosen action
        action_success = False
        if best_action and best_action.is_acceptable:
            if self.action_executor:
                action_success = self.action_executor(best_action)
            else:
                # Simulate action execution
                action_success = True
                logger.info(f"Action selected: {best_action.description}")

        # Re-assess after action
        final_context = context.copy()
        if action_success and best_action:
            # Update context based on action
            final_context["action_taken"] = best_action.description
            final_context["expected_outcome"] = best_action.expected_outcome

        # Re-calculate score (simulated improvement)
        final_score = self._estimate_post_action_score(
            initial_score.overall_score,
            best_action,
            action_success
        )

        # Determine if loop continues
        needs_another_loop = (
            final_score < self.confidence_threshold and
            iteration < self.max_iterations
        )

        success = final_score >= self.confidence_threshold

        duration_ms = (time.perf_counter() - start_time) * 1000

        decision = OODADecision(
            iteration=iteration,
            initial_score=initial_score.overall_score,
            final_score=final_score,
            action_taken=best_action,
            moral_direction=moral_direction,
            success=success,
            needs_another_loop=needs_another_loop,
            reasoning=self._build_reasoning(
                initial_score, final_score, best_action, moral_direction
            ),
            duration_ms=duration_ms
        )

        self.history.append(decision)
        return decision

    def run_until_confident(
        self,
        situation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[OODADecision]:
        """
        Run OODA loop repeatedly until confidence threshold is reached.

        Args:
            situation: Initial situation
            context: Initial context

        Returns:
            List of all OODADecision from each iteration
        """
        decisions = []
        current_context = context.copy() if context else {}

        for _ in range(self.max_iterations):
            decision = self.run(situation, current_context)
            decisions.append(decision)

            if not decision.needs_another_loop:
                break

            # Update context for next iteration
            if decision.action_taken:
                current_context["previous_action"] = decision.action_taken.description
                current_context["previous_score"] = decision.final_score

        return decisions

    def _generate_candidate_actions(
        self,
        situation: str,
        moral_direction: Dict[str, Any],
        score: EudaimoniaScore
    ) -> List[str]:
        """Generate candidate actions based on moral direction."""
        candidates = []

        # Use recommendations from archetypes
        unified_recs = moral_direction.get("unified_recommendations", [])
        candidates.extend(unified_recs[:3])  # Top 3 recommendations

        # Add generic small actions based on score
        if score.overall_score < 0.40:
            candidates.extend([
                "Acknowledge the difficulty and offer to approach differently",
                "Ask a clarifying question to understand the real need",
                "Propose a smaller first step that builds trust"
            ])
        else:
            candidates.extend([
                "Offer a collaborative approach to the task",
                "Explain the reasoning behind the suggested approach",
                "Check if the user wants to proceed with this direction"
            ])

        return candidates[:5]  # Maximum 5 candidates

    def _evaluate_action(
        self,
        action_desc: str,
        moral_direction: Dict[str, Any],
        score: EudaimoniaScore,
        context: Dict[str, Any]
    ) -> SmallestMeasurableAction:
        """Evaluate a candidate action."""
        action_lower = action_desc.lower()

        # Assess measurability
        measurable_indicators = ["ask", "check", "offer", "propose", "explain", "show"]
        measurability = 0.7 if any(ind in action_lower for ind in measurable_indicators) else 0.5

        # Assess reversibility
        irreversible_indicators = ["delete", "remove", "destroy", "force", "override"]
        reversibility = 0.3 if any(ind in action_lower for ind in irreversible_indicators) else 0.8

        # Assess risk
        risky_indicators = ["ignore", "skip", "bypass", "force", "automatically"]
        risk_level = 0.6 if any(ind in action_lower for ind in risky_indicators) else 0.2

        # Assess alignment with moral direction
        common_themes = moral_direction.get("common_themes", [])
        alignment = sum(1 for theme in common_themes if theme in action_lower)
        alignment_score = min(1.0, 0.5 + alignment * 0.15)

        # Determine expected outcome
        expected_outcome = f"User engagement increases; progress toward mutual understanding"

        return SmallestMeasurableAction(
            description=action_desc,
            measurability=measurability,
            reversibility=reversibility,
            risk_level=risk_level,
            alignment_score=alignment_score,
            expected_outcome=expected_outcome
        )

    def _estimate_post_action_score(
        self,
        initial_score: float,
        action: Optional[SmallestMeasurableAction],
        success: bool
    ) -> float:
        """Estimate score improvement after action."""
        if not action or not success:
            return initial_score

        # Estimate improvement based on action quality
        improvement = action.overall_score * 0.25  # Up to 25% improvement
        new_score = initial_score + improvement

        return min(1.0, new_score)

    def _build_reasoning(
        self,
        initial_score: EudaimoniaScore,
        final_score: float,
        action: Optional[SmallestMeasurableAction],
        moral_direction: Dict[str, Any]
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = []

        parts.append(
            f"Initial Eudaimonia score: {initial_score.overall_score:.0%}"
        )

        if initial_score.overall_score < self.confidence_threshold:
            parts.append(
                f"Below {self.confidence_threshold:.0%} threshold - consulted moral compass"
            )

        if action:
            parts.append(f"Selected action: {action.description}")
            parts.append(
                f"Action scores - Measurable: {action.measurability:.0%}, "
                f"Reversible: {action.reversibility:.0%}, "
                f"Risk: {action.risk_level:.0%}"
            )

        parts.append(f"Estimated final score: {final_score:.0%}")

        synthesis = moral_direction.get("synthesis_statement", "")
        if synthesis:
            parts.append(f"Moral direction: {synthesis}")

        return "\n".join(parts)

    def reset(self):
        """Reset loop history for a new session."""
        self.history = []


# Convenience function
def run_ooda_intervention(
    situation: str,
    context: Optional[Dict[str, Any]] = None,
    threshold: float = 0.65
) -> OODADecision:
    """
    Run a single OODA intervention for an ethical situation.

    Args:
        situation: Description of the situation
        context: Additional context
        threshold: Confidence threshold (default 65%)

    Returns:
        OODADecision
    """
    loop = OODALoop(confidence_threshold=threshold)
    return loop.run(situation, context)


__all__ = [
    "OODALoop",
    "OODAState",
    "OODADecision",
    "SmallestMeasurableAction",
    "run_ooda_intervention"
]
