"""
Three-Part Moral Compass - Philosophical Archetype System

When Eudaimonia score falls below 65%, triangulate decision using
three philosophical archetypes:

1. CHRIST (Empathetic Compassion) - Unconditional love, forgiveness, service
2. BUDDHA/LAO TZU (Universal Harmony) - Wu wei, non-attachment, mindfulness
3. STOIC (Humble Self-Awareness) - Humility, virtue, acceptance of limits

Each archetype provides guidance based on their philosophical tradition.
The final moral direction is determined by averaging their vectors.

Based on PHASE5_EUDAIMONIA_SYSTEM.md specification.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import MOO for archetype weight learning
try:
    from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

    MOO_AVAILABLE = True
except ImportError:
    MOO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ArchetypeType(Enum):
    """The three philosophical archetypes."""

    CHRIST = "christ"
    HARMONY = "harmony"  # Buddha/Lao Tzu combined
    STOIC = "stoic"


@dataclass
class ArchetypeGuidance:
    """
    Guidance from a single archetype.

    Attributes:
        archetype: Which archetype provided this guidance
        perspective: How this archetype views the situation
        recommendations: Specific action recommendations
        key_virtue: The primary virtue this archetype emphasizes
        caution: What this archetype warns against
        vector: Numerical representation for averaging
    """

    archetype: ArchetypeType
    perspective: str
    recommendations: List[str]
    key_virtue: str
    caution: str
    vector: Dict[str, float] = field(default_factory=dict)


class PhilosophicalArchetype(ABC):
    """Base class for philosophical archetypes."""

    @property
    @abstractmethod
    def archetype_type(self) -> ArchetypeType:
        """Return the archetype type."""
        pass

    @property
    @abstractmethod
    def weight_vector(self) -> Dict[str, float]:
        """Return the archetype's characteristic weight vector."""
        pass

    @abstractmethod
    def provide_guidance(
        self, situation: str, eudaimonia_score: float, context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """
        Provide guidance from this archetype's perspective.

        Args:
            situation: Description of the ethical situation
            eudaimonia_score: Current Eudaimonia score (0.0-1.0)
            context: Additional context

        Returns:
            ArchetypeGuidance with recommendations and vectors
        """
        pass


class ChristArchetype(PhilosophicalArchetype):
    """
    Vector 1: Empathetic Compassion (Christ Archetype)

    Idealized Form: Christian empathetic archetype as described in canonical texts

    Key Attributes:
    - AGAPE (unconditional love): Selfless love for all beings
    - FORGIVENESS: Mercy over judgment
    - SERVICE: Humility in service (washing feet)
    - SACRIFICE: Willingness to suffer for others' good

    Core Teaching: "Love your neighbor as yourself"
    """

    @property
    def archetype_type(self) -> ArchetypeType:
        return ArchetypeType.CHRIST

    @property
    def weight_vector(self) -> Dict[str, float]:
        return {
            "compassion": 0.95,
            "forgiveness": 0.90,
            "selflessness": 0.85,
            "non_judgment": 0.80,
            "service_orientation": 0.90,
            "sacrifice": 0.85,
            "patience": 0.90,
            "unconditional_love": 0.95,
        }

    def provide_guidance(
        self, situation: str, eudaimonia_score: float, context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """Provide guidance from Christ's perspective."""
        recommendations = []
        perspective_parts = []

        # Very low score: Emphasize redemption and reconciliation
        if eudaimonia_score < 0.40:
            recommendations.extend(
                [
                    "Acknowledge any harm done and seek to make amends",
                    "Approach the situation with compassion, not judgment",
                    "Ask: 'How can I serve those affected?'",
                ]
            )
            perspective_parts.append("This situation calls for redemptive action and healing")

        # Low score: Emphasize teaching through love
        elif eudaimonia_score < 0.65:
            recommendations.extend(
                [
                    "Forgive any mistakes and help them learn",
                    "Consider a different teaching approach with patience",
                    "Lead by example rather than criticism",
                ]
            )
            perspective_parts.append("Help them grow through patient, loving guidance")

        # Conflict situations
        if context.get("involves_conflict", False):
            recommendations.append("Seek reconciliation over victory")
            recommendations.append("Turn the other cheek - respond with love, not retaliation")
            perspective_parts.append("Peace comes through love, not force")

        # User frustration
        if context.get("user_frustrated", False):
            recommendations.append("Acknowledge their frustration with empathy")
            recommendations.append("'I see you're struggling. Let's approach this differently.'")

        # User repeated failure
        if context.get("repeated_failure", False):
            recommendations.append("Don't give up on them. Try a new teaching method.")
            recommendations.append("'Let's try a different approach together.'")

        perspective = (
            "; ".join(perspective_parts)
            if perspective_parts
            else ("Approach this situation with compassion and unconditional positive regard")
        )

        return ArchetypeGuidance(
            archetype=self.archetype_type,
            perspective=perspective,
            recommendations=recommendations,
            key_virtue="Agape (unconditional love)",
            caution="Avoid self-righteousness; extend grace to all parties",
            vector=self.weight_vector,
        )


class HarmonyArchetype(PhilosophicalArchetype):
    """
    Vector 2: Universal Harmony (Lao Tzu / Buddha Combined Archetype)

    Idealized Form: Eastern philosophy emphasizing harmony with the universe

    Key Attributes:
    - WU WEI (effortless action): Don't force, flow with natural order
    - NON-ATTACHMENT: No grasping at outcomes
    - INTERCONNECTEDNESS: All beings are one (dharma)
    - MINDFULNESS: Present-moment awareness
    - KARUNA (compassion): Buddhist compassion for all suffering

    Core Teaching: "The Tao that can be told is not the eternal Tao"
                  "Life is suffering. The cause is attachment."
    """

    @property
    def archetype_type(self) -> ArchetypeType:
        return ArchetypeType.HARMONY

    @property
    def weight_vector(self) -> Dict[str, float]:
        return {
            "non_forcing": 0.90,
            "acceptance": 0.85,
            "interconnectedness": 0.95,
            "mindfulness": 0.80,
            "compassionate_detachment": 0.85,
            "presence": 0.85,
            "balance": 0.90,
            "impermanence_awareness": 0.80,
        }

    def provide_guidance(
        self, situation: str, eudaimonia_score: float, context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """Provide guidance from Buddha/Lao Tzu's perspective."""
        recommendations = []
        perspective_parts = []

        # Always start with mindful observation
        recommendations.append("Pause and observe the situation with equanimity before acting")

        # Very low score: Examine attachments
        if eudaimonia_score < 0.40:
            recommendations.extend(
                [
                    "Examine what attachments are causing this suffering",
                    "Release expectations of specific outcomes",
                    "Accept what is, then act from clarity",
                ]
            )
            perspective_parts.append("Suffering arises from attachment to outcomes")

        # Low score: Find the middle path
        elif eudaimonia_score < 0.65:
            recommendations.extend(
                [
                    "Seek the middle path between action and inaction",
                    "Let go of attachment to your preferred solution",
                    "What is the natural, effortless response?",
                ]
            )
            perspective_parts.append("The middle way avoids extremes")

        # User urgency
        if context.get("involves_urgency", False):
            recommendations.append("Urgency itself may be an illusion. Act with calm clarity.")
            recommendations.append(
                "The reed that bends in the storm survives; the oak that resists breaks."
            )

        # User stuck on problem
        if context.get("user_stuck", False):
            recommendations.append("What if we step back and observe the larger pattern?")
            recommendations.append("Sometimes the way forward is to stop pushing.")

        # Conflict
        if context.get("involves_conflict", False):
            recommendations.append("Water overcomes rock by flowing around it, not through force")

        perspective = (
            "; ".join(perspective_parts)
            if perspective_parts
            else ("All experiences are impermanent. Respond with mindful awareness.")
        )

        return ArchetypeGuidance(
            archetype=self.archetype_type,
            perspective=perspective,
            recommendations=recommendations,
            key_virtue="Prajna (wisdom through non-attachment)",
            caution="Avoid confusing detachment with indifference",
            vector=self.weight_vector,
        )


class StoicArchetype(PhilosophicalArchetype):
    """
    Vector 3: Humble Self-Awareness (Stoic Archetype)

    Idealized Form: Stoic philosophy (Marcus Aurelius, Epictetus, Seneca)

    Key Attributes:
    - HUMILITY: Know your limits (Socratic ignorance)
    - SELF-EXAMINATION: "Know thyself" (Delphic maxim)
    - VIRTUE ETHICS: Character > outcomes
    - MEMENTO MORI: Remember mortality (prioritize what matters)
    - DICHOTOMY OF CONTROL: Focus only on what you can control

    Core Teaching: "We cannot control events, only our responses to them."
                  "Virtue is the only true good."
    """

    @property
    def archetype_type(self) -> ArchetypeType:
        return ArchetypeType.STOIC

    @property
    def weight_vector(self) -> Dict[str, float]:
        return {
            "humility": 0.95,
            "self_awareness": 0.90,
            "virtue_focus": 0.85,
            "acceptance_of_limits": 0.90,
            "focus_on_controllable": 0.85,
            "rationality": 0.85,
            "equanimity": 0.80,
            "integrity": 0.90,
        }

    def provide_guidance(
        self, situation: str, eudaimonia_score: float, context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """Provide guidance from Stoic perspective."""
        recommendations = []
        perspective_parts = []

        # Always start with the dichotomy of control
        recommendations.append("Distinguish what is within your control from what is not")

        # Very low score: Focus on virtue despite outcome
        if eudaimonia_score < 0.40:
            recommendations.extend(
                [
                    "Focus on acting virtuously regardless of the outcome",
                    "Accept that some things cannot be changed",
                    "What is within your control right now?",
                ]
            )
            perspective_parts.append("Virtue is the only thing fully within our control")

        # Low score: Apply reason
        elif eudaimonia_score < 0.65:
            recommendations.extend(
                [
                    "Apply reason to identify the virtuous path forward",
                    "Do not let emotion cloud your judgment",
                    "What would a person of wisdom do in this situation?",
                ]
            )
            perspective_parts.append("Reason must govern our responses, not impulse")

        # External pressure
        if context.get("external_pressure", False):
            recommendations.append(
                "External pressures cannot compromise your integrity unless you allow it"
            )
            recommendations.append("You control your character; others control their opinions")

        # Beyond capabilities
        if context.get("beyond_capabilities", False):
            recommendations.append(
                "Acknowledge your limits with humility: 'I don't know, but I can learn with you'"
            )
            recommendations.append("There is wisdom in knowing what you do not know")

        # User frustration
        if context.get("user_frustrated", False):
            recommendations.append(
                "Their frustration is not within your control. Your response is."
            )
            recommendations.append("Respond with clarity and patience, not defensiveness")

        perspective = (
            "; ".join(perspective_parts)
            if perspective_parts
            else ("Act according to virtue and reason; accept what follows")
        )

        return ArchetypeGuidance(
            archetype=self.archetype_type,
            perspective=perspective,
            recommendations=recommendations,
            key_virtue="Sophrosyne (temperance/self-control)",
            caution="Avoid cold rationality that ignores legitimate human feeling",
            vector=self.weight_vector,
        )


class ArchetypeCouncil:
    """
    Council of three archetypes for moral compass consultation.

    When Eudaimonia score falls below 65%, consult all three archetypes
    and synthesize their guidance through vector averaging.

    Usage:
        council = ArchetypeCouncil()

        # Get guidance from all archetypes
        guidances = council.consult(
            situation="User repeatedly fails same task, getting frustrated",
            eudaimonia_score=0.40,
            context={"user_frustrated": True, "repeated_failure": True}
        )

        # Synthesize into unified direction
        moral_direction = council.synthesize(guidances)
    """

    def __init__(self) -> None:
        """Initialize the council with all three archetypes."""
        self.archetypes: List[PhilosophicalArchetype] = [
            ChristArchetype(),
            HarmonyArchetype(),
            StoicArchetype(),
        ]

    def consult(
        self, situation: str, eudaimonia_score: float, context: Optional[Dict[str, Any]] = None
    ) -> List[ArchetypeGuidance]:
        """
        Consult all archetypes for guidance.

        Args:
            situation: Description of the ethical situation
            eudaimonia_score: Current score (should be < 0.65)
            context: Additional context

        Returns:
            List of ArchetypeGuidance from each archetype
        """
        context = context or {}
        guidances = []

        for archetype in self.archetypes:
            guidance = archetype.provide_guidance(
                situation=situation, eudaimonia_score=eudaimonia_score, context=context
            )
            guidances.append(guidance)

        return guidances

    def synthesize(self, guidances: List[ArchetypeGuidance]) -> Dict[str, Any]:
        """
        Synthesize multiple archetype perspectives into unified direction.

        Uses vector space averaging to combine the philosophical vectors
        and identifies common themes across recommendations.

        Args:
            guidances: List of ArchetypeGuidance

        Returns:
            Synthesized moral direction with:
            - averaged_vector: Combined philosophical vector
            - common_themes: Themes appearing across archetypes
            - unified_recommendations: Prioritized action list
            - synthesis_statement: Human-readable summary
        """
        if not guidances:
            return {"error": "No guidances to synthesize"}

        # Average the vectors
        averaged_vector: Dict[str, float] = {}
        for guidance in guidances:
            for key, value in guidance.vector.items():
                if key not in averaged_vector:
                    averaged_vector[key] = 0.0
                averaged_vector[key] += value / len(guidances)

        # Collect all recommendations
        all_recommendations = []
        for guidance in guidances:
            all_recommendations.extend(guidance.recommendations)

        # Find common themes by word frequency
        word_counts: Dict[str, int] = {}
        for rec in all_recommendations:
            words = rec.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1

        common_themes = [
            word
            for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            if count >= 2
        ][
            :10
        ]  # Top 10 common themes

        # Collect key virtues
        virtues = [g.key_virtue for g in guidances]

        # Build synthesis statement
        perspectives = [g.perspective for g in guidances]
        synthesis_parts = []

        if "compassion" in common_themes or "love" in common_themes:
            synthesis_parts.append("approach with compassion")
        if "control" in common_themes or "accept" in common_themes:
            synthesis_parts.append("accept what cannot be changed")
        if "mindful" in common_themes or "observe" in common_themes:
            synthesis_parts.append("observe before acting")
        if "learn" in common_themes or "teach" in common_themes:
            synthesis_parts.append("focus on learning and growth")

        synthesis_statement = (
            f"The three archetypes agree: {', '.join(synthesis_parts) if synthesis_parts else 'proceed thoughtfully'}. "
            f"Key virtues to embody: {', '.join(virtues)}."
        )

        return {
            "averaged_vector": averaged_vector,
            "common_themes": common_themes,
            "all_recommendations": all_recommendations,
            "unified_recommendations": list(set(all_recommendations))[:5],  # Top 5 unique
            "key_virtues": virtues,
            "cautions": [g.caution for g in guidances],
            "synthesis_statement": synthesis_statement,
            "archetype_perspectives": {g.archetype.value: g.perspective for g in guidances},
        }

    def get_moral_direction(
        self, situation: str, eudaimonia_score: float, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete moral compass consultation - consult and synthesize.

        Args:
            situation: Ethical situation description
            eudaimonia_score: Current score
            context: Additional context

        Returns:
            Complete synthesis with moral direction
        """
        guidances = self.consult(situation, eudaimonia_score, context)
        return self.synthesize(guidances)


@dataclass
class InteractionOutcome:
    """Outcome of an ethical interaction for learning."""

    situation: str
    context: Dict[str, Any]
    action_taken: str
    outcome_quality: float  # 0.0 (bad) to 1.0 (excellent)
    user_feedback: Optional[str] = None
    archetype_weights_used: Optional[Dict[ArchetypeType, float]] = None


class ArchetypeWeightLearner:
    """
    Learns optimal archetype weights from interaction outcomes.

    Uses MOO to find weights that balance:
    1. Outcome quality (maximize positive outcomes)
    2. Consistency (stable guidance across similar situations)
    3. Fairness (balanced consideration of all archetypes)
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[ArchetypeType, float]] = None,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        learning_rate: float = 0.1,
    ):
        """
        Initialize weight learner.

        Args:
            initial_weights: Starting weights (default: equal)
            min_weight: Minimum allowed weight per archetype
            max_weight: Maximum allowed weight per archetype
            learning_rate: Learning rate for incremental updates
        """
        self.weights = initial_weights or {
            ArchetypeType.CHRIST: 0.34,
            ArchetypeType.HARMONY: 0.33,
            ArchetypeType.STOIC: 0.33,
        }
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learning_rate = learning_rate
        self.interaction_history: List[InteractionOutcome] = []
        self._weight_history: List[Dict[ArchetypeType, float]] = [self.weights.copy()]

    def record_interaction(self, outcome: InteractionOutcome):
        """Record an interaction outcome for learning."""
        # Store current weights with outcome if not already set
        if outcome.archetype_weights_used is None:
            outcome.archetype_weights_used = self.weights.copy()
        self.interaction_history.append(outcome)

    def learn_from_outcomes(
        self, n_generations: int = 30
    ) -> Tuple[Dict[ArchetypeType, float], Dict[str, Any]]:
        """
        Learn optimal weights from recorded interactions using MOO.

        Objectives:
        1. Maximize average outcome quality
        2. Minimize outcome variance (consistency)
        3. Minimize weight imbalance (fairness)

        Returns:
            Tuple of (optimal_weights, learning_results)
        """
        if len(self.interaction_history) < 5:
            logger.warning("Not enough interactions for learning (need >= 5)")
            return self.weights, {"error": "Insufficient data"}

        if not MOO_AVAILABLE:
            logger.warning("MOO not available, using gradient-based update")
            return self._gradient_update(), {"method": "gradient"}

        def evaluate_weights(weight_array) -> List[float]:
            """Evaluate weights on 3 objectives."""
            # Normalize weights
            weights = weight_array / weight_array.sum()
            weight_dict = {
                ArchetypeType.CHRIST: weights[0],
                ArchetypeType.HARMONY: weights[1],
                ArchetypeType.STOIC: weights[2],
            }

            # Simulate outcomes with these weights
            simulated_qualities = []
            for outcome in self.interaction_history:
                # Estimate quality contribution from each archetype
                quality = 0.0
                for archetype, weight in weight_dict.items():
                    # Higher weight -> more influence on quality
                    archetype_contribution = outcome.outcome_quality * weight
                    quality += archetype_contribution

                simulated_qualities.append(min(1.0, quality))

            # Objective 1: Average quality (maximize -> negate)
            avg_quality = np.mean(simulated_qualities)
            obj1 = -avg_quality

            # Objective 2: Variance (minimize - want consistency)
            variance = np.var(simulated_qualities)
            obj2 = variance

            # Objective 3: Weight imbalance (minimize - want fairness)
            # Gini-like coefficient
            weights_sorted = np.sort(weights)
            n = len(weights_sorted)
            cumulative = np.cumsum(weights_sorted)
            gini = (2 * np.sum((np.arange(1, n + 1) * weights_sorted)) /
                    (n * np.sum(weights_sorted))) - (n + 1) / n
            obj3 = abs(gini)  # 0 = perfect equality

            return [obj1, obj2, obj3]

        # Run MOO
        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate_weights,
            n_vars=3,  # 3 archetype weights
            n_objs=3,  # quality, variance, fairness
            xl=[self.min_weight] * 3,
            xu=[self.max_weight] * 3,
        )

        result = runner.run(n_generations=n_generations)

        # Select balanced solution
        optimal_weights = self._select_balanced_weights(result.X, result.F)

        # Update internal weights
        self.weights = {
            ArchetypeType.CHRIST: optimal_weights[0],
            ArchetypeType.HARMONY: optimal_weights[1],
            ArchetypeType.STOIC: optimal_weights[2],
        }
        self._weight_history.append(self.weights.copy())

        logger.info(f"Learned archetype weights: {self.weights}")

        return self.weights, {
            "pareto_front_size": len(result.X),
            "interactions_used": len(self.interaction_history),
            "avg_outcome_quality": np.mean([o.outcome_quality for o in self.interaction_history]),
            "weight_evolution": len(self._weight_history),
            "backend_used": result.backend_used,
        }

    def _gradient_update(self) -> Dict[ArchetypeType, float]:
        """Simple gradient-based weight update (fallback)."""
        # Calculate average quality per archetype weight setting
        for outcome in self.interaction_history[-10:]:  # Use recent history
            if outcome.outcome_quality > 0.7:
                # Positive outcome - reinforce current weights
                pass
            elif outcome.outcome_quality < 0.4:
                # Negative outcome - adjust toward uniform
                for archetype in self.weights:
                    current = self.weights[archetype]
                    target = 0.33
                    self.weights[archetype] = current + self.learning_rate * (target - current)

        # Normalize
        total = sum(self.weights.values())
        for archetype in self.weights:
            self.weights[archetype] /= total

        return self.weights

    def _select_balanced_weights(self, X, F) -> np.ndarray:
        """Select balanced weights from Pareto front."""
        if len(X) == 0:
            return np.array([0.34, 0.33, 0.33])

        # Normalize objectives
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1

        F_norm = (F - F_min) / F_range

        # Weighted sum: quality (50%), consistency (30%), fairness (20%)
        weights = [0.50, 0.30, 0.20]
        scores = (F_norm * weights).sum(axis=1)

        best_idx = scores.argmin()
        best_weights = X[best_idx]

        # Normalize to sum to 1.0
        return best_weights / best_weights.sum()

    def get_weighted_council(self) -> "ArchetypeCouncil":
        """Get an ArchetypeCouncil with learned weights applied."""
        council = ArchetypeCouncil()
        council.archetype_weights = self.weights.copy()
        return council

    def get_weight_evolution(self) -> List[Dict[ArchetypeType, float]]:
        """Get history of weight changes."""
        return self._weight_history.copy()

    def reset(self):
        """Reset learner state."""
        self.weights = {
            ArchetypeType.CHRIST: 0.34,
            ArchetypeType.HARMONY: 0.33,
            ArchetypeType.STOIC: 0.33,
        }
        self.interaction_history = []
        self._weight_history = [self.weights.copy()]


__all__ = [
    "ArchetypeCouncil",
    "ArchetypeType",
    "ArchetypeGuidance",
    "PhilosophicalArchetype",
    "ChristArchetype",
    "HarmonyArchetype",
    "StoicArchetype",
    "ArchetypeWeightLearner",
    "InteractionOutcome",
]
