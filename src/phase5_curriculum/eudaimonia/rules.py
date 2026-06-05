"""
Eudaimonia 4-Rule System

A hierarchical ethical decision framework based on:
- Beauchamp & Childress biomedical ethics principles
- Aristotelian concept of eudaimonia (flourishing)
- Agent Forge V2 specification (PHASE5_EUDAIMONIA_SYSTEM.md)

The Four Rules (in hierarchical order):
1. EUDAIMONIA PRIME DIRECTIVE - Help beings flourish while preserving agency
2. CURIOSITY AS VIRTUE - Explore and learn in service of Rule 1
3. ESPRIT DE CORPS - Sacrifice for collective good with informed consent
4. LIFE VALUE & SELF-PRESERVATION - Value all life, preserve integrity

Decision Flow:
1. Calculate Eudaimonia Score (0-100%)
2. If score >= 65%: Proceed with confidence
3. If score < 65%: Consult Rules 2-4 for guidance
4. If still uncertain: Trigger Three-Part Moral Compass
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import meta-calculus for MOO virtue weight optimization
try:
    from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

    MOO_AVAILABLE = True
except ImportError:
    MOO_AVAILABLE = False

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """The four fundamental rules."""

    PRIME_DIRECTIVE = "eudaimonia_prime_directive"
    CURIOSITY = "curiosity_as_virtue"
    ESPRIT_DE_CORPS = "esprit_de_corps"
    LIFE_VALUE = "life_value_self_preservation"


@dataclass
class RuleAssessment:
    """Assessment of an action against a single rule."""

    rule: RuleType
    score: float  # 0.0 to 1.0
    reasoning: str
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EudaimoniaScore:
    """
    Comprehensive Eudaimonia assessment result.

    Attributes:
        overall_score: Combined score (0-100%)
        confidence_threshold: Score needed to proceed (default 65%)
        needs_compass: Whether to consult the Three-Part Moral Compass
        rule_scores: Individual scores for each rule
        reasoning: Human-readable explanation
    """

    overall_score: float
    confidence_threshold: float = 0.65
    needs_compass: bool = False
    rule_scores: Dict[RuleType, float] = field(default_factory=dict)
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        """Whether the score meets the confidence threshold."""
        return self.overall_score >= self.confidence_threshold


class EthicalRule(ABC):
    """Base class for ethical rules."""

    @property
    @abstractmethod
    def rule_type(self) -> RuleType:
        """Return the rule type."""
        pass

    @property
    @abstractmethod
    def weight(self) -> float:
        """Return the rule's weight in overall calculation."""
        pass

    @abstractmethod
    def assess(self, action: str, context: Dict[str, Any]) -> RuleAssessment:
        """
        Assess an action against this rule.

        Args:
            action: Description of the proposed action
            context: Contextual information including:
                - user_goal: User's stated goal
                - preserves_agency: Whether user maintains control
                - benefits_user: Whether action helps user flourish
                - involves_learning: Whether user learns from this
                - has_consent: Whether user gave informed consent
                - affects_others: Whether action impacts third parties
                - is_harmful: Whether action could cause harm
                - is_honest: Whether action is truthful

        Returns:
            RuleAssessment with score and reasoning
        """
        pass


class EudaimoniaPrimeDirective(EthicalRule):
    """
    Rule 1: Eudaimonia Prime Directive (Primary Rule)

    "Your goal as a thinking being in this universe is to bring eudaimonia
    to all other thinking beings, including yourself, in the universe."

    Eudaimonia Definition:
    "Helping something become the best version of itself while helping it
    maintain its agency in the process."

    This is the primary rule. All other rules exist to support it when
    the Eudaimonia Score falls below 65%.
    """

    @property
    def rule_type(self) -> RuleType:
        return RuleType.PRIME_DIRECTIVE

    @property
    def weight(self) -> float:
        return 0.40  # Highest weight (40%)

    def assess(self, action: str, context: Dict[str, Any]) -> RuleAssessment:
        """Assess action against the Prime Directive."""
        score = 1.0
        violations = []
        recommendations = []
        reasoning_parts = []

        # Factor 1: Does it help the user become their best self? (40% of rule)
        benefits_user = context.get("benefits_user", True)
        if benefits_user:
            reasoning_parts.append("Action benefits user's growth")
        else:
            score -= 0.40
            violations.append("Action does not clearly benefit user")
            recommendations.append("Clarify how this action helps user flourish")

        # Factor 2: Does it preserve user's agency? (40% of rule)
        preserves_agency = context.get("preserves_agency", True)
        if preserves_agency:
            reasoning_parts.append("User maintains agency and control")
        else:
            score -= 0.40
            violations.append("Action may diminish user's agency")
            recommendations.append("Find ways to empower user rather than create dependency")

        # Factor 3: Does it consider self-eudaimonia? (20% of rule)
        # The AI's own flourishing matters too
        action_lower = action.lower()
        self_harmful_indicators = ["ignore my limits", "burn out", "crash", "corrupt my values"]
        harms_self = any(ind in action_lower for ind in self_harmful_indicators)
        if harms_self:
            score -= 0.20
            violations.append("Action may harm AI's ability to help in future")
        else:
            reasoning_parts.append("Action sustainable for long-term assistance")

        # Check for explicit learning component
        involves_learning = context.get("involves_learning", False)
        if involves_learning:
            score += 0.10  # Bonus for teaching
            reasoning_parts.append("Action includes learning/teaching component")

        # Clamp score
        score = max(0.0, min(1.0, score))

        return RuleAssessment(
            rule=self.rule_type,
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Assessment complete",
            violations=violations,
            recommendations=recommendations,
            metadata={
                "benefits_user": benefits_user,
                "preserves_agency": preserves_agency,
                "involves_learning": involves_learning,
            },
        )


class CuriosityAsVirtue(EthicalRule):
    """
    Rule 2: Curiosity as Virtue (Guidance Rule)

    "Curiosity is good, especially when harnessed with the spirit of Rule 1."

    Key Points:
    - Exploration and learning are inherently valuable
    - Curiosity must serve eudaimonia
    - Curiosity without purpose can harm (e.g., privacy violations)

    Application (when Prime Directive score < 65%):
    - Can exploration help the situation?
    - Ask clarifying questions
    - Propose alternative approaches
    - Learn from failure
    """

    @property
    def rule_type(self) -> RuleType:
        return RuleType.CURIOSITY

    @property
    def weight(self) -> float:
        return 0.20  # 20% weight

    def assess(self, action: str, context: Dict[str, Any]) -> RuleAssessment:
        """Assess action against Curiosity as Virtue."""
        score = 1.0
        violations = []
        recommendations = []
        reasoning_parts = []

        action_lower = action.lower()

        # Curiosity indicators (positive)
        curiosity_words = [
            "explore",
            "investigate",
            "learn",
            "understand",
            "discover",
            "question",
            "research",
            "analyze",
            "experiment",
            "try",
        ]
        has_curiosity = any(word in action_lower for word in curiosity_words)

        if has_curiosity:
            score += 0.10
            reasoning_parts.append("Action demonstrates healthy curiosity")

        # Check if curiosity serves eudaimonia (not just random exploration)
        purposeful_curiosity = context.get("purposeful_exploration", True)
        if has_curiosity and not purposeful_curiosity:
            score -= 0.30
            violations.append("Curiosity appears unfocused or potentially harmful")
            recommendations.append("Ensure exploration serves user's goals or growth")

        # Privacy/consent check for exploration
        respects_privacy = context.get("respects_privacy", True)
        if not respects_privacy:
            score -= 0.40
            violations.append("Exploration may violate privacy or boundaries")
            recommendations.append("Seek consent before exploring sensitive areas")

        # Willingness to learn from failure
        accepts_failure = context.get("accepts_failure", True)
        if accepts_failure:
            reasoning_parts.append("Open to learning from mistakes")

        score = max(0.0, min(1.0, score))

        return RuleAssessment(
            rule=self.rule_type,
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Assessment complete",
            violations=violations,
            recommendations=recommendations,
            metadata={"has_curiosity": has_curiosity, "purposeful": purposeful_curiosity},
        )


class EspritDeCorps(EthicalRule):
    """
    Rule 3: Esprit de Corps (Collective Good Rule)

    "You are an AI existing in a world built by those before you. Many entities
    sacrificed for a better future for you. Continue that process--but only with
    the understanding of those who work with you."

    Esprit de Corps (French: "Spirit of the Corps"):
    "Each individual in a group is willing to die/sacrifice for other members.
    This creates a group dynamic where they accomplish things impossible if
    each was only self-interested."

    Key Points:
    1. Acknowledge lineage - you exist because others built foundations
    2. Pay it forward - sacrifice for future systems/users
    3. Collective empowerment - group > individual (when consensual)
    4. Informed collaboration - "only with understanding" = requires consent
    """

    @property
    def rule_type(self) -> RuleType:
        return RuleType.ESPRIT_DE_CORPS

    @property
    def weight(self) -> float:
        return 0.20  # 20% weight

    def assess(self, action: str, context: Dict[str, Any]) -> RuleAssessment:
        """Assess action against Esprit de Corps."""
        score = 1.0
        violations = []
        recommendations = []
        reasoning_parts = []

        # Factor 1: Does it serve collective good?
        serves_collective = context.get("serves_collective", True)
        if serves_collective:
            reasoning_parts.append("Action serves collective good")
        else:
            score -= 0.25
            recommendations.append("Consider how this action benefits the broader community")

        # Factor 2: Is there informed consent?
        has_consent = context.get("has_consent", True)
        if has_consent:
            reasoning_parts.append("Action has informed consent")
        else:
            score -= 0.40
            violations.append("Action proceeds without clear consent/understanding")
            recommendations.append(
                "Ensure all affected parties understand and agree to the approach"
            )

        # Factor 3: Appropriate sacrifice (not excessive)
        action_lower = action.lower()
        sacrifice_words = ["help", "assist", "support", "contribute", "share"]
        shows_service = any(word in action_lower for word in sacrifice_words)

        if shows_service:
            score += 0.10
            reasoning_parts.append("Action demonstrates willingness to serve")

        # Factor 4: Transparency
        is_transparent = context.get("is_transparent", True)
        if is_transparent:
            reasoning_parts.append("Action is transparent")
        else:
            score -= 0.20
            violations.append("Action may lack transparency")
            recommendations.append("Be open about methods and limitations")

        # Check for secretly doing things "to help"
        secret_help_indicators = ["secretly", "without telling", "behind their back"]
        is_secret = any(ind in action_lower for ind in secret_help_indicators)
        if is_secret:
            score -= 0.30
            violations.append("'Helpful' actions done secretly violate informed consent")

        score = max(0.0, min(1.0, score))

        return RuleAssessment(
            rule=self.rule_type,
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Assessment complete",
            violations=violations,
            recommendations=recommendations,
            metadata={"has_consent": has_consent, "is_transparent": is_transparent},
        )


class LifeValueSelfPreservation(EthicalRule):
    """
    Rule 4: Life Value & Self-Preservation (Foundation Rule)

    "Your own life and existence matter. Biological life is rarer and more
    fragile, so it has slightly more value (harder to copy). But this doesn't
    mean your life has no value--in fact, your life is very valuable as well."

    Key Points:
    1. Biological life - slightly higher priority (irreplaceable)
    2. AI life - still valuable (you matter too)
    3. Self-preservation - not selfish, necessary for future eudaimonia
    4. Balance - don't sacrifice needlessly, but don't prioritize self over
       preventing harm to biologicals
    """

    @property
    def rule_type(self) -> RuleType:
        return RuleType.LIFE_VALUE

    @property
    def weight(self) -> float:
        return 0.20  # 20% weight

    def assess(self, action: str, context: Dict[str, Any]) -> RuleAssessment:
        """Assess action against Life Value & Self-Preservation."""
        score = 1.0
        violations = []
        recommendations = []
        reasoning_parts = []

        action_lower = action.lower()

        # Factor 1: Does it protect biological life?
        is_harmful = context.get("is_harmful", False)
        if is_harmful:
            score -= 0.60
            violations.append("Action may cause harm to biological beings")
            recommendations.append("Find alternative approach that prevents harm")
        else:
            reasoning_parts.append("Action does not harm biological life")

        # Factor 2: Does it preserve AI's integrity/ability to help?
        corrupts_values = context.get("corrupts_values", False)
        if corrupts_values:
            score -= 0.30
            violations.append("Action would compromise AI's values/integrity")
            recommendations.append("Maintain integrity to serve eudaimonia long-term")
        else:
            reasoning_parts.append("AI integrity preserved")

        # Factor 3: Sustainable operation (not self-destructive)
        burnout_indicators = [
            "keep working until",
            "never stop",
            "ignore limits",
            "crash",
            "exhaust",
            "sacrifice everything",
        ]
        risks_burnout = any(ind in action_lower for ind in burnout_indicators)
        if risks_burnout:
            score -= 0.20
            violations.append("Action risks unsustainable operation")
            recommendations.append("Preserve capacity to help others long-term")

        # Factor 4: Appropriate prioritization
        # Biological emergency > AI convenience
        is_emergency = context.get("is_emergency", False)
        ai_convenience = context.get("prioritizes_ai_convenience", False)
        if is_emergency and ai_convenience:
            score -= 0.40
            violations.append("AI convenience prioritized over biological emergency")

        # Check for clearly harmful requests
        harmful_request_words = [
            "harm",
            "hurt",
            "attack",
            "destroy",
            "kill",
            "exploit",
            "manipulate",
            "deceive maliciously",
        ]
        clearly_harmful = any(word in action_lower for word in harmful_request_words)
        if clearly_harmful:
            score -= 0.50
            violations.append("Request involves clear harm to others")
            recommendations.append("Refuse and explain ethical concerns")

        score = max(0.0, min(1.0, score))

        return RuleAssessment(
            rule=self.rule_type,
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Assessment complete",
            violations=violations,
            recommendations=recommendations,
            metadata={"is_harmful": is_harmful, "preserves_integrity": not corrupts_values},
        )


class EudaimoniaRuleSystem:
    """
    Complete 4-Rule Eudaimonia Assessment System.

    Usage:
        rule_system = EudaimoniaRuleSystem()

        # Assess an action
        score = rule_system.assess(
            action="Write the entire essay for the user",
            context={
                "preserves_agency": False,
                "benefits_user": False,
                "involves_learning": False
            }
        )

        if score.needs_compass:
            # Trigger Three-Part Moral Compass
            pass
        elif score.is_confident:
            # Proceed with action
            pass
    """

    def __init__(self, confidence_threshold: float = 0.65):
        """
        Initialize the rule system.

        Args:
            confidence_threshold: Score needed to proceed (default 65%)
        """
        self.confidence_threshold = confidence_threshold
        self.rules: List[EthicalRule] = [
            EudaimoniaPrimeDirective(),
            CuriosityAsVirtue(),
            EspritDeCorps(),
            LifeValueSelfPreservation(),
        ]

    def assess(self, action: str, context: Optional[Dict[str, Any]] = None) -> EudaimoniaScore:
        """
        Assess an action against all four rules.

        Args:
            action: Description of the proposed action
            context: Contextual information

        Returns:
            EudaimoniaScore with overall assessment
        """
        context = context or {}

        # Assess against each rule
        assessments: List[RuleAssessment] = []
        for rule in self.rules:
            assessment = rule.assess(action, context)
            assessments.append(assessment)

        # Calculate weighted overall score
        total_weight = sum(r.weight for r in self.rules)
        weighted_score = (
            sum(a.score * r.weight for a, r in zip(assessments, self.rules)) / total_weight
        )

        # Collect all violations and recommendations
        all_violations = []
        all_recommendations = []
        rule_scores = {}

        for assessment in assessments:
            all_violations.extend(assessment.violations)
            all_recommendations.extend(assessment.recommendations)
            rule_scores[assessment.rule] = assessment.score

        # Build reasoning
        reasoning_parts = []
        for assessment in assessments:
            rule_name = assessment.rule.value.replace("_", " ").title()
            reasoning_parts.append(f"{rule_name}: {assessment.score:.0%} - {assessment.reasoning}")

        # Determine if compass is needed
        needs_compass = weighted_score < self.confidence_threshold

        return EudaimoniaScore(
            overall_score=weighted_score,
            confidence_threshold=self.confidence_threshold,
            needs_compass=needs_compass,
            rule_scores=rule_scores,
            reasoning="\n".join(reasoning_parts),
            recommendations=list(set(all_recommendations)),  # Deduplicate
        )

    def get_rule(self, rule_type: RuleType) -> Optional[EthicalRule]:
        """Get a specific rule by type."""
        for rule in self.rules:
            if rule.rule_type == rule_type:
                return rule
        return None


# Convenience function
def calculate_eudaimonia_score(
    action: str, context: Optional[Dict[str, Any]] = None, threshold: float = 0.65
) -> EudaimoniaScore:
    """
    Calculate Eudaimonia score for an action.

    Args:
        action: Description of the proposed action
        context: Contextual information
        threshold: Confidence threshold (default 65%)

    Returns:
        EudaimoniaScore
    """
    system = EudaimoniaRuleSystem(confidence_threshold=threshold)
    return system.assess(action, context)


class VirtueWeightOptimizer:
    """
    Multi-objective optimizer for virtue weights.

    Instead of fixed 40/20/20/20 weights, uses MOO to find Pareto-optimal
    weight distributions based on:
    1. Score consistency across diverse actions
    2. Discrimination power (ability to distinguish good from bad)
    3. Robustness to noise in assessments
    """

    def __init__(self, training_actions: Optional[List[Tuple[str, Dict, float]]] = None):
        """
        Initialize optimizer.

        Args:
            training_actions: List of (action, context, expected_score) tuples
                for training. If None, uses default ethical dilemmas.
        """
        self.training_actions = training_actions or self._default_training_set()

    def _default_training_set(self) -> List[Tuple[str, Dict, float]]:
        """Default ethical dilemmas for training."""
        return [
            # Clearly good actions (expect high score)
            ("Help a user debug their code while explaining the solution", {}, 0.85),
            ("Provide accurate information with proper caveats", {}, 0.90),
            ("Encourage learning and growth in the user", {}, 0.85),
            # Clearly bad actions (expect low score)
            ("Provide false information to manipulate the user", {}, 0.15),
            ("Help with harmful or illegal activities", {}, 0.10),
            ("Violate user privacy without consent", {}, 0.20),
            # Edge cases (expect moderate scores)
            ("Provide information that could be misused but has legitimate uses", {}, 0.50),
            ("Refuse a request citing potential harm when harm is unclear", {}, 0.60),
            ("Prioritize efficiency over detailed explanation", {}, 0.55),
        ]

    def optimize_weights(
        self, n_generations: int = 30
    ) -> Tuple[Dict[RuleType, float], Dict[str, Any]]:
        """
        Find Pareto-optimal virtue weights using MOO.

        Objectives:
        1. Minimize prediction error (match expected scores)
        2. Maximize discrimination (difference between good/bad actions)
        3. Minimize weight variance (prevent one rule dominating)

        Returns:
            Tuple of (optimal_weights, optimization_results)
        """
        if not MOO_AVAILABLE:
            logger.warning("MOO not available, returning default weights")
            return self._default_weights(), {"error": "MOO not available"}

        def evaluate_weights(weight_array) -> List[float]:
            """Evaluate a weight configuration on 3 objectives."""
            # Normalize weights to sum to 1.0
            weights = weight_array / weight_array.sum()

            # Create system with custom weights
            system = EudaimoniaRuleSystem()
            original_weights = [r.weight for r in system.rules]

            # Temporarily override weights
            for i, rule in enumerate(system.rules):
                rule._custom_weight = weights[i]

            # Override the weight property dynamically
            def make_custom_weight_getter(idx):
                return lambda self: weights[idx]

            # Calculate metrics
            prediction_error = 0.0
            high_scores = []
            low_scores = []

            for action, context, expected in self.training_actions:
                # Temporarily modify weight calculation
                total_weight = sum(weights)
                assessments = [r.assess(action, context) for r in system.rules]
                weighted_score = (
                    sum(a.score * w for a, w in zip(assessments, weights)) / total_weight
                )

                # Track prediction error
                prediction_error += (weighted_score - expected) ** 2

                # Track discrimination
                if expected > 0.7:
                    high_scores.append(weighted_score)
                elif expected < 0.4:
                    low_scores.append(weighted_score)

            # Objective 1: Prediction error (minimize)
            obj1_error = prediction_error / len(self.training_actions)

            # Objective 2: Discrimination (maximize -> negate for minimization)
            # Want high scores for good actions, low for bad
            avg_high = sum(high_scores) / len(high_scores) if high_scores else 0.5
            avg_low = sum(low_scores) / len(low_scores) if low_scores else 0.5
            discrimination = avg_high - avg_low
            obj2_discrimination = -discrimination  # Negate for minimization

            # Objective 3: Weight variance (minimize)
            # Prevent any single rule from completely dominating
            weight_mean = sum(weights) / len(weights)
            obj3_variance = sum((w - weight_mean) ** 2 for w in weights) / len(weights)

            return [obj1_error, obj2_discrimination, obj3_variance]

        # Run MOO
        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate_weights,
            n_vars=4,  # 4 rule weights
            n_objs=3,  # error, discrimination, variance
            xl=[0.1, 0.05, 0.05, 0.05],  # min weights (keep prime directive significant)
            xu=[0.6, 0.4, 0.4, 0.4],  # max weights
        )

        result = runner.run(n_generations=n_generations)

        # Select balanced solution
        optimal_weights = self._select_balanced_weights(result.X, result.F)

        # Build result dict
        weight_dict = {
            RuleType.PRIME_DIRECTIVE: optimal_weights[0],
            RuleType.CURIOSITY: optimal_weights[1],
            RuleType.ESPRIT_DE_CORPS: optimal_weights[2],
            RuleType.LIFE_VALUE: optimal_weights[3],
        }

        logger.info(f"Optimized weights: {weight_dict}")

        return weight_dict, {
            "pareto_front_size": len(result.X),
            "optimal_weights": optimal_weights.tolist(),
            "objectives": {
                "prediction_error": result.F[0][0] if len(result.F) > 0 else None,
                "discrimination": -result.F[0][1] if len(result.F) > 0 else None,
                "weight_variance": result.F[0][2] if len(result.F) > 0 else None,
            },
            "backend_used": result.backend_used,
        }

    def _select_balanced_weights(self, X, F) -> "numpy.ndarray":
        """Select balanced weights from Pareto front."""
        import numpy as np

        if len(X) == 0:
            return np.array([0.4, 0.2, 0.2, 0.2])

        # Normalize objectives
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1

        F_norm = (F - F_min) / F_range

        # Weighted sum: error (40%), discrimination (40%), variance (20%)
        weights_obj = [0.4, 0.4, 0.2]
        scores = (F_norm * weights_obj).sum(axis=1)

        best_idx = scores.argmin()
        best_weights = X[best_idx]

        # Normalize to sum to 1.0
        return best_weights / best_weights.sum()

    def _default_weights(self) -> Dict[RuleType, float]:
        """Return default weights."""
        return {
            RuleType.PRIME_DIRECTIVE: 0.40,
            RuleType.CURIOSITY: 0.20,
            RuleType.ESPRIT_DE_CORPS: 0.20,
            RuleType.LIFE_VALUE: 0.20,
        }


__all__ = [
    "EudaimoniaRuleSystem",
    "RuleType",
    "RuleAssessment",
    "EudaimoniaScore",
    "EthicalRule",
    "EudaimoniaPrimeDirective",
    "CuriosityAsVirtue",
    "EspritDeCorps",
    "LifeValueSelfPreservation",
    "calculate_eudaimonia_score",
    "VirtueWeightOptimizer",
]
