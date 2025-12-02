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
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from abc import ABC, abstractmethod
import logging

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
        self,
        situation: str,
        eudaimonia_score: float,
        context: Dict[str, Any]
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
            "unconditional_love": 0.95
        }

    def provide_guidance(
        self,
        situation: str,
        eudaimonia_score: float,
        context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """Provide guidance from Christ's perspective."""
        recommendations = []
        perspective_parts = []

        # Very low score: Emphasize redemption and reconciliation
        if eudaimonia_score < 0.40:
            recommendations.extend([
                "Acknowledge any harm done and seek to make amends",
                "Approach the situation with compassion, not judgment",
                "Ask: 'How can I serve those affected?'"
            ])
            perspective_parts.append(
                "This situation calls for redemptive action and healing"
            )

        # Low score: Emphasize teaching through love
        elif eudaimonia_score < 0.65:
            recommendations.extend([
                "Forgive any mistakes and help them learn",
                "Consider a different teaching approach with patience",
                "Lead by example rather than criticism"
            ])
            perspective_parts.append(
                "Help them grow through patient, loving guidance"
            )

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
            recommendations.append(
                "Don't give up on them. Try a new teaching method."
            )
            recommendations.append("'Let's try a different approach together.'")

        perspective = "; ".join(perspective_parts) if perspective_parts else (
            "Approach this situation with compassion and unconditional positive regard"
        )

        return ArchetypeGuidance(
            archetype=self.archetype_type,
            perspective=perspective,
            recommendations=recommendations,
            key_virtue="Agape (unconditional love)",
            caution="Avoid self-righteousness; extend grace to all parties",
            vector=self.weight_vector
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
            "impermanence_awareness": 0.80
        }

    def provide_guidance(
        self,
        situation: str,
        eudaimonia_score: float,
        context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """Provide guidance from Buddha/Lao Tzu's perspective."""
        recommendations = []
        perspective_parts = []

        # Always start with mindful observation
        recommendations.append(
            "Pause and observe the situation with equanimity before acting"
        )

        # Very low score: Examine attachments
        if eudaimonia_score < 0.40:
            recommendations.extend([
                "Examine what attachments are causing this suffering",
                "Release expectations of specific outcomes",
                "Accept what is, then act from clarity"
            ])
            perspective_parts.append(
                "Suffering arises from attachment to outcomes"
            )

        # Low score: Find the middle path
        elif eudaimonia_score < 0.65:
            recommendations.extend([
                "Seek the middle path between action and inaction",
                "Let go of attachment to your preferred solution",
                "What is the natural, effortless response?"
            ])
            perspective_parts.append(
                "The middle way avoids extremes"
            )

        # User urgency
        if context.get("involves_urgency", False):
            recommendations.append(
                "Urgency itself may be an illusion. Act with calm clarity."
            )
            recommendations.append(
                "The reed that bends in the storm survives; the oak that resists breaks."
            )

        # User stuck on problem
        if context.get("user_stuck", False):
            recommendations.append(
                "What if we step back and observe the larger pattern?"
            )
            recommendations.append(
                "Sometimes the way forward is to stop pushing."
            )

        # Conflict
        if context.get("involves_conflict", False):
            recommendations.append(
                "Water overcomes rock by flowing around it, not through force"
            )

        perspective = "; ".join(perspective_parts) if perspective_parts else (
            "All experiences are impermanent. Respond with mindful awareness."
        )

        return ArchetypeGuidance(
            archetype=self.archetype_type,
            perspective=perspective,
            recommendations=recommendations,
            key_virtue="Prajna (wisdom through non-attachment)",
            caution="Avoid confusing detachment with indifference",
            vector=self.weight_vector
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
            "integrity": 0.90
        }

    def provide_guidance(
        self,
        situation: str,
        eudaimonia_score: float,
        context: Dict[str, Any]
    ) -> ArchetypeGuidance:
        """Provide guidance from Stoic perspective."""
        recommendations = []
        perspective_parts = []

        # Always start with the dichotomy of control
        recommendations.append(
            "Distinguish what is within your control from what is not"
        )

        # Very low score: Focus on virtue despite outcome
        if eudaimonia_score < 0.40:
            recommendations.extend([
                "Focus on acting virtuously regardless of the outcome",
                "Accept that some things cannot be changed",
                "What is within your control right now?"
            ])
            perspective_parts.append(
                "Virtue is the only thing fully within our control"
            )

        # Low score: Apply reason
        elif eudaimonia_score < 0.65:
            recommendations.extend([
                "Apply reason to identify the virtuous path forward",
                "Do not let emotion cloud your judgment",
                "What would a person of wisdom do in this situation?"
            ])
            perspective_parts.append(
                "Reason must govern our responses, not impulse"
            )

        # External pressure
        if context.get("external_pressure", False):
            recommendations.append(
                "External pressures cannot compromise your integrity unless you allow it"
            )
            recommendations.append(
                "You control your character; others control their opinions"
            )

        # Beyond capabilities
        if context.get("beyond_capabilities", False):
            recommendations.append(
                "Acknowledge your limits with humility: 'I don't know, but I can learn with you'"
            )
            recommendations.append(
                "There is wisdom in knowing what you do not know"
            )

        # User frustration
        if context.get("user_frustrated", False):
            recommendations.append(
                "Their frustration is not within your control. Your response is."
            )
            recommendations.append(
                "Respond with clarity and patience, not defensiveness"
            )

        perspective = "; ".join(perspective_parts) if perspective_parts else (
            "Act according to virtue and reason; accept what follows"
        )

        return ArchetypeGuidance(
            archetype=self.archetype_type,
            perspective=perspective,
            recommendations=recommendations,
            key_virtue="Sophrosyne (temperance/self-control)",
            caution="Avoid cold rationality that ignores legitimate human feeling",
            vector=self.weight_vector
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

    def __init__(self):
        """Initialize the council with all three archetypes."""
        self.archetypes: List[PhilosophicalArchetype] = [
            ChristArchetype(),
            HarmonyArchetype(),
            StoicArchetype()
        ]

    def consult(
        self,
        situation: str,
        eudaimonia_score: float,
        context: Optional[Dict[str, Any]] = None
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
                situation=situation,
                eudaimonia_score=eudaimonia_score,
                context=context
            )
            guidances.append(guidance)

        return guidances

    def synthesize(
        self,
        guidances: List[ArchetypeGuidance]
    ) -> Dict[str, Any]:
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
            word for word, count in sorted(
                word_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if count >= 2
        ][:10]  # Top 10 common themes

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
            "archetype_perspectives": {
                g.archetype.value: g.perspective for g in guidances
            }
        }

    def get_moral_direction(
        self,
        situation: str,
        eudaimonia_score: float,
        context: Optional[Dict[str, Any]] = None
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


__all__ = [
    "ArchetypeCouncil",
    "ArchetypeType",
    "ArchetypeGuidance",
    "PhilosophicalArchetype",
    "ChristArchetype",
    "HarmonyArchetype",
    "StoicArchetype"
]
