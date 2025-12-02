"""
Phase 3 Reasoning Data Generator

Generates 25,000 high-quality reasoning examples using 5 frontier models:
- OpenAI GPT-4o (best OpenAI model)
- Anthropic Claude 3.5 Sonnet (best Anthropic model)
- Google Gemini Pro 1.5 (best Google model)
- xAI Grok Beta (best xAI model)
- Qwen 2.5 72B (best Qwen model)

Each model generates 500 examples per strategy (10 strategies total).
Batch generation: 100 examples per API call (minimizes API calls to ~250 total).
Output format is TRAINING-READY for prompt baking.

Token Structure:
- OUTER: [thinking] and [/endthinking] (2 tokens)
- INNER: 10 reasoning strategies (10 tokens):
  1. Chain-of-Thought (<step>)
  2. MECE Decomposition (<mece>)
  3. Falsification Testing (<falsify>)
  4. Expert Perspective (<expert>)
  5. Orthogonal Wisdom (<orthogonal>)
  6. Self-Doubt (<doubt>)
  7. Bayesian Rationalist (<bayesian>)
  8. Multidomain Consultant (<multidomain>)
  9. Self-Correction (<correct>)
  10. Uncertainty Expression (<uncertain>)

Total special tokens: 12 (2 outer + 10 inner)

Version: 2.0.0
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import wandb
from openrouter_client import (
    OpenRouterClient,
    PRODUCTION_MODELS,
    GenerationRequest,
    GenerationResponse
)


# Reasoning Strategies with [thinking] Wrapper
# OUTER WRAPPER: [thinking] and [/endthinking] (2 tokens)
# INSIDE WRAPPER: 10 reasoning strategies (each with its own tokens)
REASONING_STRATEGIES = {
    "chain_of_thought": {
        "name": "Chain-of-Thought",
        "description": "Step-by-step logical reasoning",
        "special_tokens": ["<step>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["easy", "medium", "hard"]
    },
    "mece": {
        "name": "MECE Decomposition",
        "description": "Mutually Exclusive, Collectively Exhaustive categorization",
        "special_tokens": ["<mece>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["medium", "hard"]
    },
    "falsification": {
        "name": "Falsification Testing",
        "description": "Identifying what would prove a belief wrong",
        "special_tokens": ["<falsify>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["medium", "hard"]
    },
    "expert_perspective": {
        "name": "Expert Perspective",
        "description": "Thinking from domain expert viewpoints",
        "special_tokens": ["<expert>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["medium", "hard"]
    },
    "orthogonal_wisdom": {
        "name": "Orthogonal Wisdom",
        "description": "Drawing insights from unrelated fields",
        "special_tokens": ["<orthogonal>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["hard"]
    },
    "self_doubt": {
        "name": "Self-Doubt & Error Checking",
        "description": "Questioning assumptions and checking for errors",
        "special_tokens": ["<doubt>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["easy", "medium", "hard"]
    },
    "bayesian": {
        "name": "Bayesian Rationalist",
        "description": "Updating beliefs based on evidence",
        "special_tokens": ["<bayesian>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["hard"]
    },
    "multidomain": {
        "name": "Multidomain Consultant",
        "description": "Synthesizing insights from multiple expert domains",
        "special_tokens": ["<multidomain>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["hard"]
    },
    "self_correction": {
        "name": "Self-Correction",
        "description": "Identifying and fixing errors in reasoning",
        "special_tokens": ["<correct>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["medium", "hard"]
    },
    "uncertainty": {
        "name": "Uncertainty Expression",
        "description": "Explicitly acknowledging limits of knowledge",
        "special_tokens": ["<uncertain>"],  # Inside [thinking] bubble
        "examples_per_model": 500,
        "difficulty_levels": ["easy", "medium", "hard"]
    }
}


# All special tokens (for tokenizer expansion)
# OUTER WRAPPER (2 tokens)
OUTER_TOKENS = ["[thinking]", "[/endthinking]"]

# INNER STRATEGY TOKENS (10 tokens)
INNER_TOKENS = []
for strategy in REASONING_STRATEGIES.values():
    INNER_TOKENS.extend(strategy['special_tokens'])
INNER_TOKENS = list(set(INNER_TOKENS))  # Deduplicate

# ALL SPECIAL TOKENS (12-15 total)
ALL_SPECIAL_TOKENS = OUTER_TOKENS + INNER_TOKENS
# Result: 2 outer + 10 inner = 12 special tokens total


# Batch generation prompt (generates many examples at once)
def create_batch_generation_prompt(strategy_key: str, num_examples: int) -> str:
    """
    Create a prompt that generates MANY examples in one API call

    Args:
        strategy_key: Reasoning strategy
        num_examples: Number of examples to generate (e.g., 100)

    Returns:
        Batch generation prompt
    """
    strategy = REASONING_STRATEGIES[strategy_key]

    # Strategy-specific instructions
    strategy_instructions = {
        "chain_of_thought": """
Chain-of-Thought examples demonstrate step-by-step reasoning.
Format:
[thinking]
<step>Step 1 explanation</step>
<step>Step 2 explanation</step>
<step>Step 3 explanation</step>
[/endthinking]
Answer: [final answer]

Topics to cover: math, logic, science, planning, debugging, analysis.
Difficulty: Mix of easy (arithmetic), medium (algebra), hard (multi-step proofs).
""",
        "mece": """
MECE examples show mutually exclusive, collectively exhaustive decomposition.
Format:
[thinking]
<mece>Breaking this into ME and CE categories:</mece>
<step>Analysis of category 1</step>
<step>Analysis of category 2</step>
<step>Verification: Categories are mutually exclusive and collectively exhaustive</step>
[/endthinking]
Answer: [synthesized answer]

Topics: business strategy, problem classification, root cause analysis, system design.
""",
        "falsification": """
Falsification examples demonstrate critical thinking and hypothesis testing.
Format:
[thinking]
<falsify>What would prove this wrong?</falsify>
<step>State the claim clearly</step>
<step>Specific test: If claim is true, then X; if X doesn't happen, claim is false</step>
<step>Evaluate evidence</step>
<step>Conclusion about whether claim survives falsification</step>
[/endthinking]
Answer: [verdict on claim]

Topics: scientific hypotheses, beliefs, predictions, theories.
""",
        "expert_perspective": """
Expert perspective examples show domain-specific thinking.
Format:
[thinking]
<expert>How would a [domain] expert think about this?</expert>
<step>Expert framework 1: [name and application]</step>
<step>Expert framework 2: [name and application]</step>
<step>Synthesis of expert insights</step>
[/endthinking]
Answer: [expert-informed answer]

Use REAL expert frameworks: CAPM (finance), P/E analysis (investing), load-bearing analysis (engineering), etc.
""",
        "orthogonal_wisdom": """
Orthogonal wisdom examples draw from unrelated fields.
Format:
[thinking]
<orthogonal>What unrelated fields solved similar problems?</orthogonal>
<step>Field 1 insight: [specific example from unrelated field]</step>
<step>Field 2 insight: [another unrelated field example]</step>
<step>Common pattern across fields</step>
[/endthinking]
Answer: [novel solution from orthogonal thinking]

Example: Solve business problem with military tactics, design problem with biology.
""",
        "self_doubt": """
Self-doubt examples demonstrate error checking and intellectual humility.
Format:
[thinking]
<step>Initial approach</step>
<doubt>Wait, could I be wrong?</doubt>
<step>Check 1: [specific verification]</step>
<step>Result of check 1</step>
<step>Check 2: [another verification]</step>
<step>Result of check 2</step>
<step>Final answer (corrected if needed)</step>
[/endthinking]
Answer: [verified answer]

IMPORTANT: Include examples where initial answer WAS wrong and gets corrected.
""",
        "bayesian": """
Bayesian examples use Bayes' theorem for probabilistic reasoning.
Format:
[thinking]
<bayesian>Updating beliefs based on evidence</bayesian>
<step>P(Hypothesis) = [value]</step>
<step>P(Evidence | Hypothesis) = [value]</step>
<step>P(Evidence) = [calculation]</step>
<step>Bayes' theorem: P(H|E) = P(E|H) × P(H) / P(E)</step>
<step>P(Hypothesis | Evidence) = [value]</step>
<step>Interpretation and common pitfalls</step>
[/endthinking]
Answer: [probabilistically correct answer]

Topics: medical diagnosis, risk assessment, belief updating. Show actual calculations.
""",
        "multidomain": """
Multidomain examples synthesize insights from 3+ expert domains.
Format:
[thinking]
<multidomain>Consulting multiple expert domains</multidomain>
<step>Domain 1 ([field]): Framework/principle</step>
<step>Domain 2 ([field]): Framework/principle</step>
<step>Domain 3 ([field]): Framework/principle</step>
<step>Where domains agree/disagree</step>
<step>Integrated recommendation</step>
[/endthinking]
Answer: [multidomain-informed decision]

Topics: complex decisions, strategy, product management. Use 3-5 different domains.
""",
        "self_correction": """
Self-correction examples show identifying and fixing reasoning errors.
Format:
[thinking]
<step>Initial reasoning</step>
<correct>Wait, I made an error: [specific error]</correct>
<step>Corrected reasoning</step>
<step>Verification that correction is valid</step>
[/endthinking]
Answer: [corrected answer]

IMPORTANT: Show REAL errors being caught and fixed.
""",
        "uncertainty": """
Uncertainty examples demonstrate intellectual honesty about knowledge limits.
Format:
[thinking]
<step>What I know for certain: [facts]</step>
<uncertain>What I'm unsure about: [specific uncertainties]</uncertain>
<step>Why uncertainty exists: [reasons]</step>
<step>How to reduce uncertainty: [methods]</step>
<step>Best answer given uncertainty: [qualified conclusion]</step>
[/endthinking]
Answer: [answer with confidence qualifiers]

Topics: ambiguous scenarios, incomplete information, probabilistic outcomes.
"""
    }

    prompt = f"""
You are generating {num_examples} high-quality reasoning examples for the "{strategy['name']}" strategy.

**Strategy**: {strategy['description']}

**Instructions**:
{strategy_instructions[strategy_key]}

**Output Format** (EXACTLY):
```json
[
  {{
    "id": 1,
    "prompt": "Problem statement here",
    "reasoning": "[thinking]...[/endthinking] formatted reasoning here",
    "answer": "Final answer here",
    "difficulty": "easy|medium|hard",
    "topic": "Topic area (e.g., math, science, strategy)"
  }},
  {{
    "id": 2,
    "prompt": "...",
    "reasoning": "...",
    "answer": "...",
    "difficulty": "...",
    "topic": "..."
  }},
  ...
]
```

**Requirements**:
1. Generate EXACTLY {num_examples} examples
2. Use the EXACT format shown above (JSON array)
3. Every example MUST use the correct special tokens
4. Vary difficulty levels: {', '.join(strategy.get('difficulty_levels', ['medium']))}
5. Vary topics broadly
6. Ensure high quality (clear, correct, educational)
7. NO explanatory text outside the JSON array

Generate {num_examples} examples NOW:
"""

    return prompt


@dataclass
class TrainingExample:
    """Training-ready example for prompt baking"""
    id: str  # Unique identifier
    prompt: str  # Input problem
    reasoning: str  # [thinking]...[/endthinking] formatted reasoning
    answer: str  # Final answer
    strategy: str  # Reasoning strategy
    difficulty: str  # easy, medium, hard
    topic: str  # Topic area
    model_source: str  # Which model generated this
    timestamp: str
    quality_score: float = 0.0  # Post-generation quality scoring


class Phase3DataGenerator:
    """Generate training-ready reasoning data for Phase 3"""

    def __init__(self, wandb_enabled: bool = True):
        """
        Initialize generator

        Args:
            wandb_enabled: Whether to log to W&B
        """
        self.client = OpenRouterClient(wandb_enabled=wandb_enabled)
        self.wandb_enabled = wandb_enabled

        if wandb_enabled:
            wandb.init(
                project="agent-forge-v2-phase3",
                name=f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['phase3', 'reasoning_data', 'frontier_models', 'batched']
            )

    async def generate_strategy_data(
        self,
        model_key: str,
        strategy_key: str,
        total_examples: int = 500,
        batch_size: int = 100
    ) -> List[TrainingExample]:
        """
        Generate examples for one strategy using one model

        Args:
            model_key: Model key (e.g., "gpt-4o")
            strategy_key: Strategy key (e.g., "chain_of_thought")
            total_examples: Total examples to generate (default 500)
            batch_size: Examples per API call (default 100 to minimize calls)

        Returns:
            List of TrainingExample
        """
        model = PRODUCTION_MODELS[model_key]
        strategy = REASONING_STRATEGIES[strategy_key]

        print(f"\n🔄 Generating {total_examples} {strategy['name']} examples using {model.name}...")
        print(f"   Strategy: {batch_size} examples per API call → {total_examples // batch_size} API calls")

        all_examples = []

        # Generate in large batches (minimize API calls!)
        num_batches = (total_examples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            examples_this_batch = min(batch_size, total_examples - len(all_examples))

            # Create batch generation request
            prompt = create_batch_generation_prompt(strategy_key, examples_this_batch)

            request = GenerationRequest(
                prompt=prompt,
                system_message="You are an expert reasoning instructor generating training data for AI models. Output ONLY valid JSON.",
                metadata={
                    'strategy': strategy_key,
                    'model': model_key,
                    'batch_size': examples_this_batch,
                    'batch_idx': batch_idx
                }
            )

            # Generate (single API call for entire batch!)
            responses = await self.client.generate_batch(
                model,
                [request],
                batch_size=1  # One request at a time (but request contains many examples)
            )

            if responses and responses[0].success:
                # Parse JSON response
                examples = self._parse_batch_response(
                    responses[0],
                    model_key,
                    strategy_key
                )
                all_examples.extend(examples)

                print(f"  ✅ Batch {batch_idx + 1}/{num_batches}: Generated {len(examples)} examples (cost: ${responses[0].cost_usd:.4f})")

                if self.wandb_enabled:
                    wandb.log({
                        f'generation/{model_key}/{strategy_key}/batch_{batch_idx}/examples': len(examples),
                        f'generation/{model_key}/{strategy_key}/batch_{batch_idx}/cost_usd': responses[0].cost_usd
                    })
            else:
                print(f"  ❌ Batch {batch_idx + 1}/{num_batches} failed")

        print(f"✅ Total generated: {len(all_examples)}/{total_examples} {strategy['name']} examples")

        return all_examples

    def _parse_batch_response(
        self,
        response: GenerationResponse,
        model_key: str,
        strategy_key: str
    ) -> List[TrainingExample]:
        """
        Parse batch response into TrainingExample objects

        Args:
            response: API response
            model_key: Model identifier
            strategy_key: Strategy identifier

        Returns:
            List of TrainingExample
        """
        examples = []

        try:
            # Extract JSON from response
            text = response.text.strip()

            # Remove markdown code blocks if present
            if text.startswith('```json'):
                text = text[7:]  # Remove ```json
            if text.startswith('```'):
                text = text[3:]  # Remove ```
            if text.endswith('```'):
                text = text[:-3]  # Remove ```

            text = text.strip()

            # Parse JSON
            data = json.loads(text)

            if not isinstance(data, list):
                print(f"  ⚠️  Expected JSON array, got {type(data)}")
                return []

            # Convert to TrainingExample
            for item in data:
                example = TrainingExample(
                    id=f"{model_key}_{strategy_key}_{item.get('id', 0)}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    prompt=item.get('prompt', ''),
                    reasoning=item.get('reasoning', ''),
                    answer=item.get('answer', ''),
                    strategy=strategy_key,
                    difficulty=item.get('difficulty', 'medium'),
                    topic=item.get('topic', 'general'),
                    model_source=model_key,
                    timestamp=datetime.now().isoformat(),
                    quality_score=0.0  # Will score later
                )
                examples.append(example)

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error: {str(e)}")
            print(f"  Response text: {text[:200]}...")

        except Exception as e:
            print(f"  ❌ Parse error: {str(e)}")

        return examples

    async def generate_all_data(
        self,
        examples_per_model_per_strategy: int = 500,
        batch_size: int = 100
    ) -> List[TrainingExample]:
        """
        Generate ALL reasoning data from ALL models for ALL strategies

        Total: 5 models × 10 strategies × 500 examples = 25,000 examples
        API calls: 5 models × 10 strategies × 5 batches = 250 API calls (if batch_size=100)

        Args:
            examples_per_model_per_strategy: Examples each model generates per strategy (default 500)
            batch_size: Examples per API call (default 100)

        Returns:
            List of all TrainingExample
        """
        # Models to use
        model_keys = [
            "gpt-4o",              # Best OpenAI
            "claude-3.5-sonnet",   # Best Anthropic
            "gemini-pro-1.5",      # Best Google
            "grok-beta",           # Best xAI
            "qwen-2.5-72b"         # Best Qwen
        ]

        all_examples = []

        print(f"\n{'='*80}")
        print(f"🚀 STARTING PHASE 3 DATA GENERATION")
        print(f"{'='*80}")
        print(f"Models: {len(model_keys)}")
        print(f"Strategies: {len(REASONING_STRATEGIES)}")
        print(f"Examples per model per strategy: {examples_per_model_per_strategy}")
        print(f"Batch size: {batch_size} examples per API call")
        print(f"Total target: {len(model_keys) * len(REASONING_STRATEGIES) * examples_per_model_per_strategy:,} examples")
        print(f"Total API calls: ~{len(model_keys) * len(REASONING_STRATEGIES) * (examples_per_model_per_strategy // batch_size)}")
        print(f"{'='*80}\n")

        for model_key in model_keys:
            print(f"\n{'='*80}")
            print(f"🤖 MODEL: {PRODUCTION_MODELS[model_key].name}")
            print(f"{'='*80}")

            for strategy_key in REASONING_STRATEGIES.keys():
                examples = await self.generate_strategy_data(
                    model_key,
                    strategy_key,
                    total_examples=examples_per_model_per_strategy,
                    batch_size=batch_size
                )

                all_examples.extend(examples)

        print(f"\n{'='*80}")
        print(f"🎉 GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total examples generated: {len(all_examples):,}")

        stats = self.client.get_stats()
        print(f"Total API calls: {stats['total_requests']}")
        print(f"Total cost: ${stats['total_cost_usd']:.2f}")
        print(f"Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']:,}")

        if self.wandb_enabled:
            wandb.log({
                'generation/total_examples': len(all_examples),
                'generation/total_cost_usd': stats['total_cost_usd'],
                'generation/total_api_calls': stats['total_requests']
            })

        return all_examples

    def save_training_data(self, examples: List[TrainingExample], output_path: str):
        """
        Save in TRAINING-READY format for Phase 3 prompt baking

        Format optimized for:
        1. Direct loading into HuggingFace datasets
        2. Prompt baking training loop
        3. Quality filtering

        Args:
            examples: List of TrainingExample
            output_path: Output JSON file path
        """
        # Group by strategy
        by_strategy = {}
        for ex in examples:
            if ex.strategy not in by_strategy:
                by_strategy[ex.strategy] = []
            by_strategy[ex.strategy].append(ex)

        # Create training-ready format
        training_data = {
            'metadata': {
                'total_examples': len(examples),
                'generation_date': datetime.now().isoformat(),
                'special_tokens': ALL_SPECIAL_TOKENS,
                'num_special_tokens': len(ALL_SPECIAL_TOKENS),
                'strategies': list(REASONING_STRATEGIES.keys()),
                'models_used': list(set(ex.model_source for ex in examples)),
                'examples_per_strategy': {k: len(v) for k, v in by_strategy.items()}
            },
            'special_tokens': ALL_SPECIAL_TOKENS,  # For tokenizer.add_special_tokens()
            'examples': [asdict(ex) for ex in examples]  # Training examples
        }

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved {len(examples)} training examples to {output_path}")
        print(f"   Special tokens: {len(ALL_SPECIAL_TOKENS)}")
        print(f"   Ready for prompt baking!")

        # Also save by strategy (for analysis)
        strategy_output_dir = output_path.replace('.json', '_by_strategy')
        os.makedirs(strategy_output_dir, exist_ok=True)

        for strategy_key, strategy_examples in by_strategy.items():
            strategy_path = os.path.join(strategy_output_dir, f"{strategy_key}.json")
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(ex) for ex in strategy_examples], f, indent=2, ensure_ascii=False)

        print(f"   Also saved by strategy to {strategy_output_dir}/")

        if self.wandb_enabled:
            # Upload as W&B artifact
            artifact = wandb.Artifact(
                name='phase3_reasoning_training_data',
                type='dataset',
                description=f'Training-ready reasoning data ({len(examples)} examples, {len(ALL_SPECIAL_TOKENS)} special tokens)'
            )
            artifact.add_file(output_path)
            wandb.log_artifact(artifact)
            print(f"📊 Uploaded to W&B as artifact 'phase3_reasoning_training_data'")


async def main():
    """Main execution"""

    generator = Phase3DataGenerator(wandb_enabled=True)

    # Generate all data
    examples = await generator.generate_all_data(
        examples_per_model_per_strategy=500,  # 500 examples per model per strategy
        batch_size=100  # 100 examples per API call (5 calls per model per strategy)
    )

    # Save in training-ready format
    generator.save_training_data(
        examples,
        output_path="data/phase3_reasoning_training_data.json"
    )

    # Print statistics
    print(f"\n📈 FINAL STATISTICS:")
    print(f"{'='*80}")

    # By strategy
    print("\nExamples per strategy:")
    for strategy_key, strategy_info in REASONING_STRATEGIES.items():
        count = len([ex for ex in examples if ex.strategy == strategy_key])
        print(f"  {strategy_info['name']:30s}: {count:5d} examples")

    # By model
    print("\nExamples per model:")
    model_keys = ["gpt-4o", "claude-3.5-sonnet", "gemini-pro-1.5", "grok-beta", "qwen-2.5-72b"]
    for model_key in model_keys:
        count = len([ex for ex in examples if ex.model_source == model_key])
        model_name = PRODUCTION_MODELS[model_key].name
        print(f"  {model_name:30s}: {count:5d} examples")

    # By difficulty
    print("\nExamples per difficulty:")
    for difficulty in ["easy", "medium", "hard"]:
        count = len([ex for ex in examples if ex.difficulty == difficulty])
        print(f"  {difficulty:10s}: {count:5d} examples")

    print(f"\n✅ Phase 3 data generation complete!")
    print(f"   Total examples: {len(examples):,}")
    print(f"   Special tokens: {len(ALL_SPECIAL_TOKENS)}")
    print(f"   Ready for prompt baking training!")

    if generator.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
