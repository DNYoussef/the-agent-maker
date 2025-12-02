"""
Frontier Model Data Generation via OpenRouter API

Generates reasoning examples for Phase 3 prompt baking using 5 frontier models:
- OpenAI GPT-4
- Anthropic Claude 3.5 Sonnet
- Google Gemini Pro
- xAI Grok
- Qwen

Each model generates 500 examples × 8 reasoning strategies = 4000 examples per model
Total: 5 models × 4000 = 20,000 examples (deduplicated to ~2500 high-quality examples)

Version: 1.0.0
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from tqdm.asyncio import tqdm_asyncio
import wandb


@dataclass
class ReasoningExample:
    """Single reasoning example with metadata"""
    prompt: str
    reasoning: str
    answer: str
    strategy: str  # e.g., "mece", "falsification", "expert_perspective"
    model: str     # e.g., "gpt-4-turbo", "claude-3.5-sonnet"
    timestamp: str
    quality_score: float = 0.0
    hash: str = ""

    def __post_init__(self):
        """Generate hash for deduplication"""
        content = f"{self.prompt}{self.reasoning}{self.answer}"
        self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class FrontierModelConfig:
    """Configuration for a frontier model via OpenRouter"""
    name: str
    openrouter_id: str
    cost_per_1k_tokens: float
    max_tokens: int = 4096
    temperature: float = 0.8
    top_p: float = 0.95


# Frontier Model Configurations
FRONTIER_MODELS = {
    "openai": FrontierModelConfig(
        name="GPT-4 Turbo",
        openrouter_id="openai/gpt-4-turbo",
        cost_per_1k_tokens=0.01,
        max_tokens=4096,
        temperature=0.8
    ),
    "anthropic": FrontierModelConfig(
        name="Claude 3.5 Sonnet",
        openrouter_id="anthropic/claude-3.5-sonnet",
        cost_per_1k_tokens=0.003,
        max_tokens=8192,
        temperature=0.8
    ),
    "google": FrontierModelConfig(
        name="Gemini Pro 1.5",
        openrouter_id="google/gemini-pro-1.5",
        cost_per_1k_tokens=0.00125,
        max_tokens=8192,
        temperature=0.8
    ),
    "xai": FrontierModelConfig(
        name="Grok Beta",
        openrouter_id="x-ai/grok-beta",
        cost_per_1k_tokens=0.005,
        max_tokens=4096,
        temperature=0.8
    ),
    "qwen": FrontierModelConfig(
        name="Qwen2.5 72B",
        openrouter_id="qwen/qwen-2.5-72b-instruct",
        cost_per_1k_tokens=0.0004,
        max_tokens=4096,
        temperature=0.8
    )
}


# Reasoning Strategies (8 total)
REASONING_STRATEGIES = {
    "chain_of_thought": {
        "name": "Chain-of-Thought",
        "description": "Step-by-step logical reasoning",
        "special_tokens": ["<think>", "</think>", "<step>", "</step>"],
        "examples_per_model": 500
    },
    "mece": {
        "name": "MECE Decomposition",
        "description": "Mutually Exclusive, Collectively Exhaustive categorization",
        "special_tokens": ["<mece>", "</mece>", "<category>", "</category>"],
        "examples_per_model": 500
    },
    "falsification": {
        "name": "Falsification Testing",
        "description": "Identifying what would prove a belief wrong",
        "special_tokens": ["<falsify>", "</falsify>", "<test>", "</test>"],
        "examples_per_model": 500
    },
    "expert_perspective": {
        "name": "Expert Perspective",
        "description": "Thinking from domain expert viewpoints",
        "special_tokens": ["<expert>", "</expert>", "<domain>", "</domain>"],
        "examples_per_model": 500
    },
    "orthogonal_wisdom": {
        "name": "Orthogonal Wisdom",
        "description": "Drawing insights from unrelated fields",
        "special_tokens": ["<orthogonal>", "</orthogonal>", "<source>", "</source>"],
        "examples_per_model": 500
    },
    "self_doubt": {
        "name": "Self-Doubt & Error Checking",
        "description": "Questioning assumptions and checking for errors",
        "special_tokens": ["<doubt>", "</doubt>", "<check>", "</check>"],
        "examples_per_model": 500
    },
    "bayesian": {
        "name": "Bayesian Rationalist",
        "description": "Updating beliefs based on evidence",
        "special_tokens": ["<bayesian>", "</bayesian>", "<prior>", "</prior>", "<posterior>", "</posterior>"],
        "examples_per_model": 500
    },
    "multidomain": {
        "name": "Multidomain Consultant",
        "description": "Synthesizing insights from multiple expert domains",
        "special_tokens": ["<multidomain>", "</multidomain>", "<synthesis>", "</synthesis>"],
        "examples_per_model": 500
    }
}


# Generation prompts for each strategy
STRATEGY_PROMPTS = {
    "chain_of_thought": """
Generate a chain-of-thought reasoning example that demonstrates step-by-step logical thinking.

Format:
**Prompt**: [A problem that requires step-by-step reasoning]
**Reasoning**:
<think>
<step>First step of reasoning</step>
<step>Second step of reasoning</step>
<step>Third step of reasoning</step>
...
</think>
**Answer**: [Final answer]

Requirements:
- Problem should be challenging but solvable
- Each step should logically follow from the previous
- Include verification steps
- Show intermediate calculations
- Demonstrate clear logical flow

Generate 1 high-quality example now.
""",

    "mece": """
Generate a MECE (Mutually Exclusive, Collectively Exhaustive) decomposition example.

Format:
**Prompt**: [A problem requiring categorization or breakdown]
**Reasoning**:
<think>
<mece>Let me break this down into mutually exclusive and collectively exhaustive categories:</mece>
<category>Category 1: [name and description]</category>
<step>Analysis of category 1</step>
<category>Category 2: [name and description]</category>
<step>Analysis of category 2</step>
...
<step>Verification that categories are ME and CE</step>
</think>
**Answer**: [Synthesized answer using MECE framework]

Requirements:
- Categories must be mutually exclusive (no overlap)
- Categories must be collectively exhaustive (cover everything)
- Explicitly verify ME and CE properties
- Use MECE for strategic, analytical, or classification problems
- Show how MECE prevents gaps and overlaps

Generate 1 high-quality example now.
""",

    "falsification": """
Generate a falsification testing example that demonstrates critical thinking.

Format:
**Prompt**: [A claim, hypothesis, or belief to test]
**Reasoning**:
<think>
<falsify>What would actually falsify this belief?</falsify>
<step>Identify the core claim</step>
<step>Design a falsification test: "If X is true, then NOT-Y should never occur"</step>
<test>Specific test that could disprove the claim</test>
<step>Evaluate evidence against the falsification test</step>
<step>Conclusion about whether claim survives falsification</step>
</think>
**Answer**: [Whether claim is falsified, partially supported, or needs more testing]

Requirements:
- Start with a testable claim
- Design SPECIFIC falsification tests (not just "look for counterexamples")
- Show what evidence would DISPROVE the claim
- Demonstrate Popperian falsificationism
- Include real or hypothetical test results

Generate 1 high-quality example now.
""",

    "expert_perspective": """
Generate an expert perspective example showing domain expertise.

Format:
**Prompt**: [A domain-specific problem]
**Reasoning**:
<think>
<expert>How would an expert in [domain] think about this?</expert>
<domain>Domain: [specific field like "behavioral finance", "structural engineering"]</domain>
<step>Expert lens 1: [first expert perspective]</step>
<step>Key insight from expert lens 1</step>
<step>Expert lens 2: [second expert perspective, ideally different domain]</step>
<step>Key insight from expert lens 2</step>
<step>Synthesis: How these expert perspectives inform the solution</step>
</think>
**Answer**: [Answer informed by expert thinking]

Requirements:
- Use REAL expert frameworks (e.g., "CAPM model in finance", "Load-bearing analysis in engineering")
- Show how experts in the field actually approach the problem
- Include domain-specific terminology and concepts
- Demonstrate deep domain knowledge
- Consider multiple expert sub-domains when relevant

Generate 1 high-quality example now.
""",

    "orthogonal_wisdom": """
Generate an orthogonal wisdom example drawing from unrelated fields.

Format:
**Prompt**: [A problem in domain X]
**Reasoning**:
<think>
<step>What problem am I really trying to solve?</step>
<orthogonal>What orthogonal sources of wisdom exist from people who solved similar problems in DIFFERENT fields?</orthogonal>
<source>Field 1 (unrelated to original domain): [specific example]</source>
<step>Insight from field 1 that applies</step>
<source>Field 2 (also unrelated): [specific example]</source>
<step>Insight from field 2 that applies</step>
<source>Field 3 (completely different): [specific example]</source>
<step>Insight from field 3 that applies</step>
<step>Synthesis: Common pattern across orthogonal fields</step>
</think>
**Answer**: [Novel solution inspired by orthogonal wisdom]

Requirements:
- Original problem in domain X
- Draw insights from domains Y, Z, W (UNRELATED to X)
- Show the ABSTRACT PATTERN that transfers
- Demonstrate creative cross-domain thinking
- Examples: solving business problem with military strategy, design problem with biology

Generate 1 high-quality example now.
""",

    "self_doubt": """
Generate a self-doubt and error checking example.

Format:
**Prompt**: [A problem with potential for errors]
**Reasoning**:
<think>
<step>Initial approach/answer</step>
<doubt>Wait, could I be wrong about this?</doubt>
<check>Check 1: [specific verification]</check>
<step>Result of check 1</step>
<check>Check 2: [another verification]</check>
<step>Result of check 2</step>
<doubt>What assumptions am I making that could be false?</doubt>
<step>Identify and test assumptions</step>
<step>Revised answer if needed</step>
</think>
**Answer**: [Final answer after error checking]

Requirements:
- Show initial thinking that COULD have errors
- Actively question the first answer
- Perform specific checks (not vague "double-check")
- Catch actual errors when present
- Demonstrate intellectual humility
- Show correction process if initial answer was wrong

Generate 1 high-quality example now.
""",

    "bayesian": """
Generate a Bayesian rationalist reasoning example.

Format:
**Prompt**: [A problem involving probability, uncertainty, or belief updating]
**Reasoning**:
<think>
<bayesian>I need to update my beliefs based on evidence using Bayes' theorem</bayesian>
<prior>Prior probability: P(Hypothesis) = [value]</prior>
<step>Likelihood: P(Evidence | Hypothesis) = [value]</step>
<step>Base rate: P(Evidence) = [calculation]</step>
<step>Bayes' theorem: P(Hypothesis | Evidence) = P(Evidence | Hypothesis) × P(Hypothesis) / P(Evidence)</step>
<posterior>Posterior probability: P(Hypothesis | Evidence) = [value]</posterior>
<step>Interpretation: [what the posterior means]</step>
<doubt>Common pitfall: base rate neglect - many people would intuitively say [wrong answer]</doubt>
</think>
**Answer**: [Probabilistically correct answer with posterior probability]

Requirements:
- Use actual Bayes' theorem with numbers
- Show prior, likelihood, and posterior calculations
- Explain base rate (P(Evidence)) calculation
- Address common probabilistic fallacies (base rate neglect, conjunction fallacy, etc.)
- Include intuitive explanation of counterintuitive results

Generate 1 high-quality example now.
""",

    "multidomain": """
Generate a multidomain consultant synthesis example.

Format:
**Prompt**: [A complex decision or strategic problem]
**Reasoning**:
<think>
<multidomain>How would experts from multiple domains approach this problem?</multidomain>
<expert>Domain 1: [specific field]</expert>
<step>Key framework/principle from domain 1</step>
<step>Application to the problem</step>
<expert>Domain 2: [different field]</expert>
<step>Key framework/principle from domain 2</step>
<step>Application to the problem</step>
<expert>Domain 3: [another different field]</expert>
<step>Key framework/principle from domain 3</step>
<step>Application to the problem</step>
<synthesis>Integration: Where do these expert perspectives agree/disagree?</synthesis>
<step>Synthesized recommendation based on multidomain analysis</step>
</think>
**Answer**: [Decision/recommendation synthesizing all expert perspectives]

Requirements:
- Use 3-5 DIFFERENT expert domains
- Each domain contributes a SPECIFIC framework or principle
- Show where perspectives align vs. conflict
- Synthesize into coherent recommendation
- Demonstrate how multiple lenses provide better decisions than single domain

Generate 1 high-quality example now.
"""
}


class FrontierModelGenerator:
    """Generate reasoning examples using frontier models via OpenRouter"""

    def __init__(self, api_key: str, wandb_enabled: bool = True):
        """
        Initialize generator

        Args:
            api_key: OpenRouter API key
            wandb_enabled: Whether to log to Weights & Biases
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.wandb_enabled = wandb_enabled

        if wandb_enabled:
            wandb.init(
                project="agent-forge-v2-phase3",
                name=f"frontier_data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['phase3', 'data_generation', 'frontier_models']
            )

    async def call_model(
        self,
        model_config: FrontierModelConfig,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> Optional[str]:
        """
        Call a frontier model via OpenRouter API

        Args:
            model_config: Model configuration
            prompt: Generation prompt
            session: aiohttp session

        Returns:
            Generated text or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/agent-forge-v2",
            "X-Title": "Agent Forge V2 - Phase 3 Data Generation"
        }

        payload = {
            "model": model_config.openrouter_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert reasoning instructor creating high-quality training examples for AI reasoning enhancement. Generate examples that demonstrate advanced reasoning patterns."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p
        }

        try:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    print(f"❌ Error from {model_config.name}: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"❌ Exception calling {model_config.name}: {str(e)}")
            return None

    def parse_example(self, text: str, strategy: str, model: str) -> Optional[ReasoningExample]:
        """
        Parse generated text into ReasoningExample

        Args:
            text: Generated text from model
            strategy: Reasoning strategy used
            model: Model name

        Returns:
            ReasoningExample or None if parsing failed
        """
        try:
            # Extract sections
            if "**Prompt**:" in text and "**Reasoning**:" in text and "**Answer**:" in text:
                parts = text.split("**Prompt**:")
                if len(parts) < 2:
                    return None

                prompt_and_rest = parts[1].split("**Reasoning**:")
                if len(prompt_and_rest) < 2:
                    return None

                prompt = prompt_and_rest[0].strip()

                reasoning_and_answer = prompt_and_rest[1].split("**Answer**:")
                if len(reasoning_and_answer) < 2:
                    return None

                reasoning = reasoning_and_answer[0].strip()
                answer = reasoning_and_answer[1].strip()

                return ReasoningExample(
                    prompt=prompt,
                    reasoning=reasoning,
                    answer=answer,
                    strategy=strategy,
                    model=model,
                    timestamp=datetime.now().isoformat(),
                    quality_score=0.0  # Will be scored later
                )
            else:
                return None
        except Exception as e:
            print(f"❌ Parse error: {str(e)}")
            return None

    async def generate_examples_for_strategy(
        self,
        model_config: FrontierModelConfig,
        strategy: str,
        num_examples: int,
        session: aiohttp.ClientSession
    ) -> List[ReasoningExample]:
        """
        Generate examples for a specific strategy using a specific model

        Args:
            model_config: Model configuration
            strategy: Reasoning strategy
            num_examples: Number of examples to generate
            session: aiohttp session

        Returns:
            List of ReasoningExample
        """
        strategy_info = REASONING_STRATEGIES[strategy]
        prompt_template = STRATEGY_PROMPTS[strategy]

        examples = []

        print(f"\n🔄 Generating {num_examples} {strategy_info['name']} examples using {model_config.name}...")

        # Generate in batches of 10 (to avoid rate limits)
        batch_size = 10
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_tasks = []

            for i in range(batch_start, batch_end):
                task = self.call_model(model_config, prompt_template, session)
                batch_tasks.append(task)

            # Wait for batch
            batch_results = await asyncio.gather(*batch_tasks)

            # Parse results
            for result in batch_results:
                if result:
                    example = self.parse_example(result, strategy, model_config.name)
                    if example:
                        examples.append(example)

            print(f"  ✅ Batch {batch_start}-{batch_end}: {len([r for r in batch_results if r])} successful")

            # Small delay between batches
            await asyncio.sleep(1)

        print(f"✅ Generated {len(examples)}/{num_examples} {strategy_info['name']} examples using {model_config.name}")

        return examples

    async def generate_all_examples(self) -> List[ReasoningExample]:
        """
        Generate all examples from all models for all strategies

        Returns:
            List of all ReasoningExample
        """
        all_examples = []
        total_cost = 0.0

        async with aiohttp.ClientSession() as session:
            for model_key, model_config in FRONTIER_MODELS.items():
                print(f"\n{'='*80}")
                print(f"🤖 MODEL: {model_config.name}")
                print(f"{'='*80}")

                for strategy_key, strategy_info in REASONING_STRATEGIES.items():
                    num_examples = strategy_info['examples_per_model']

                    examples = await self.generate_examples_for_strategy(
                        model_config,
                        strategy_key,
                        num_examples,
                        session
                    )

                    all_examples.extend(examples)

                    # Estimate cost (rough)
                    # Assume ~1000 tokens per example (500 input + 500 output)
                    estimated_tokens = len(examples) * 1000
                    cost = (estimated_tokens / 1000) * model_config.cost_per_1k_tokens
                    total_cost += cost

                    if self.wandb_enabled:
                        wandb.log({
                            f'generation/{model_key}/{strategy_key}/examples_generated': len(examples),
                            f'generation/{model_key}/{strategy_key}/cost_usd': cost
                        })

        print(f"\n{'='*80}")
        print(f"🎉 GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total examples generated: {len(all_examples)}")
        print(f"Total estimated cost: ${total_cost:.2f}")

        if self.wandb_enabled:
            wandb.log({
                'generation/total_examples': len(all_examples),
                'generation/total_cost_usd': total_cost
            })

        return all_examples

    def deduplicate_examples(self, examples: List[ReasoningExample]) -> List[ReasoningExample]:
        """
        Deduplicate examples based on content hash

        Args:
            examples: List of ReasoningExample

        Returns:
            Deduplicated list
        """
        seen_hashes = set()
        deduplicated = []

        for example in examples:
            if example.hash not in seen_hashes:
                seen_hashes.add(example.hash)
                deduplicated.append(example)

        print(f"\n🔍 Deduplication: {len(examples)} → {len(deduplicated)} ({len(examples) - len(deduplicated)} duplicates removed)")

        if self.wandb_enabled:
            wandb.log({
                'deduplication/original_count': len(examples),
                'deduplication/deduplicated_count': len(deduplicated),
                'deduplication/duplicates_removed': len(examples) - len(deduplicated)
            })

        return deduplicated

    def save_examples(self, examples: List[ReasoningExample], output_path: str):
        """
        Save examples to JSON file

        Args:
            examples: List of ReasoningExample
            output_path: Output file path
        """
        data = {
            'metadata': {
                'total_examples': len(examples),
                'generation_date': datetime.now().isoformat(),
                'models_used': list(FRONTIER_MODELS.keys()),
                'strategies': list(REASONING_STRATEGIES.keys())
            },
            'examples': [asdict(ex) for ex in examples]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved {len(examples)} examples to {output_path}")

        if self.wandb_enabled:
            # Save as W&B artifact
            artifact = wandb.Artifact(
                name='phase3_reasoning_examples',
                type='dataset',
                description=f'Reasoning examples generated by {len(FRONTIER_MODELS)} frontier models'
            )
            artifact.add_file(output_path)
            wandb.log_artifact(artifact)
            print(f"📊 Uploaded to W&B as artifact 'phase3_reasoning_examples'")


async def main():
    """Main execution function"""

    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Initialize generator
    generator = FrontierModelGenerator(api_key=api_key, wandb_enabled=True)

    # Generate all examples
    print("🚀 Starting frontier model data generation for Phase 3...")
    print(f"📊 Target: 5 models × 8 strategies × 500 examples = 20,000 examples")
    print(f"⏱️  Estimated time: ~2-3 hours (with rate limiting)")
    print(f"💰 Estimated cost: $50-100 (varies by model usage)")

    all_examples = await generator.generate_all_examples()

    # Deduplicate
    deduplicated = generator.deduplicate_examples(all_examples)

    # Save
    output_path = "data/phase3_reasoning_examples.json"
    os.makedirs("data", exist_ok=True)
    generator.save_examples(deduplicated, output_path)

    # Statistics
    print(f"\n📈 FINAL STATISTICS:")
    print(f"{'='*80}")

    # By strategy
    print("\nExamples per strategy:")
    for strategy in REASONING_STRATEGIES.keys():
        count = len([ex for ex in deduplicated if ex.strategy == strategy])
        print(f"  {strategy:20s}: {count:4d} examples")

    # By model
    print("\nExamples per model:")
    for model in FRONTIER_MODELS.values():
        count = len([ex for ex in deduplicated if ex.model == model.name])
        print(f"  {model.name:25s}: {count:4d} examples")

    print(f"\n✅ Data generation complete!")

    if generator.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
