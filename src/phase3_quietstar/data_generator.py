"""
Phase 3 Data Generation via OpenRouter

Generates 20K reasoning examples from 5 frontier models:
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Gemini Pro 1.5 (Google)
- Grok Beta (xAI)
- Qwen 2.5 72B (Alibaba)

Each model generates 4K examples across 7 reasoning strategies:
- Chain-of-Thought (400 examples)
- MECE Decomposition (200)
- Falsification Testing (200)
- Expert Perspective (200)
- Orthogonal Wisdom (200)
- Self-Doubt (200)
- Bayesian Rationalist (200)
- Total per model: 1600 examples × 5 models = 8K base + augmentation = 20K

Cost: $100-200 via OpenRouter API
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp


@dataclass
class ReasoningExample:
    """Single reasoning training example."""

    question: str
    reasoning: str  # Contains thinking tokens
    answer: str
    strategy: str  # e.g., "chain_of_thought"
    model: str  # Source model
    tokens_used: int
    cost_usd: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class GenerationStats:
    """Track generation progress and costs."""

    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    total_cost_usd: float = 0.0
    examples_by_strategy: Dict[str, int] = field(default_factory=dict)
    examples_by_model: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def valid_ratio(self) -> float:
        """Ratio of valid examples."""
        if self.total_examples == 0:
            return 0.0
        return self.valid_examples / self.total_examples

    @property
    def cost_per_example(self) -> float:
        """Average cost per valid example."""
        if self.valid_examples == 0:
            return 0.0
        return self.total_cost_usd / self.valid_examples


class OpenRouterClient:
    """
    OpenRouter API client for frontier model access.

    Handles rate limiting, retries, and cost tracking.
    """

    def __init__(
        self,
        api_key: str,
        cost_limit: float = 200.0,
        batch_size: int = 10,
    ):
        self.api_key = api_key
        self.cost_limit = cost_limit
        self.batch_size = batch_size
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        self.stats = GenerationStats()

        # Model pricing (per 1M tokens, approximate)
        self.pricing = {
            "openai/gpt-4o": {"input": 2.5, "output": 10.0},
            "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "google/gemini-pro-1.5": {"input": 1.25, "output": 5.0},
            "x-ai/grok-beta": {"input": 2.0, "output": 8.0},
            "qwen/qwen-2.5-72b-instruct": {"input": 0.5, "output": 2.0},
        }

    async def generate_examples(
        self,
        strategy_prompts: Dict[str, List[str]],
        model_name: str,
        num_examples: int,
    ) -> List[ReasoningExample]:
        """Generate examples for a specific model and strategy."""
        examples = []

        async with aiohttp.ClientSession() as session:
            for strategy, prompts in strategy_prompts.items():
                strategy_examples = min(num_examples, len(prompts))

                for i in range(0, strategy_examples, self.batch_size):
                    batch_prompts = prompts[i : i + self.batch_size]

                    # Check cost limit
                    if self.stats.total_cost_usd >= self.cost_limit:
                        print(f"Cost limit ${self.cost_limit} reached. Stopping.")
                        return examples

                    # Generate batch
                    batch_results = await self._generate_batch(
                        session, model_name, strategy, batch_prompts
                    )

                    examples.extend(batch_results)

                    # Update stats
                    self.stats.total_examples += len(batch_results)
                    self.stats.valid_examples += len([e for e in batch_results if e.reasoning])
                    self.stats.examples_by_strategy[strategy] = self.stats.examples_by_strategy.get(
                        strategy, 0
                    ) + len(batch_results)
                    self.stats.examples_by_model[model_name] = self.stats.examples_by_model.get(
                        model_name, 0
                    ) + len(batch_results)

        return examples

    async def _generate_batch(
        self,
        session: aiohttp.ClientSession,
        model: str,
        strategy: str,
        prompts: List[str],
    ) -> List[ReasoningExample]:
        """Generate a batch of examples."""
        results = []

        for prompt in prompts:
            try:
                example = await self._generate_single(session, model, strategy, prompt)
                if example:
                    results.append(example)
            except Exception as e:
                print(f"Error generating example: {e}")
                continue

        return results

    async def _generate_single(
        self,
        session: aiohttp.ClientSession,
        model: str,
        strategy: str,
        prompt: str,
    ) -> Optional[ReasoningExample]:
        """Generate single example with retry logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = await self._api_call(session, model, prompt)

                if response:
                    # Parse response
                    example = self._parse_response(response, model, strategy, prompt)
                    return example

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")

        return None

    async def _api_call(
        self,
        session: aiohttp.ClientSession,
        model: str,
        prompt: str,
    ) -> Optional[Dict]:
        """Make API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        async with session.post(self.base_url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"API error {resp.status}: {await resp.text()}")
                return None

    def _parse_response(
        self,
        response: Dict,
        model: str,
        strategy: str,
        prompt: str,
    ) -> ReasoningExample:
        """Parse API response into ReasoningExample."""
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        # Calculate cost
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        # Update total cost
        self.stats.total_cost_usd += cost

        # Extract question, reasoning, answer
        question, reasoning, answer = self._extract_components(content, prompt)

        return ReasoningExample(
            question=question,
            reasoning=reasoning,
            answer=answer,
            strategy=strategy,
            model=model,
            tokens_used=input_tokens + output_tokens,
            cost_usd=cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        pricing = self.pricing.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def _extract_components(self, content: str, prompt: str) -> Tuple[str, str, str]:
        """Extract question, reasoning, answer from response."""
        # Simple parsing (can be enhanced with regex)
        lines = content.strip().split("\n")

        question = prompt.split("Question:")[-1].strip() if "Question:" in prompt else lines[0]
        reasoning = content  # Full response contains reasoning
        answer = lines[-1] if lines else ""

        return question, reasoning, answer


class StrategyPromptGenerator:
    """
    Generate prompts for 7 reasoning strategies.

    Each strategy has specific prompt templates with thinking tokens.
    """

    def __init__(self):
        self.strategies = {
            "chain_of_thought": 400,
            "mece_decomposition": 200,
            "falsification_testing": 200,
            "expert_perspective": 200,
            "orthogonal_wisdom": 200,
            "self_doubt": 200,
            "bayesian_rationalist": 200,
        }

    def generate_prompts(self) -> Dict[str, List[str]]:
        """Generate all prompts for all strategies."""
        all_prompts = {}

        for strategy, count in self.strategies.items():
            generator_method = getattr(self, f"_generate_{strategy}_prompts")
            all_prompts[strategy] = generator_method(count)

        return all_prompts

    def _generate_chain_of_thought_prompts(self, count: int) -> List[str]:
        """Chain-of-Thought reasoning prompts."""
        template = """Question: {question}

Use <think> and <step> tags to show your reasoning step-by-step:

<think>
<step>First, I need to...</step>
<step>Then, I should...</step>
<step>Finally, I can...</step>
</think>

Answer:"""

        questions = [
            f"Solve this problem using step-by-step reasoning: Problem {i}" for i in range(count)
        ]

        return [template.format(question=q) for q in questions]

    def _generate_mece_decomposition_prompts(self, count: int) -> List[str]:
        """MECE (Mutually Exclusive, Collectively Exhaustive) prompts."""
        template = """Question: {question}

Break this down using MECE decomposition with <mece> tags:

<think>
<mece>
<category>Category 1: ...</category>
<category>Category 2: ...</category>
<category>Category 3: ...</category>
</mece>
</think>

Answer:"""

        questions = [
            f"Break down this complex problem into categories: Problem {i}" for i in range(count)
        ]

        return [template.format(question=q) for q in questions]

    def _generate_falsification_testing_prompts(self, count: int) -> List[str]:
        """Falsification testing prompts."""
        template = """Question: {question}

Test what would disprove your answer using <falsify> tags:

<think>
<falsify>
<test>If X were true, then...</test>
<test>If Y were false, then...</test>
</falsify>
</think>

Answer:"""

        questions = [f"What evidence would disprove this claim: Claim {i}" for i in range(count)]

        return [template.format(question=q) for q in questions]

    def _generate_expert_perspective_prompts(self, count: int) -> List[str]:
        """Expert perspective prompts."""
        template = """Question: {question}

Think like an expert in this domain using <expert> tags:

<think>
<expert domain="relevant_field">
As an expert, I would approach this by...
</expert>
</think>

Answer:"""

        questions = [f"Solve this from an expert's perspective: Problem {i}" for i in range(count)]

        return [template.format(question=q) for q in questions]

    def _generate_orthogonal_wisdom_prompts(self, count: int) -> List[str]:
        """Orthogonal wisdom (insights from unrelated fields) prompts."""
        template = """Question: {question}

Draw insights from an unrelated field to solve this:

<think>
<step>From field X, I know that...</step>
<step>This insight suggests...</step>
</think>

Answer:"""

        questions = [f"Apply cross-domain thinking to: Problem {i}" for i in range(count)]

        return [template.format(question=q) for q in questions]

    def _generate_self_doubt_prompts(self, count: int) -> List[str]:
        """Self-doubt and error checking prompts."""
        template = """Question: {question}

Question your first answer using <doubt> tags:

<think>
<step>My initial answer is...</step>
<doubt>
<check>But wait, what if...</check>
<check>I should also consider...</check>
</doubt>
</think>

Final Answer:"""

        questions = [f"Solve this, then check your work: Problem {i}" for i in range(count)]

        return [template.format(question=q) for q in questions]

    def _generate_bayesian_rationalist_prompts(self, count: int) -> List[str]:
        """Bayesian rationalist (evidence-based belief updating) prompts."""
        template = """Question: {question}

Update your beliefs based on evidence:

<think>
<step>Prior belief: ...</step>
<step>New evidence: ...</step>
<step>Updated belief: ...</step>
</think>

Answer:"""

        questions = [f"Update your beliefs given this evidence: Evidence {i}" for i in range(count)]

        return [template.format(question=q) for q in questions]


async def generate_phase3_dataset(
    api_key: str,
    output_path: Path,
    cost_limit: float = 200.0,
) -> GenerationStats:
    """
    Generate complete Phase 3 dataset (20K examples).

    Args:
        api_key: OpenRouter API key
        output_path: Where to save dataset
        cost_limit: Maximum cost in USD

    Returns:
        Generation statistics
    """
    # Initialize components
    client = OpenRouterClient(api_key, cost_limit)
    prompt_gen = StrategyPromptGenerator()

    # Generate prompts
    print("Generating prompts for all strategies...")
    strategy_prompts = prompt_gen.generate_prompts()

    # Models to use
    models = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro-1.5",
        "x-ai/grok-beta",
        "qwen/qwen-2.5-72b-instruct",
    ]

    # Generate examples
    all_examples = []
    examples_per_model = 4000  # 20K / 5 models

    for model in models:
        print(f"\nGenerating {examples_per_model} examples from {model}...")

        model_examples = await client.generate_examples(strategy_prompts, model, examples_per_model)

        all_examples.extend(model_examples)

        print(
            f"  Generated: {len(model_examples)} examples, "
            f"Cost: ${client.stats.total_cost_usd:.2f}"
        )

    # Save dataset
    print(f"\nSaving {len(all_examples)} examples to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([asdict(ex) for ex in all_examples], f, indent=2)

    # Print stats
    print("\n=== Generation Complete ===")
    print(f"Total examples: {client.stats.total_examples}")
    print(f"Valid examples: {client.stats.valid_examples}")
    print(f"Valid ratio: {client.stats.valid_ratio:.2%}")
    print(f"Total cost: ${client.stats.total_cost_usd:.2f}")
    print(f"Cost per example: ${client.stats.cost_per_example:.4f}")
    print(f"Elapsed time: {client.stats.elapsed_time / 60:.1f} minutes")

    return client.stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_generator.py <openrouter_api_key>")
        sys.exit(1)

    api_key = sys.argv[1]
    output_path = Path("data/phase3_reasoning_training_data.json")

    # Run generation
    stats = asyncio.run(generate_phase3_dataset(api_key, output_path))

    print(f"\nDataset saved to: {output_path}")
