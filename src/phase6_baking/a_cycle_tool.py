"""
Phase 6: A-Cycle Tool Optimizer

Optimizes tool use capabilities via SWE-Bench style tasks.
Bakes tool-use prompts into model weights for improved tool calling.

Research: "Prompt Baking" (arXiv:2409.13697v1)

M4 TIER 1: Integrated with SWEBenchEvaluator for real code generation evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swe_bench_eval import SWEBenchEvaluator, SWEBenchTask


@dataclass
class ToolTask:
    """A tool-use task for evaluation."""
    description: str
    expected_tools: List[str]
    ground_truth: str
    difficulty: int  # 1-10


class SWEBenchToolEvaluator:
    """
    Adapter for SWEBenchEvaluator to work with A-cycle optimization.

    Wraps SWEBenchEvaluator to provide an evaluate(model) interface
    that returns a composite score for tool/code generation ability.

    M4 TIER 1: Integrates real SWE-Bench evaluation into Phase 6.

    Usage:
        evaluator = SWEBenchToolEvaluator(
            data_path="data/swe_bench_lite.json",
            max_tasks=20,
            tokenizer=tokenizer
        )
        score = evaluator.evaluate(model)  # Returns 0.0-1.0
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_tasks: int = 20,
        tokenizer: Any = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize SWE-Bench tool evaluator.

        Args:
            data_path: Path to SWE-Bench JSON data (None = use synthetic)
            max_tasks: Maximum tasks per evaluation (controls cost/time)
            tokenizer: Tokenizer for encoding/decoding
            max_new_tokens: Max tokens to generate per task
            temperature: Generation temperature
            verbose: Print evaluation progress
        """
        self.swe_bench = SWEBenchEvaluator(data_path=data_path)
        self.max_tasks = max_tasks
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.verbose = verbose
        self._tasks_loaded = False
        self._last_metrics: Dict[str, float] = {}

    def evaluate(self, model: nn.Module) -> float:
        """
        Evaluate model's code generation ability on SWE-Bench tasks.

        Args:
            model: Model to evaluate (must support generate() or forward())

        Returns:
            Composite score from 0.0 to 1.0 based on:
            - 40% exact match
            - 30% partial match (line overlap)
            - 20% syntax validity
            - 10% semantic similarity
        """
        # Load tasks on first call
        if not self._tasks_loaded:
            self.swe_bench.load_tasks(max_tasks=self.max_tasks)
            self._tasks_loaded = True

        # Reset for fresh evaluation
        self.swe_bench.reset()

        # Create generation function that wraps model
        def generate_fn(problem_statement: str) -> str:
            return self._generate_code(model, problem_statement)

        # Run evaluation
        metrics = self.swe_bench.run_evaluation(
            generate_fn=generate_fn,
            max_tasks=self.max_tasks,
            verbose=self.verbose
        )

        self._last_metrics = metrics
        return metrics.get('composite_score', 0.0)

    def _generate_code(self, model: nn.Module, problem_statement: str) -> str:
        """Generate code solution for a problem."""
        model.eval()

        # Format prompt for code generation
        prompt = self._format_code_prompt(problem_statement)

        try:
            with torch.no_grad():
                # Tokenize
                if self.tokenizer and hasattr(self.tokenizer, '__call__'):
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )
                else:
                    # Fallback for mock tokenizers
                    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()
                          if isinstance(v, torch.Tensor)}

                # Generate
                if hasattr(model, 'generate'):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=max(0.1, self.temperature),
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer and hasattr(self.tokenizer, 'pad_token_id') else 0
                    )
                    if self.tokenizer and hasattr(self.tokenizer, 'decode'):
                        generated = self.tokenizer.decode(
                            outputs[0][inputs['input_ids'].size(1):],
                            skip_special_tokens=True
                        )
                    else:
                        generated = ""
                else:
                    # Forward-only model - extract code from logits
                    outputs = model(**inputs)
                    if hasattr(outputs, 'logits'):
                        # Greedy decode
                        predicted_ids = outputs.logits.argmax(dim=-1)
                        if self.tokenizer and hasattr(self.tokenizer, 'decode'):
                            generated = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                        else:
                            generated = ""
                    else:
                        generated = ""

                return self._extract_code(generated)

        except Exception as e:
            if self.verbose:
                print(f"    Generation error: {e}")
            return ""

    def _format_code_prompt(self, problem_statement: str) -> str:
        """Format problem into code generation prompt."""
        return f"""You are a skilled software engineer. Fix the following issue:

{problem_statement}

Provide ONLY the code fix (no explanation):
```python
"""

    def _extract_code(self, text: str) -> str:
        """Extract code from generated text."""
        # Look for code blocks
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Return as-is if no code blocks
        return text.strip()

    def get_last_metrics(self) -> Dict[str, float]:
        """Get detailed metrics from last evaluation."""
        return self._last_metrics.copy()

    def get_failed_tasks(self, threshold: float = 0.5) -> List[Tuple[SWEBenchTask, Any]]:
        """Get tasks where model scored below threshold."""
        return self.swe_bench.get_failed_tasks(threshold)


class ACycleOptimizer:
    """
    A-Cycle: Tool Use Optimization.

    Process:
    1. Generate tool-use tasks (SWE-Bench style)
    2. Evaluate model's tool calling ability
    3. Bake tool-use prompts to improve performance
    4. Return optimized model and score
    """

    def __init__(
        self,
        tool_prompts: List[str],
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 5e-5  # Fixed: was 1e-4, now 5e-5 per M4 spec
    ):
        """
        Initialize A-cycle optimizer.

        Args:
            tool_prompts: Prompts to bake for tool use
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            num_epochs: Baking epochs
            learning_rate: Learning rate
        """
        self.tool_prompts = tool_prompts
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.state = {
            'iterations': 0,
            'scores': [],
            'best_score': 0.0,
            'prompts_used': []
        }

        # Sample tool tasks
        self.tool_tasks = self._generate_tool_tasks()

    def _generate_tool_tasks(self) -> List[ToolTask]:
        """Generate sample tool-use tasks."""
        return [
            ToolTask(
                description="Read the contents of config.json and extract the API key",
                expected_tools=["read_file", "parse_json"],
                ground_truth="api_key_value",
                difficulty=3
            ),
            ToolTask(
                description="Search for all Python files containing 'def main'",
                expected_tools=["search_files", "grep"],
                ground_truth="list_of_files",
                difficulty=4
            ),
            ToolTask(
                description="Create a new directory and initialize a git repository",
                expected_tools=["mkdir", "git_init"],
                ground_truth="success",
                difficulty=5
            ),
            ToolTask(
                description="Run the test suite and report failures",
                expected_tools=["run_tests", "parse_output"],
                ground_truth="test_results",
                difficulty=6
            ),
            ToolTask(
                description="Debug the failing function by adding logging",
                expected_tools=["read_file", "edit_file", "run_tests"],
                ground_truth="fixed_code",
                difficulty=7
            ),
        ]

    def optimize(
        self,
        model: nn.Module,
        tokenizer: Any,
        evaluator: Any = None
    ) -> Tuple[nn.Module, float]:
        """
        Run one A-cycle optimization iteration.

        Args:
            model: Model to optimize
            tokenizer: Tokenizer
            evaluator: Optional external evaluator

        Returns:
            Tuple of (optimized_model, score)
        """
        self.state['iterations'] += 1

        # Step 1: Evaluate current tool-use ability
        pre_score = self._evaluate_tool_use(model, tokenizer, evaluator)
        print(f"    Pre-bake tool score: {pre_score:.3f}")

        # Step 2: Select prompt to bake
        prompt_idx = self.state['iterations'] % len(self.tool_prompts)
        prompt = self.tool_prompts[prompt_idx]
        self.state['prompts_used'].append(prompt)

        # Step 3: Bake the prompt
        baked_model = self._bake_tool_prompt(model, prompt, tokenizer)

        # Step 4: Evaluate post-bake
        post_score = self._evaluate_tool_use(baked_model, tokenizer, evaluator)
        print(f"    Post-bake tool score: {post_score:.3f}")

        # Update state
        self.state['scores'].append(post_score)
        if post_score > self.state['best_score']:
            self.state['best_score'] = post_score

        return baked_model, post_score

    def _evaluate_tool_use(
        self,
        model: nn.Module,
        tokenizer: Any,
        evaluator: Any = None
    ) -> float:
        """Evaluate model's tool-use ability."""
        if evaluator is not None:
            return evaluator.evaluate(model)

        # Default evaluation: score based on output coherence
        model.eval()
        total_score = 0.0

        with torch.no_grad():
            for task in self.tool_tasks:
                # Create tool-use prompt
                prompt = f"Task: {task.description}\nTools available: {', '.join(task.expected_tools)}\nPlan:"

                try:
                    # Tokenize
                    if hasattr(tokenizer, '__call__'):
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                            padding=True
                        )
                    else:
                        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()
                              if isinstance(v, torch.Tensor)}

                    # Generate
                    if hasattr(model, 'generate'):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            do_sample=False
                        )
                        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True) if hasattr(tokenizer, 'decode') else ""
                    else:
                        output_text = ""

                    # Score based on tool mention
                    tool_mentions = sum(1 for tool in task.expected_tools if tool in output_text.lower())
                    task_score = tool_mentions / len(task.expected_tools)
                    total_score += task_score

                except Exception:
                    continue

        return total_score / max(1, len(self.tool_tasks))

    def _bake_tool_prompt(
        self,
        model: nn.Module,
        prompt: str,
        tokenizer: Any
    ) -> nn.Module:
        """Bake a tool-use prompt into the model."""
        import copy
        baked_model = copy.deepcopy(model)

        device = next(baked_model.parameters()).device
        optimizer = torch.optim.AdamW(baked_model.parameters(), lr=self.learning_rate)

        # Create calibration data for tool use
        calibration_samples = [
            f"{prompt}\n\nTask: Read file config.json\nStep 1: Use read_file tool",
            f"{prompt}\n\nTask: Search for errors\nStep 1: Use grep tool to search",
            f"{prompt}\n\nTask: Run tests\nStep 1: Use run_tests tool",
        ]

        baked_model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for sample in calibration_samples:
                try:
                    if hasattr(tokenizer, '__call__'):
                        inputs = tokenizer(
                            sample,
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                            padding=True
                        )
                    else:
                        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

                    inputs = {k: v.to(device) for k, v in inputs.items()
                              if isinstance(v, torch.Tensor)}

                    outputs = baked_model(**inputs)

                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs['input_ids'][..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=0
                        )
                    else:
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(baked_model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                except Exception:
                    continue

        return baked_model

    def get_state(self) -> Dict:
        """Get optimizer state."""
        return self.state.copy()


__all__ = ['ACycleOptimizer', 'ToolTask', 'SWEBenchToolEvaluator']
