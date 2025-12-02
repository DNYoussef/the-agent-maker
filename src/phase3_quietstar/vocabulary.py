"""
Phase 3 Vocabulary Extension

Adds 8 special thinking tokens to model vocabulary:
1. <think> / </think> - Wrapper for thinking block
2. <step> - Individual reasoning step
3. <reason> - Explicit reasoning statement
4. <mece> - MECE decomposition
5. <falsify> - Falsification testing
6. <expert> - Expert perspective
7. <doubt> - Self-doubt/error checking
8. Plus variants for flexibility

Total: 8 core tokens for reasoning enhancement
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


class ThinkingVocabulary:
    """
    Manages special thinking tokens for Phase 3.

    Provides token IDs, embeddings initialization,
    and validation utilities.
    """

    # Core 8 thinking tokens
    CORE_TOKENS = [
        "<think>",
        "</think>",
        "<step>",
        "<reason>",
        "<mece>",
        "<falsify>",
        "<expert>",
        "<doubt>",
    ]

    # Optional extended tokens for richer reasoning
    EXTENDED_TOKENS = [
        "<bayesian>",
        "<multidomain>",
        "<correct>",
        "<uncertain>",
    ]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        use_extended: bool = False,
    ):
        """
        Initialize thinking vocabulary.

        Args:
            tokenizer: Base tokenizer to extend
            use_extended: Whether to include extended tokens
        """
        self.tokenizer = tokenizer
        self.use_extended = use_extended

        # Determine token set
        self.thinking_tokens = (
            self.CORE_TOKENS + self.EXTENDED_TOKENS if use_extended else self.CORE_TOKENS
        )

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.original_vocab_size = len(tokenizer)

    def add_tokens(self) -> int:
        """
        Add thinking tokens to tokenizer.

        Returns:
            Number of tokens added
        """
        # Add special tokens
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.thinking_tokens}
        )

        # Build token mappings
        for token in self.thinking_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        return num_added

    def resize_embeddings(self, model: nn.Module) -> Tuple[nn.Embedding, nn.Linear]:
        """
        Resize model embeddings to accommodate new tokens.

        Args:
            model: Model with token embeddings and lm_head

        Returns:
            Tuple of (new_embeddings, new_lm_head)
        """
        old_embeddings = model.get_input_embeddings()
        old_lm_head = model.get_output_embeddings()

        old_vocab_size = old_embeddings.weight.size(0)
        new_vocab_size = len(self.tokenizer)

        if old_vocab_size == new_vocab_size:
            return old_embeddings, old_lm_head

        # Create new embeddings
        embedding_dim = old_embeddings.embedding_dim
        new_embeddings = nn.Embedding(
            new_vocab_size, embedding_dim, padding_idx=old_embeddings.padding_idx
        )

        # Copy old weights
        with torch.no_grad():
            new_embeddings.weight[:old_vocab_size] = old_embeddings.weight

            # Initialize new tokens (average of existing embeddings)
            mean_embedding = old_embeddings.weight.mean(dim=0)
            std_embedding = old_embeddings.weight.std(dim=0)

            for i in range(old_vocab_size, new_vocab_size):
                new_embeddings.weight[i] = torch.normal(mean_embedding, std_embedding)

        # Create new LM head
        hidden_size = old_lm_head.in_features
        new_lm_head = nn.Linear(hidden_size, new_vocab_size, bias=old_lm_head.bias is not None)

        # Copy old weights
        with torch.no_grad():
            new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
            if new_lm_head.bias is not None:
                new_lm_head.bias[:old_vocab_size] = old_lm_head.bias

            # Initialize new tokens
            mean_weight = old_lm_head.weight.mean(dim=0)
            std_weight = old_lm_head.weight.std(dim=0)

            for i in range(old_vocab_size, new_vocab_size):
                new_lm_head.weight[i] = torch.normal(mean_weight, std_weight)
                if new_lm_head.bias is not None:
                    new_lm_head.bias[i] = 0.0

        # Update model
        model.set_input_embeddings(new_embeddings)
        model.set_output_embeddings(new_lm_head)

        return new_embeddings, new_lm_head

    def get_token_id(self, token: str) -> Optional[int]:
        """Get token ID for a thinking token."""
        return self.token_to_id.get(token)

    def get_token(self, token_id: int) -> Optional[str]:
        """Get token string from ID."""
        return self.id_to_token.get(token_id)

    def is_thinking_token(self, token_id: int) -> bool:
        """Check if token ID is a thinking token."""
        return token_id in self.id_to_token

    def validate_tokens(self) -> bool:
        """
        Validate all thinking tokens are in tokenizer.

        Returns:
            True if all tokens present
        """
        for token in self.thinking_tokens:
            if token not in self.tokenizer.get_vocab():
                return False
        return True

    def get_stats(self) -> Dict[str, int]:
        """Get vocabulary statistics."""
        return {
            "original_vocab_size": self.original_vocab_size,
            "new_vocab_size": len(self.tokenizer),
            "thinking_tokens_added": len(self.thinking_tokens),
            "core_tokens": len(self.CORE_TOKENS),
            "extended_tokens": (len(self.EXTENDED_TOKENS) if self.use_extended else 0),
        }

    def format_with_thinking(self, text: str, strategy: str = "chain_of_thought") -> str:
        """
        Wrap text with appropriate thinking tokens.

        Args:
            text: Text to wrap
            strategy: Reasoning strategy to use

        Returns:
            Formatted text with thinking tokens
        """
        if strategy == "chain_of_thought":
            return f"<think>\n<step>{text}</step>\n</think>"

        elif strategy == "mece_decomposition":
            return f"<think>\n<mece>{text}</mece>\n</think>"

        elif strategy == "falsification_testing":
            return f"<think>\n<falsify>{text}</falsify>\n</think>"

        elif strategy == "expert_perspective":
            return f"<think>\n<expert>{text}</expert>\n</think>"

        elif strategy == "self_doubt":
            return f"<think>\n<doubt>{text}</doubt>\n</think>"

        elif strategy == "bayesian_rationalist" and self.use_extended:
            return f"<think>\n<bayesian>{text}</bayesian>\n</think>"

        else:
            return f"<think>\n<reason>{text}</reason>\n</think>"

    def extract_thinking_content(self, text: str) -> List[str]:
        """
        Extract content between thinking tokens.

        Args:
            text: Text containing thinking tokens

        Returns:
            List of extracted thinking blocks
        """
        import re

        # Pattern for thinking blocks
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, text, re.DOTALL)

        return [match.strip() for match in matches]

    def count_thinking_tokens(self, token_ids: torch.Tensor) -> int:
        """
        Count thinking tokens in a sequence.

        Args:
            token_ids: (batch, seq_len) or (seq_len,)

        Returns:
            Count of thinking tokens
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        count = 0
        for token_id in token_ids.flatten().tolist():
            if self.is_thinking_token(token_id):
                count += 1

        return count


def prepare_model_for_phase3(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    use_extended_tokens: bool = False,
) -> Tuple[nn.Module, PreTrainedTokenizer, ThinkingVocabulary]:
    """
    Prepare model for Phase 3 by adding thinking tokens.

    Args:
        model: Base model
        tokenizer: Base tokenizer
        use_extended_tokens: Whether to add extended tokens

    Returns:
        Tuple of (updated_model, updated_tokenizer, vocabulary)
    """
    # Create vocabulary manager
    vocab = ThinkingVocabulary(tokenizer, use_extended=use_extended_tokens)

    # Add tokens
    num_added = vocab.add_tokens()
    print(f"Added {num_added} thinking tokens to vocabulary")

    # Resize embeddings
    vocab.resize_embeddings(model)
    print(f"Resized embeddings: {vocab.original_vocab_size} → {len(tokenizer)}")

    # Validate
    if not vocab.validate_tokens():
        raise ValueError("Failed to add all thinking tokens")

    # Print stats
    stats = vocab.get_stats()
    print(f"Vocabulary stats: {stats}")

    return model, tokenizer, vocab


def compute_thinking_token_usage(outputs: List[str], vocab: ThinkingVocabulary) -> Dict[str, float]:
    """
    Compute thinking token usage statistics.

    Args:
        outputs: List of model outputs
        vocab: Thinking vocabulary

    Returns:
        Dictionary with usage metrics
    """
    total_outputs = len(outputs)

    # Count usage
    thinking_tag_count = 0
    step_tag_count = 0
    mece_tag_count = 0
    falsify_tag_count = 0
    doubt_tag_count = 0

    for output in outputs:
        if "<think>" in output:
            thinking_tag_count += 1
        if "<step>" in output:
            step_tag_count += 1
        if "<mece>" in output:
            mece_tag_count += 1
        if "<falsify>" in output:
            falsify_tag_count += 1
        if "<doubt>" in output:
            doubt_tag_count += 1

    return {
        "thinking_tag_usage": thinking_tag_count / total_outputs,
        "step_tag_usage": step_tag_count / total_outputs,
        "mece_tag_usage": mece_tag_count / total_outputs,
        "falsify_tag_usage": falsify_tag_count / total_outputs,
        "doubt_tag_usage": doubt_tag_count / total_outputs,
        "overall_usage": (
            thinking_tag_count
            + step_tag_count
            + mece_tag_count
            + falsify_tag_count
            + doubt_tag_count
        )
        / (total_outputs * 5),  # 5 types
    }
