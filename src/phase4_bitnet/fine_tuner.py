"""
Fine-Tuning Pipeline
MuGrokfast-based recovery for compressed models

ISS-004: Updated to use secure SafeTensors checkpoint format.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm
from pathlib import Path
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.compressed_model import CompressedModel
from src.cross_phase.mugrokfast.optimizer import MuonGrokfast
from src.cross_phase.mugrokfast.config import MuGrokConfig
from src.cross_phase.utils.checkpoint_utils import (
    save_checkpoint as secure_save,
    load_checkpoint as secure_load,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class FineTuner:
    """
    Fine-tuning pipeline for BitNet compressed models

    Features:
    - MuGrokfast optimizer with STE mode
    - Automatic quality recovery
    - Perplexity monitoring
    - Early stopping on convergence

    Trigger: Accuracy drop > fine_tune_threshold (default 5%)
    """

    def __init__(
        self,
        model: CompressedModel,
        config: Phase4Config,
        device: str = "cuda"
    ):
        """
        Initialize fine-tuner

        Args:
            model: Compressed model to fine-tune
            config: Phase 4 configuration
            device: Device for training
        """
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # ISS-008: Mixed precision support
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if config.use_mixed_precision and device == "cuda"
            else None
        )
        self.use_amp = config.use_mixed_precision and device == "cuda"

        # ISS-008: W&B initialization
        self.wandb_enabled = config.wandb_enabled and WANDB_AVAILABLE
        if self.wandb_enabled:
            wandb.init(
                project=config.wandb_project,
                tags=config.wandb_tags + ["fine-tuning", "iss-008"],
                config={
                    "fine_tune_lr": config.fine_tune_lr,
                    "fine_tune_epochs": config.fine_tune_epochs,
                    "warmup_steps": config.warmup_steps,
                    "max_grad_norm": config.max_grad_norm,
                    "use_mixed_precision": config.use_mixed_precision,
                }
            )

        # ISS-008: Checkpoint directory setup
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_perplexity = float('inf')
        self.training_history = []
        self.global_step = 0

    def _create_optimizer(self) -> MuonGrokfast:
        """
        Create MuGrokfast optimizer with STE mode

        Returns:
            MuonGrokfast optimizer instance
        """
        # ISS-008: Use MuGrokConfig.from_phase(4) or custom config
        # MuGrokConfig requires: muon_lr, fallback_lr, grokfast_alpha,
        # grokfast_lambda, qk_clip_threshold, kl_coefficient, muon_ste_mode,
        # momentum, nesterov, ns_steps
        muon_config = MuGrokConfig(
            muon_lr=self.config.fine_tune_lr,
            fallback_lr=self.config.fine_tune_lr,
            grokfast_alpha=self.config.grokfast_ema_alpha,
            grokfast_lambda=self.config.grokfast_lambda,
            qk_clip_threshold=30.0,  # Default for Phase 4
            kl_coefficient=0.0,  # No KL regularization in fine-tuning
            muon_ste_mode=True,  # STE MODE (critical for quantized models)
            momentum=0.9,
            nesterov=True,
            ns_steps=5,
        )

        # Create optimizer
        optimizer = MuonGrokfast(
            self.model.parameters(),
            config=muon_config
        )

        return optimizer

    def _get_lr_multiplier(self, step: int) -> float:
        """
        ISS-008: Learning rate warmup multiplier

        Args:
            step: Current training step

        Returns:
            Learning rate multiplier (0.0 to 1.0)
        """
        if step < self.config.warmup_steps:
            return step / max(1, self.config.warmup_steps)
        return 1.0

    def fine_tune(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        log_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Fine-tune compressed model

        Args:
            train_dataloader: Training data
            eval_dataloader: Evaluation data (optional)
            log_callback: Logging callback (for W&B)

        Returns:
            Training results dictionary
        """
        self.model.train()

        print(f"Fine-tuning for {self.config.fine_tune_epochs} epochs")
        print(f"Optimizer: MuGrokfast (STE mode enabled)")
        print(f"Learning rate: {self.config.fine_tune_lr}")
        print(f"Grokfast lambda: {self.config.grokfast_lambda}")

        for epoch in range(self.config.fine_tune_epochs):
            self.current_epoch = epoch

            # Train one epoch
            epoch_stats = self._train_epoch(
                train_dataloader,
                log_callback
            )

            # Evaluate if dataloader provided
            if eval_dataloader is not None:
                eval_stats = self._evaluate(eval_dataloader)
                epoch_stats.update(eval_stats)

                # Track best perplexity
                if eval_stats['perplexity'] < self.best_perplexity:
                    self.best_perplexity = eval_stats['perplexity']

            # Store history
            self.training_history.append(epoch_stats)

            # ISS-008: Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = (
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            # Log
            print(
                f"Epoch {epoch + 1}/{self.config.fine_tune_epochs}: "
                f"Loss={epoch_stats['loss']:.4f}"
            )

            if 'perplexity' in epoch_stats:
                print(f"  Perplexity: {epoch_stats['perplexity']:.2f}")

            # ISS-008: W&B epoch logging
            if self.wandb_enabled:
                wandb.log({
                    'fine_tune/epoch_loss': epoch_stats['loss'],
                    'fine_tune/epoch': epoch + 1,
                    'fine_tune/best_perplexity': self.best_perplexity,
                })

        # Prepare results
        results = {
            'epochs_completed': self.config.fine_tune_epochs,
            'final_loss': self.training_history[-1]['loss'],
            'best_perplexity': self.best_perplexity,
            'training_history': self.training_history,
        }

        if eval_dataloader is not None:
            results['final_perplexity'] = (
                self.training_history[-1]['perplexity']
            )

        return results

    def _train_epoch(
        self,
        dataloader: DataLoader,
        log_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train one epoch

        Args:
            dataloader: Training dataloader
            log_callback: Logging callback

        Returns:
            Epoch statistics
        """
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get(
                "attention_mask",
                torch.ones_like(input_ids)
            ).to(self.device)

            # ISS-008: Apply learning rate warmup
            lr_multiplier = self._get_lr_multiplier(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.fine_tune_lr * lr_multiplier

            # ISS-008: Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # Language modeling
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # ISS-008: Mixed precision backward pass with gradient clipping
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': f'{current_lr:.2e}'
            })

            # ISS-008: W&B batch logging
            if self.wandb_enabled and batch_idx % 10 == 0:
                wandb.log({
                    'fine_tune/batch_loss': loss.item(),
                    'fine_tune/epoch': self.current_epoch,
                    'fine_tune/learning_rate': current_lr,
                    'fine_tune/global_step': self.global_step,
                    'fine_tune/warmup_multiplier': lr_multiplier,
                })

            # Log callback
            if log_callback is not None and batch_idx % 10 == 0:
                log_callback({
                    'epoch': self.current_epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'learning_rate': current_lr,
                })

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'epoch': self.current_epoch,
            'loss': avg_loss,
            'num_batches': num_batches,
        }

    def _evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model

        Args:
            dataloader: Evaluation dataloader

        Returns:
            Evaluation statistics
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get(
                    "attention_mask",
                    torch.ones_like(input_ids)
                ).to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                loss = outputs.loss if hasattr(
                    outputs, 'loss'
                ) else outputs[0]

                # Accumulate
                batch_size = input_ids.size(0)
                seq_length = input_ids.size(1)

                total_loss += loss.item() * batch_size * seq_length
                total_tokens += batch_size * seq_length

        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
        }

    def should_fine_tune(
        self,
        pre_perplexity: float,
        post_perplexity: float
    ) -> bool:
        """
        Check if fine-tuning is needed

        Args:
            pre_perplexity: Pre-compression perplexity
            post_perplexity: Post-compression perplexity

        Returns:
            True if fine-tuning recommended
        """
        if not self.config.enable_fine_tuning:
            return False

        # Calculate degradation
        degradation = (
            (post_perplexity - pre_perplexity) / pre_perplexity
        )

        # Compare to threshold
        return degradation > self.config.fine_tune_threshold

    def save_checkpoint(self, path: Path) -> None:
        """
        ISS-004/ISS-008: Save fine-tuning checkpoint using secure SafeTensors format.

        Args:
            path: Path to save checkpoint (extension will be replaced with .safetensors)
        """
        # Remove extension if present
        base_path = path.with_suffix('') if path.suffix else path

        # Training state goes in metadata (JSON-serializable)
        training_metadata = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_perplexity': float(self.best_perplexity),
            'training_history': self.training_history,
        }

        # Note: scaler state is handled separately if needed
        if self.scaler:
            # GradScaler state is small, can go in metadata
            scaler_state = self.scaler.state_dict()
            training_metadata['scaler_scale'] = float(scaler_state.get('scale', 1.0))
            training_metadata['scaler_growth_factor'] = float(scaler_state.get('_growth_factor', 2.0))

        secure_save(
            model=self.model,
            output_path=base_path,
            config=self.config.to_dict(),
            metadata=training_metadata,
            optimizer_state=self.optimizer.state_dict(),
        )

        print(f"Saved checkpoint: {base_path}.safetensors")

    def load_checkpoint(self, path: Path) -> Dict:
        """
        ISS-004/ISS-008: Load fine-tuning checkpoint from secure SafeTensors format.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint metadata dictionary
        """
        checkpoint_data = secure_load(
            model=self.model,
            checkpoint_path=path,
            device=str(self.device),
            load_optimizer=True,
        )

        # Restore optimizer state
        if checkpoint_data.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        # Restore training state from metadata
        metadata = checkpoint_data.get('metadata', {})
        self.current_epoch = metadata.get('epoch', 0)
        self.global_step = metadata.get('global_step', 0)
        self.best_perplexity = metadata.get('best_perplexity', float('inf'))
        self.training_history = metadata.get('training_history', [])

        # Restore scaler if available
        if self.scaler and 'scaler_scale' in metadata:
            self.scaler._scale = torch.tensor(metadata['scaler_scale'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best perplexity: {self.best_perplexity:.2f}")

        return {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_perplexity': self.best_perplexity,
            'num_history_entries': len(self.training_history),
        }

    def get_training_summary(self) -> Dict:
        """
        Get training summary

        Returns:
            Summary dictionary
        """
        if not self.training_history:
            return {
                'trained': False,
                'message': 'No fine-tuning performed',
            }

        return {
            'trained': True,
            'epochs': len(self.training_history),
            'final_loss': self.training_history[-1]['loss'],
            'best_perplexity': self.best_perplexity,
            'initial_loss': self.training_history[0]['loss'],
            'improvement': (
                self.training_history[0]['loss'] -
                self.training_history[-1]['loss']
            ),
            'global_steps': self.global_step,
            'mixed_precision_enabled': self.use_amp,
        }
