"""
Phase 4: BitNet 1.58-bit Compression

Compresses models from Phase 3 using ternary quantization to {-1, 0, +1}
for 8.2x compression and 2-4x inference speedup.

Key Components:
- BitLinear: Drop-in replacement for nn.Linear (NEW)
- BitNetQuantizer: Core quantization engine
- CompressedModel: STE-enabled model wrapper
- CalibrationDataset: Calibration data loader
- FineTuner: MuGrokfast-based fine-tuning
- Phase4Controller: Pipeline integration

Outputs:
- Quantized model (12MB, 1.58-bit)
- Dequantized FP16 model (50MB, PRIMARY for Phase 5)

Paper Reference:
- BitNet b1.58 (Microsoft): arXiv:2310.11453, arXiv:2402.17764
"""

from src.phase4_bitnet.bitlinear import BitLinear, replace_linear_with_bitlinear
from src.phase4_bitnet.calibration import CalibrationDataset
from src.phase4_bitnet.compressed_model import CompressedModel
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.fine_tuner import FineTuner
from src.phase4_bitnet.phase_controller import Phase4Controller
from src.phase4_bitnet.quantizer import BitNetQuantizer, activation_quant, apply_ste

__all__ = [
    "BitLinear",
    "replace_linear_with_bitlinear",
    "BitNetQuantizer",
    "activation_quant",
    "apply_ste",
    "CompressedModel",
    "CalibrationDataset",
    "FineTuner",
    "Phase4Controller",
    "Phase4Config",
]

__version__ = "1.1.0"  # Updated with BitLinear support
