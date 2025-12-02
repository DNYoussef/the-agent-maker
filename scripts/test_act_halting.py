"""Test ACT halting behavior"""
import sys
sys.path.insert(0, 'src')

import torch
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config

# Create model
config = Phase1Config(specialization='reasoning')
model = TRMTitansMAGModel(config)
model.eval()

# Test input
input_ids = torch.randint(0, 32768, (4, 64))

with torch.no_grad():
    output = model(input_ids, return_all_steps=True)

halting = output['halting_steps']
print(f'Halting steps: {halting}')
print(f'Mean: {halting.float().mean():.2f}')
print(f'Std: {halting.float().std():.4f}')
print(f'Variance: {halting.float().var():.4f}')

print(f'\nHalt probs per step:')
for i, probs in enumerate(output['halt_probs']):
    print(f'  Step {i}: mean={probs.mean():.4f}, std={probs.std():.4f}')

print(f'\n ACT threshold (reasoning): {config.act_config.halt_threshold}')
