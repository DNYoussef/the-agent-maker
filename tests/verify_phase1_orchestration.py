
import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
src_path = str(Path(__file__).parents[1] / "src")
sys.path.insert(0, src_path)

from cross_phase.orchestrator.phase_controller import Phase1Controller, PhaseResult

class TestPhase1Orchestration(unittest.TestCase):
    def setUp(self):
        self.config = {
            "epochs": 1,
            "batch_size": 2,
            "phases": {
                "phase1": {}
            }
        }
        self.session_id = "test_session_001"

    @patch("phase1_cognate.data.dataset_downloader.download_all_datasets")
    @patch("phase1_cognate.data.dataset_processor.process_dataset")
    @patch("phase1_cognate.training.trainer.Phase1Trainer.train")
    def test_execute_flow(self, mock_train, mock_process, mock_download):
        # Setup mocks
        mock_download.return_value = {"gsm8k": []} # Return dummy raw dataset
        mock_process.return_value = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}] * 10 # Dummy processed data
        
        # Initialize Controller
        controller = Phase1Controller(self.config, self.session_id)
        
        # Execute
        print("Starting Phase 1 Execution Test...")
        result = controller.execute()
        
        # Verify
        self.assertIsInstance(result, PhaseResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.model), 3) # Should have 3 models
        self.assertIn("reasoning", result.metrics)
        self.assertIn("memory", result.metrics)
        self.assertIn("speed", result.metrics)
        
        print("\nPhase 1 Execution Test Passed!")
        print(f"Models generated: {len(result.model)}")
        print(f"Metrics: {result.metrics}")

if __name__ == "__main__":
    unittest.main()
