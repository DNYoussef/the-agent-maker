"""
Unit tests for Model Registry
Tests SQLite WAL mode, session tracking, and model registration
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest


from cross_phase.storage.model_registry import ModelRegistry


class TestModelRegistry:
    """Test ModelRegistry functionality"""

    def test_registry_creation(self, temp_dir):
        """Test registry database creation"""
        db_path = temp_dir / "test_registry.db"
        registry = ModelRegistry(str(db_path))

        assert db_path.exists()
        registry.close()

    def test_wal_mode_enabled(self, temp_dir):
        """Test WAL mode is enabled"""
        db_path = temp_dir / "test_wal.db"
        registry = ModelRegistry(str(db_path))

        cursor = registry.conn.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"
        registry.close()

    def test_session_creation(self, temp_dir):
        """Test session creation"""
        registry = ModelRegistry(str(temp_dir / "test.db"))

        session_id = "test_session_001"
        session_data = {"created": datetime.now().isoformat(), "pipeline": "test-pipeline"}

        registry.create_session(session_id, session_data)

        # Verify session exists
        cursor = registry.conn.execute(
            "SELECT session_id, status FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == session_id
        assert row[1] == "running"

        registry.close()

    def test_progress_update(self, temp_dir):
        """Test session progress updates"""
        registry = ModelRegistry(str(temp_dir / "test.db"))

        session_id = "test_session_002"
        registry.create_session(session_id, {})

        # Update progress
        registry.update_session_progress(session_id, "phase1", 25.0)

        cursor = registry.conn.execute(
            "SELECT current_phase, progress_percent FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()

        assert row[0] == "phase1"
        assert row[1] == 25.0

        registry.close()

    def test_model_registration(self, temp_dir):
        """Test model registration"""
        registry = ModelRegistry(str(temp_dir / "test.db"))

        session_id = "test_session_003"
        registry.create_session(session_id, {})

        model_data = {
            "model_id": "test_model_001",
            "session_id": session_id,
            "phase": "phase1",
            "model_path": "/path/to/model.pt",
            "params": 25_000_000,
            "size_mb": 95.4,
            "metrics": {"loss": 2.34, "accuracy": 45.2},
        }

        model_id = registry.register_model(**model_data)

        # Verify model exists
        cursor = registry.conn.execute(
            "SELECT model_id, params, size_mb FROM models WHERE model_id = ?", (model_id,)
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == "test_model_001"
        assert row[1] == 25_000_000
        assert row[2] == 95.4

        registry.close()

    def test_wal_checkpoint(self, temp_dir):
        """Test WAL checkpoint operation"""
        registry = ModelRegistry(str(temp_dir / "test.db"))

        # Perform some operations
        registry.create_session("test_wal_checkpoint", {})

        # Checkpoint
        registry.checkpoint_wal()

        # Verify no errors
        registry.close()

    def test_incremental_vacuum(self, temp_dir):
        """Test incremental vacuum"""
        registry = ModelRegistry(str(temp_dir / "test.db"))

        # Create some data
        for i in range(5):
            registry.create_session(f"test_vacuum_{i}", {})

        # Vacuum
        registry.incremental_vacuum()

        # Verify no errors
        registry.close()

    def test_context_manager(self, temp_dir):
        """Test context manager support"""
        db_path = temp_dir / "test_context.db"

        with ModelRegistry(str(db_path)) as registry:
            registry.create_session("test_context", {})

        # Verify database was closed
        assert db_path.exists()
