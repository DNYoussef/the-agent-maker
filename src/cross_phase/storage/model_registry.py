"""
SQLite Model Registry with WAL Mode
Concurrent read/write access for model metadata and session tracking
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ModelRegistry:
    """
    Model Registry with WAL (Write-Ahead Logging) mode for concurrent access

    Features:
    - Concurrent read/write (readers don't block writers)
    - Session tracking with auto-cleanup
    - Phase handoff validation
    - Model metadata versioning
    """

    def __init__(self, db_path: str = "./storage/registry/model_registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with WAL mode
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False  # Allow multi-threading
        )

        # Enable WAL mode (CRITICAL for concurrent access)
        self._enable_wal_mode()

        # Create schema
        self._create_schema()

    def _enable_wal_mode(self):
        """Enable WAL mode and optimize for concurrent access"""
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA cache_size=10000;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA mmap_size=30000000000;")
        self.conn.execute("PRAGMA page_size=4096;")
        self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        self.conn.commit()

    def _create_schema(self):
        """Create database schema"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                phase_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                size_mb REAL,
                parameters INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT,
                tags TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_session ON models(session_id);
            CREATE INDEX IF NOT EXISTS idx_phase ON models(phase_name);
            CREATE INDEX IF NOT EXISTS idx_created ON models(created_at);

            CREATE TABLE IF NOT EXISTS phase_handoffs (
                handoff_id TEXT PRIMARY KEY,
                source_phase TEXT NOT NULL,
                target_phase TEXT NOT NULL,
                source_model_id TEXT NOT NULL,
                target_model_id TEXT NOT NULL,
                handoff_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validation_status TEXT,
                validation_metrics_json TEXT,
                FOREIGN KEY (source_model_id) REFERENCES models(model_id),
                FOREIGN KEY (target_model_id) REFERENCES models(model_id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                pipeline_config_json TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                current_phase TEXT,
                progress_percent REAL
            );
        """)
        self.conn.commit()

    def register_model(
        self,
        session_id: str,
        phase_name: str,
        model_name: str,
        model_path: str,
        metadata: Dict
    ) -> str:
        """Register a model in the registry"""
        model_id = f"{phase_name}_{model_name}_{session_id}"

        # Get model size
        size_mb = os.path.getsize(model_path) / (1024 ** 2)

        self.conn.execute("""
            INSERT INTO models (
                model_id, session_id, phase_name, model_name,
                model_path, size_mb, parameters, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, session_id, phase_name, model_name,
            model_path, size_mb, metadata.get('parameters', 0),
            json.dumps(metadata)
        ))
        self.conn.commit()

        return model_id

    def get_model(
        self,
        model_id: Optional[str] = None,
        session_id: Optional[str] = None,
        phase_name: Optional[str] = None
    ) -> Dict:
        """Get model info from registry"""
        if model_id:
            query = "SELECT * FROM models WHERE model_id = ?"
            params = (model_id,)
        elif session_id and phase_name:
            query = """
                SELECT * FROM models
                WHERE session_id = ? AND phase_name = ?
                ORDER BY created_at DESC LIMIT 1
            """
            params = (session_id, phase_name)
        else:
            raise ValueError("Must provide model_id or (session_id, phase_name)")

        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        if not row:
            raise FileNotFoundError(f"Model not found: {model_id or (session_id, phase_name)}")

        return {
            'model_id': row[0],
            'session_id': row[1],
            'phase_name': row[2],
            'model_name': row[3],
            'model_path': row[4],
            'size_mb': row[5],
            'parameters': row[6],
            'created_at': row[7],
            'metadata': json.loads(row[8]) if row[8] else {}
        }

    def create_session(self, session_id: str, config: Dict):
        """Create new session"""
        self.conn.execute("""
            INSERT INTO sessions (
                session_id, pipeline_config_json, status, current_phase, progress_percent
            ) VALUES (?, ?, 'running', 'phase1', 0.0)
        """, (session_id, json.dumps(config)))
        self.conn.commit()

    def update_session_progress(
        self,
        session_id: str,
        current_phase: str,
        progress_percent: float
    ):
        """Update session progress"""
        self.conn.execute("""
            UPDATE sessions
            SET current_phase = ?, progress_percent = ?
            WHERE session_id = ?
        """, (current_phase, progress_percent, session_id))
        self.conn.commit()

    def checkpoint_wal(self):
        """Manually checkpoint WAL to main database"""
        self.conn.execute("PRAGMA wal_checkpoint(RESTART);")
        self.conn.commit()

    def vacuum_incremental(self, pages: int = 100):
        """Incremental vacuum to reclaim space"""
        self.conn.execute(f"PRAGMA incremental_vacuum({pages});")
        self.conn.commit()

    def get_all_models(
        self,
        phase_filter: Optional[List[str]] = None,
        session_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get all models from registry with optional filters.

        Args:
            phase_filter: List of phase names to filter by
            session_filter: Session ID to filter by
            limit: Maximum number of results

        Returns:
            List of model dictionaries
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if phase_filter:
            placeholders = ','.join('?' * len(phase_filter))
            query += f" AND phase_name IN ({placeholders})"
            params.extend(phase_filter)

        if session_filter:
            query += " AND session_id = ?"
            params.append(session_filter)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        models = []
        for row in rows:
            metadata = json.loads(row[8]) if row[8] else {}
            models.append({
                'model_id': row[0],
                'session_id': row[1],
                'phase': row[2],
                'name': row[3],
                'model_path': row[4],
                'size_mb': row[5] or 0,
                'params': row[6] or 0,
                'created': row[7],
                'status': metadata.get('status', 'complete'),
                'loss': metadata.get('loss', 0.0),
                'accuracy': metadata.get('accuracy', 0.0),
                'perplexity': metadata.get('perplexity', 0.0),
                'metadata': metadata
            })

        return models

    def get_storage_stats(self) -> Dict:
        """Get storage statistics for all models.

        Returns:
            Dictionary with storage breakdown
        """
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as model_count,
                COALESCE(SUM(size_mb), 0) as total_size_mb,
                COUNT(DISTINCT session_id) as session_count,
                COUNT(DISTINCT phase_name) as phase_count
            FROM models
        """)
        row = cursor.fetchone()

        # Count checkpoints by checking for checkpoint-related models
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM models
            WHERE model_name LIKE '%checkpoint%' OR model_name LIKE '%ckpt%'
        """)
        checkpoint_count = cursor.fetchone()[0]

        return {
            'model_count': row[0],
            'total_size_mb': row[1],
            'session_count': row[2],
            'phase_count': row[3],
            'checkpoint_count': checkpoint_count
        }

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry.

        Args:
            model_id: ID of model to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            cursor = self.conn.execute(
                "DELETE FROM models WHERE model_id = ?",
                (model_id,)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False

    def register_phase_handoff(
        self,
        from_phase: int,
        to_phase: int,
        session_id: str,
        input_model_metadata: Dict,
        output_model_metadata: Dict,
        validation_status: str,
        validation_metrics: Dict
    ) -> str:
        """Register a phase handoff in the registry.

        Args:
            from_phase: Source phase number
            to_phase: Target phase number
            session_id: Session identifier
            input_model_metadata: Metadata from source model
            output_model_metadata: Metadata from target model
            validation_status: 'passed' or 'failed'
            validation_metrics: Validation metrics dictionary

        Returns:
            Handoff ID
        """
        import uuid
        handoff_id = f"handoff_{from_phase}_{to_phase}_{uuid.uuid4().hex[:8]}"

        source_model_id = f"phase{from_phase}_{session_id}"
        target_model_id = f"phase{to_phase}_{session_id}"

        self.conn.execute("""
            INSERT INTO phase_handoffs (
                handoff_id, source_phase, target_phase,
                source_model_id, target_model_id,
                validation_status, validation_metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            handoff_id,
            f"phase{from_phase}",
            f"phase{to_phase}",
            source_model_id,
            target_model_id,
            validation_status,
            json.dumps(validation_metrics)
        ))
        self.conn.commit()

        return handoff_id

    def close(self):
        """Close database connection"""
        self.checkpoint_wal()  # Final checkpoint
        self.conn.close()
