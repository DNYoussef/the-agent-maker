# Migration Guide: curriculum_engine.py Refactoring

## Summary

The 536-line `curriculum_engine.py` has been refactored into a modular structure with 5 files in the `engine/` subdirectory.

## New Structure

```
src/phase5_curriculum/
├── curriculum_engine.py (original - 536 lines)
└── engine/
    ├── __init__.py (18 lines) - Re-exports for backward compatibility
    ├── config.py (55 lines) - Configuration classes
    ├── progress.py (20 lines) - Progress tracking
    ├── result.py (24 lines) - Result dataclass
    └── curriculum_engine.py (459 lines) - Main engine
```

## File Breakdown

### 1. `config.py` (55 lines)
Contains:
- `SpecializationType` enum
- `CurriculumConfig` dataclass

### 2. `progress.py` (20 lines)
Contains:
- `LevelProgress` dataclass

### 3. `result.py` (24 lines)
Contains:
- `Phase5Result` dataclass

### 4. `curriculum_engine.py` (459 lines)
Contains:
- `CurriculumEngine` class (main orchestrator)

### 5. `__init__.py` (18 lines)
Re-exports all classes for backward compatibility.

## Migration Options

### Option 1: Update Imports (Recommended)

**Before:**
```python
from phase5_curriculum.curriculum_engine import (
    CurriculumEngine,
    CurriculumConfig,
    SpecializationType,
    Phase5Result,
    LevelProgress
)
```

**After (Modular):**
```python
from phase5_curriculum.engine.config import CurriculumConfig, SpecializationType
from phase5_curriculum.engine.progress import LevelProgress
from phase5_curriculum.engine.result import Phase5Result
from phase5_curriculum.engine.curriculum_engine import CurriculumEngine
```

### Option 2: No Changes Required (Backward Compatible)

**Still works:**
```python
from phase5_curriculum.engine import (
    CurriculumEngine,
    CurriculumConfig,
    SpecializationType,
    Phase5Result,
    LevelProgress
)
```

The `__init__.py` re-exports all classes, maintaining backward compatibility.

## Benefits of Refactoring

1. **Separation of Concerns**: Configuration, progress, results, and engine logic are now in separate files
2. **Easier Testing**: Can test config, progress, and result classes independently
3. **Better Maintainability**: Smaller files are easier to understand and modify
4. **Backward Compatible**: Existing code continues to work via `__init__.py` re-exports
5. **Reduced Complexity**: Main engine reduced from 536 to 459 lines

## Usage Examples

### Creating a Config (New Way)
```python
from phase5_curriculum.engine.config import CurriculumConfig, SpecializationType

config = CurriculumConfig(
    specialization=SpecializationType.CODING,
    num_levels=10,
    edge_of_chaos_threshold=0.75
)
```

### Running the Engine (Same as Before)
```python
from phase5_curriculum.engine import CurriculumEngine

engine = CurriculumEngine(config)
result = engine.run(model, tokenizer, frontier_client, coding_env)
```

### Tracking Progress (New Way)
```python
from phase5_curriculum.engine.progress import LevelProgress

progress = LevelProgress(
    level=1,
    initial_questions=2000,
    current_questions=1500,
    mastered_questions=500,
    variants_generated=100,
    hints_given=50,
    accuracy=0.75
)
```

## Next Steps

1. **Test backward compatibility**: Run existing code to ensure nothing breaks
2. **Update imports**: Gradually migrate to modular imports for clarity
3. **Review dependencies**: Check if any other files import from `curriculum_engine.py`
4. **Update tests**: Add unit tests for individual modules

## Questions?

See `engine/__init__.py` for the complete list of re-exported classes.
