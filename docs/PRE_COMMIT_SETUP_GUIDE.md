# Pre-Commit Hooks Setup Guide

**Agent Forge V2** - Code Quality Enforcement

This guide explains how to install and use the pre-commit hooks that enforce:
- **NASA POT10 compliance** (≤60 lines per function)
- **Code formatting** (Black, isort)
- **Type checking** (MyPy ≥98% coverage)
- **Linting** (Flake8)
- **Security checks** (no secrets, no backup files)
- **Test coverage** (ensure tests exist for new code)

---

## Quick Start (5 minutes)

```bash
# 1. Install pre-commit
pip install pre-commit

# 2. Install hooks (run from project root)
cd "C:\Users\17175\Desktop\the agent maker"
pre-commit install

# 3. Test the installation
pre-commit run --all-files
```

That's it! Hooks will now run automatically on every `git commit`.

---

## What Gets Checked

### On Every Commit

The following checks run automatically when you commit code:

| Check | Tool | Purpose | Fix Command |
|-------|------|---------|-------------|
| **Code Formatting** | Black | Enforce consistent style | `black --line-length=100 .` |
| **Import Sorting** | isort | Organize imports | `isort --profile=black .` |
| **Linting** | Flake8 | Catch common errors | Fix manually |
| **Type Checking** | MyPy | Enforce type hints (≥98%) | Add type hints |
| **NASA POT10** | Custom | ≤60 lines per function | Refactor long functions |
| **No Backup Files** | Custom | Prevent `_backup`, `.bak` commits | Use git branches |
| **No Secrets** | Custom | Detect hardcoded API keys | Use environment variables |
| **Test Coverage** | Custom | Ensure tests exist | Create test files |
| **Trailing Whitespace** | pre-commit-hooks | Remove trailing spaces | Auto-fixed |
| **YAML/JSON/TOML** | pre-commit-hooks | Validate config files | Fix syntax errors |

---

## Installation Options

### Option 1: System-Wide (Recommended)

Install pre-commit globally, available for all projects:

```bash
# Using pip
pip install pre-commit

# Or using pipx (isolated installation)
pipx install pre-commit

# Verify installation
pre-commit --version
# Expected output: pre-commit 3.x.x
```

### Option 2: Project Virtual Environment

Install within the project's virtual environment:

```bash
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

---

## Setup Steps

### 1. Install Pre-Commit Hooks

```bash
# From project root
cd "C:\Users\17175\Desktop\the agent maker"

# Install hooks to .git/hooks/
pre-commit install

# Expected output:
# pre-commit installed at .git/hooks/pre-commit
```

### 2. Initial Run (First Time)

Run all checks on existing codebase:

```bash
# Check all files (may take 2-5 minutes first time)
pre-commit run --all-files

# Expected output:
# black................................................Passed
# isort................................................Passed
# flake8...............................................Passed
# mypy.................................................Passed
# NASA POT10 Function Length Check.....................Passed
# Prevent Backup File Commits..........................Passed
# Prevent Secret Commits...............................Passed
# Test Coverage Check..................................Passed
```

**If checks fail on existing code:**
- V1 reference code is excluded (see [Exclusions](#exclusions))
- Fix V2 code issues before committing
- Or use `git commit --no-verify` (not recommended)

### 3. Auto-Fix Formatting Issues

Some checks auto-fix issues:

```bash
# Run black formatter
black --line-length=100 .

# Run isort import sorter
isort --profile=black .

# Remove trailing whitespace (auto-fixed by pre-commit)
pre-commit run trailing-whitespace --all-files
```

---

## Usage Examples

### Daily Development Workflow

```bash
# 1. Write code
vim src/phase1/model.py

# 2. Write tests
vim tests/phase1/test_model.py

# 3. Commit (hooks run automatically)
git add src/phase1/model.py tests/phase1/test_model.py
git commit -m "Add TinyTitan model architecture"

# Hooks run automatically:
# ✅ black: Passed
# ✅ isort: Passed
# ✅ flake8: Passed
# ✅ mypy: Passed
# ✅ NASA POT10: Passed
# ✅ Test Coverage: Passed
# [main abc1234] Add TinyTitan model architecture
```

### If Hooks Fail

```bash
# Scenario: Function too long (NASA POT10 violation)
git commit -m "Add training loop"

# Output:
# NASA POT10 Function Length Check.........................Failed
# - hook id: nasa-pot10-check
# - exit code: 1
#
# ❌ src/phase5/trainer.py:42 - Function 'train_epoch' exceeds limit
#    └─ 78 lines (limit: 60, excess: 18 lines)
#
# 💡 Suggestions:
#    • Extract helper functions to reduce complexity
#    • Split large functions into smaller, single-purpose functions

# Fix: Refactor the function
vim src/phase5/trainer.py
# Split train_epoch() into train_epoch() + _update_batch() + _log_metrics()

# Retry commit
git add src/phase5/trainer.py
git commit -m "Add training loop"
# ✅ All hooks pass
```

### Bypass Hooks (Not Recommended)

Only use in emergencies:

```bash
# Skip all hooks for this commit
git commit --no-verify -m "Emergency hotfix"

# Skip specific hook
SKIP=mypy git commit -m "Commit without type checking"
```

---

## Custom Scripts

### NASA POT10 Function Length Check

**Script**: `scripts/check_function_length.py`

**What it checks:**
- No function exceeds 60 lines of code (excluding docstrings)
- Based on NASA's Power of Ten rule #4

**Manual run:**
```bash
# Check specific files
python scripts/check_function_length.py src/phase1/model.py

# Check all files in src/
python scripts/check_function_length.py src/**/*.py

# Custom limit (50 lines)
python scripts/check_function_length.py --max-lines=50 src/*.py

# Verbose mode (show compliant files too)
python scripts/check_function_length.py --verbose src/*.py
```

**Example output:**
```
🔍 Checking 3 file(s) for NASA POT10 compliance (max 60 LOC/function)

📄 src/phase1/model.py:
   ❌ src/phase1/model.py:142 - Function 'forward' exceeds limit
      └─ 75 lines (limit: 60, excess: 15 lines)

──────────────────────────────────────────────────────────────────────
❌ FAILED: 1 violation(s) in 1 file(s)

💡 Suggestions:
   • Extract helper functions to reduce complexity
   • Split large functions into smaller, single-purpose functions
```

---

### Test Coverage Check

**Script**: `scripts/check_test_coverage.py`

**What it checks:**
- Every source file in `src/` has a corresponding test file
- Searches for: `tests/test_{filename}.py`, `tests/{module}/test_{filename}.py`

**Manual run:**
```bash
# Check specific files
python scripts/check_test_coverage.py src/phase1/model.py

# Check all staged files
python scripts/check_test_coverage.py $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

# Verbose mode
python scripts/check_test_coverage.py --verbose src/phase1/*.py
```

**Example output:**
```
🧪 Checking test coverage for 2 file(s)

❌ src/phase1/model.py: No test file found
   └─ Searched: tests/test_model.py, tests/phase1/test_model.py
✅ src/phase1/utils.py: Test found at tests/phase1/test_utils.py

──────────────────────────────────────────────────────────────────────
❌ MISSING TESTS: 1 file(s) without test coverage

💡 Expected test file locations:
   src/phase1/model.py:
     • tests/test_model.py
     • tests/phase1/test_model.py
```

**Exempt files** (don't require tests):
- `__init__.py`, `__main__.py`, `setup.py`
- Files in `scripts/`, `docs/`, `examples/`
- CLI scripts (`cli_*.py`)

---

## Exclusions

The following directories/files are **excluded** from checks:

- `v1-reference/` - V1 implementation (historical reference)
- `.venv/`, `venv/` - Virtual environments
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/` - Generated files
- `storage/`, `registry/`, `.wandb/` - Runtime data
- `*.egg-info/` - Package metadata

**Configured in:** [.pre-commit-config.yaml](../.pre-commit-config.yaml) (global `exclude` section)

---

## Troubleshooting

### Issue: "command not found: pre-commit"

**Solution:**
```bash
# Install pre-commit
pip install pre-commit

# Verify installation
pre-commit --version
```

### Issue: "No such file or directory: scripts/check_function_length.py"

**Solution:**
```bash
# Ensure scripts are executable (Linux/Mac)
chmod +x scripts/check_function_length.py
chmod +x scripts/check_test_coverage.py

# Or run with python explicitly (Windows)
python scripts/check_function_length.py
```

### Issue: Hooks run too slowly

**Solution:**
```bash
# Update pre-commit cache
pre-commit clean
pre-commit gc

# Run specific hook only
pre-commit run black

# Skip slow hooks temporarily
SKIP=mypy git commit -m "Quick commit"
```

### Issue: MyPy errors on valid code

**Solution:**
```bash
# Add type: ignore comment
result = some_function()  # type: ignore[attr-defined]

# Or add to mypy.ini (project root)
[mypy]
ignore_missing_imports = True

[mypy-some_problematic_module.*]
ignore_errors = True
```

### Issue: Pre-commit installed but not running

**Solution:**
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Check .git/hooks/pre-commit exists
ls .git/hooks/pre-commit

# Test manually
pre-commit run --all-files
```

---

## Configuration Files

### `.pre-commit-config.yaml`

Main configuration file defining all hooks.

**Location:** Project root
**Key sections:**
- `repos`: External hook sources (Black, Flake8, MyPy)
- `local`: Custom hooks (NASA POT10, test coverage)
- `exclude`: Global exclusions (V1 code, venvs)

**Edit to:**
- Change line length: `args: ['--line-length=100']`
- Add new hooks
- Modify exclusions

### `scripts/check_function_length.py`

NASA POT10 compliance checker.

**Customize:**
```python
# Change default max lines
parser.add_argument('--max-lines', type=int, default=60)  # Change 60 to desired limit

# Add exempt patterns
EXEMPT_PATTERNS = [
    '__init__.py',
    '__main__.py',
    'setup.py',
    'your_exempt_file.py',  # Add here
]
```

### `scripts/check_test_coverage.py`

Test coverage checker.

**Customize:**
```python
# Add exempt directories
EXEMPT_DIRECTORIES = [
    'scripts',
    'docs',
    'examples',
    'your_exempt_dir',  # Add here
]
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/pre-commit.yml
name: Pre-Commit Checks

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install pre-commit
        run: pip install pre-commit
      - name: Run pre-commit
        run: pre-commit run --all-files
```

---

## Best Practices

### 1. Run Before Committing

Get in the habit of running manually before commit:

```bash
# Check staged files only
pre-commit run

# Check all files
pre-commit run --all-files
```

### 2. Fix Issues Incrementally

Don't disable hooks permanently:

```bash
# ❌ Bad: Disable hooks permanently
git config core.hooksPath /dev/null

# ✅ Good: Bypass once, fix later
git commit --no-verify -m "WIP: Will fix linting"
# Then fix immediately:
pre-commit run --all-files
git add -u
git commit --amend --no-edit
```

### 3. Update Hooks Regularly

Keep hooks up to date:

```bash
# Update to latest versions
pre-commit autoupdate

# Install updated hooks
pre-commit install --install-hooks
```

### 4. Team Consistency

Ensure all team members have hooks installed:

```bash
# Add to README.md setup instructions
echo "pre-commit install" >> README.md

# Or add setup script
cat > setup.sh << 'EOF'
#!/bin/bash
pip install pre-commit
pre-commit install
echo "✅ Pre-commit hooks installed"
EOF
chmod +x setup.sh
```

---

## Advanced Usage

### Run Specific Hook

```bash
# Run only Black formatter
pre-commit run black

# Run only NASA POT10 check
pre-commit run nasa-pot10-check

# Run only test coverage check
pre-commit run test-coverage-check
```

### Run on Specific Files

```bash
# Check specific file
pre-commit run --files src/phase1/model.py

# Check multiple files
pre-commit run --files src/phase1/model.py src/phase1/utils.py
```

### Manual Hook Execution

Run scripts directly without pre-commit:

```bash
# NASA POT10 check
python scripts/check_function_length.py src/**/*.py

# Test coverage check
python scripts/check_test_coverage.py src/**/*.py

# Black formatter
black --check --line-length=100 src/

# MyPy type checker
mypy --strict src/
```

---

## FAQ

### Q: Can I disable specific hooks for certain files?

**A:** Yes, use `exclude` in `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/psf/black
  hooks:
    - id: black
      exclude: '^src/legacy/.*'  # Exclude legacy code
```

### Q: What if I need a function >60 lines?

**A:** Refactor into smaller functions. If absolutely necessary:

```python
# Add comment explaining why
def complex_algorithm():  # pylint: disable=too-many-lines
    """
    JUSTIFICATION: Algorithm must be atomic for correctness.
    Splitting would introduce bugs. Approved by team lead.
    """
    # 80 lines of complex logic
    pass
```

Then add to exempt patterns in `check_function_length.py`.

### Q: How do I test hooks without committing?

**A:**
```bash
# Stage files
git add src/phase1/model.py

# Run hooks on staged files
pre-commit run

# Unstage if needed
git reset HEAD src/phase1/model.py
```

### Q: Can I use different Python versions?

**A:** Yes, configure in `.pre-commit-config.yaml`:

```yaml
default_language_version:
  python: python3.11  # Change to your version
```

---

## Summary Checklist

**Initial Setup (One-Time):**
- [ ] Install pre-commit: `pip install pre-commit`
- [ ] Install hooks: `pre-commit install`
- [ ] Test installation: `pre-commit run --all-files`
- [ ] Fix any existing violations in V2 code

**Daily Workflow:**
- [ ] Write code in `src/`
- [ ] Write tests in `tests/`
- [ ] Run `pre-commit run` before committing
- [ ] Commit: `git commit -m "message"` (hooks run automatically)
- [ ] Fix any violations and retry

**Maintenance:**
- [ ] Update hooks monthly: `pre-commit autoupdate`
- [ ] Review custom scripts for improvements
- [ ] Keep NASA POT10 compliance at 100%

---

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Code quality standards (V2 section)
- [PREMORTEM_ANALYSIS.md](v2-planning/PREMORTEM_ANALYSIS.md) - Lines 932-968 (pre-commit spec)
- [ISSUE_RESOLUTION_MATRIX.md](v2-planning/ISSUE_RESOLUTION_MATRIX.md) - ISSUE-030 details

---

**Pre-Commit Hooks**: Enforcing NASA POT10, type safety, and test coverage from day 1 ✅
