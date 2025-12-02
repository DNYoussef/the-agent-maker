"""
Docker Sandbox for Phase 5 Code Execution

Provides secure, isolated code execution in ephemeral Docker containers.
Used for validating generated code during curriculum learning.

Features:
- Timeout enforcement (default 5s)
- Memory limits (256MB default)
- Network isolation
- Automatic container cleanup
- Support for Python, JavaScript, Bash

Security:
- No host filesystem access
- No network access (except to container)
- Resource limits enforced
- Containers destroyed after execution
"""
import os
import asyncio
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging
import subprocess
import time

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    timed_out: bool = False
    error: Optional[str] = None


@dataclass
class SandboxConfig:
    """Configuration for Docker sandbox."""
    timeout_seconds: float = 5.0
    memory_limit_mb: int = 256
    cpu_limit: float = 0.5  # Half a CPU core
    network_disabled: bool = True
    auto_cleanup: bool = True


# Docker images for each language
DOCKER_IMAGES: Dict[Language, str] = {
    Language.PYTHON: "python:3.10-slim",
    Language.JAVASCRIPT: "node:18-slim",
    Language.BASH: "alpine:latest"
}

# Default commands to run code
RUN_COMMANDS: Dict[Language, List[str]] = {
    Language.PYTHON: ["python", "/sandbox/code.py"],
    Language.JAVASCRIPT: ["node", "/sandbox/code.js"],
    Language.BASH: ["sh", "/sandbox/code.sh"]
}

# File extensions
FILE_EXTENSIONS: Dict[Language, str] = {
    Language.PYTHON: ".py",
    Language.JAVASCRIPT: ".js",
    Language.BASH: ".sh"
}


class DockerSandbox:
    """
    Secure Docker sandbox for code execution.

    Usage:
        sandbox = DockerSandbox()

        # Execute Python code
        result = await sandbox.execute(
            code="print('Hello, World!')",
            language=Language.PYTHON
        )

        if result.success:
            print(f"Output: {result.stdout}")
        else:
            print(f"Error: {result.stderr}")
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize Docker sandbox.

        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Docker not available. Using fallback execution.")
            return False

    async def execute(
        self,
        code: str,
        language: Language = Language.PYTHON,
        timeout: Optional[float] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute code in isolated Docker container.

        Args:
            code: Source code to execute
            language: Programming language
            timeout: Execution timeout (overrides config)
            env_vars: Environment variables to set

        Returns:
            ExecutionResult with output and metadata
        """
        timeout = timeout or self.config.timeout_seconds

        if self._docker_available:
            return await self._execute_docker(code, language, timeout, env_vars)
        else:
            return await self._execute_fallback(code, language, timeout)

    async def _execute_docker(
        self,
        code: str,
        language: Language,
        timeout: float,
        env_vars: Optional[Dict[str, str]]
    ) -> ExecutionResult:
        """Execute code in Docker container."""
        start_time = time.perf_counter()
        temp_dir = None

        try:
            # Create temporary directory with code file
            temp_dir = tempfile.mkdtemp(prefix="sandbox_")
            file_ext = FILE_EXTENSIONS[language]
            code_file = os.path.join(temp_dir, f"code{file_ext}")

            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Build Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "--read-only",  # Read-only filesystem
                f"--memory={self.config.memory_limit_mb}m",
                f"--cpus={self.config.cpu_limit}",
                f"--timeout={int(timeout)}",
                "-v", f"{temp_dir}:/sandbox:ro"  # Mount code read-only
            ]

            # Network isolation
            if self.config.network_disabled:
                docker_cmd.append("--network=none")

            # Environment variables
            if env_vars:
                for key, value in env_vars.items():
                    docker_cmd.extend(["-e", f"{key}={value}"])

            # Image and command
            docker_cmd.append(DOCKER_IMAGES[language])
            docker_cmd.extend(RUN_COMMANDS[language])

            # Execute
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout + 5  # Extra buffer for Docker overhead
                )
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                timed_out = True
                stdout = b""
                stderr = b"Execution timed out"

            execution_time = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                success=process.returncode == 0 and not timed_out,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or -1,
                execution_time_ms=execution_time,
                timed_out=timed_out
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                execution_time_ms=execution_time,
                error=str(e)
            )

        finally:
            # Cleanup
            if temp_dir and self.config.auto_cleanup:
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _execute_fallback(
        self,
        code: str,
        language: Language,
        timeout: float
    ) -> ExecutionResult:
        """
        Fallback execution without Docker (less secure, for testing only).

        WARNING: Only use for trusted code in development!
        """
        if language != Language.PYTHON:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Fallback execution only supports Python. Got: {language}",
                exit_code=-1,
                execution_time_ms=0.0,
                error="Docker not available and language not supported in fallback"
            )

        start_time = time.perf_counter()
        temp_file = None

        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            )
            temp_file.write(code)
            temp_file.close()

            # Execute with subprocess
            process = await asyncio.create_subprocess_exec(
                "python", temp_file.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                timed_out = True
                stdout = b""
                stderr = b"Execution timed out"

            execution_time = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                success=process.returncode == 0 and not timed_out,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or -1,
                execution_time_ms=execution_time,
                timed_out=timed_out
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                execution_time_ms=execution_time,
                error=str(e)
            )

        finally:
            # Cleanup
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    async def validate_code(
        self,
        code: str,
        expected_output: Optional[str] = None,
        language: Language = Language.PYTHON
    ) -> Dict[str, Any]:
        """
        Validate code by executing and checking output.

        Args:
            code: Code to validate
            expected_output: Expected stdout (optional)
            language: Programming language

        Returns:
            Validation result with success flag and details
        """
        result = await self.execute(code, language)

        validation = {
            "executed": result.success,
            "output_match": True,
            "execution_time_ms": result.execution_time_ms,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "error": result.error
        }

        # Check output if expected
        if expected_output is not None:
            actual = result.stdout.strip()
            expected = expected_output.strip()
            validation["output_match"] = actual == expected
            validation["expected"] = expected
            validation["actual"] = actual

        validation["success"] = result.success and validation["output_match"]

        return validation

    def is_docker_available(self) -> bool:
        """Check if Docker is available for secure execution."""
        return self._docker_available


# Synchronous wrapper for simple use cases
def run_code_sync(
    code: str,
    language: Language = Language.PYTHON,
    timeout: float = 5.0
) -> ExecutionResult:
    """
    Synchronous wrapper for code execution.

    Args:
        code: Code to execute
        language: Programming language
        timeout: Execution timeout

    Returns:
        ExecutionResult
    """
    sandbox = DockerSandbox(SandboxConfig(timeout_seconds=timeout))
    return asyncio.run(sandbox.execute(code, language))


__all__ = [
    "DockerSandbox",
    "SandboxConfig",
    "ExecutionResult",
    "Language",
    "run_code_sync"
]
