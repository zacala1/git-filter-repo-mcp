"""Shared test fixtures."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Marker for tests requiring git-filter-repo
requires_git_filter_repo = pytest.mark.skipif(
    not any(
        (Path(p) / "git-filter-repo").exists() or (Path(p) / "git-filter-repo.exe").exists()
        for p in os.environ.get("PATH", "").split(os.pathsep)
    ),
    reason="git-filter-repo not installed",
)


def _init_repo(repo_path: Path) -> None:
    """Initialize a git repo with standard config."""
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path, capture_output=True,
    )


def _commit(repo_path: Path, message: str) -> None:
    """Stage all and commit."""
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path, capture_output=True,
    )


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository with 3 commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()
        _init_repo(repo_path)

        (repo_path / "README.md").write_text("# Test Repo")
        _commit(repo_path, "Initial commit")

        (repo_path / "main.py").write_text("print('hello')")
        _commit(repo_path, "Add main.py")

        (repo_path / "config.json").write_text('{"key": "value"}')
        _commit(repo_path, "Add config")

        yield repo_path


@pytest.fixture
def empty_git_repo():
    """Create a temporary git repository with zero commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "empty_repo"
        repo_path.mkdir()
        _init_repo(repo_path)
        yield repo_path


@pytest.fixture
def single_commit_repo():
    """Create a temporary git repository with exactly one commit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "single_repo"
        repo_path.mkdir()
        _init_repo(repo_path)

        (repo_path / "README.md").write_text("# Single Commit Repo")
        _commit(repo_path, "Only commit")

        yield repo_path


@pytest.fixture
def unicode_git_repo():
    """Create a git repo with unicode content in commit messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "unicode_repo"
        repo_path.mkdir()
        _init_repo(repo_path)

        (repo_path / "readme.md").write_text("# Unicode test")
        _commit(repo_path, "feat: add unicode support \u2728")

        yield repo_path
