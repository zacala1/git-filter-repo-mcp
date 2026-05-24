"""Shared test fixtures and collection hooks.

Test organisation conventions:
- Pure-Python unit tests sit at module level with no real-git fixtures.
- Integration tests use ``temp_git_repo`` (or its siblings) and are
  automatically tagged ``integration`` by ``pytest_collection_modifyitems``
  below so CI can run only the fast unit slice with ``-m "not integration"``.
"""

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


# Fixture names that spin up a real git repo — any test using these is
# integration by definition. Listing them here keeps the marker logic in
# one place; add new repo fixtures here when defined.
_REPO_FIXTURES = frozenset({
    "temp_git_repo",
    "empty_git_repo",
    "single_commit_repo",
    "unicode_git_repo",
    "multiline_commit_repo",
})


def pytest_collection_modifyitems(config, items):
    """Auto-tag tests touching real git repos as ``integration``.

    Lets ``pytest -m "not integration"`` produce a quick unit-only run.
    """
    integration = pytest.mark.integration
    for item in items:
        if _REPO_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(integration)


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
def adapter(temp_git_repo):
    """A ``GitFilterRepoAdapter`` bound to the 3-commit ``temp_git_repo``.

    Most integration tests want this — having it as a fixture removes the
    ``GitFilterRepoAdapter(str(temp_git_repo))`` line from every test body
    and centralises the repo lifecycle.
    """
    from git_filter_repo_mcp.adapter import GitFilterRepoAdapter
    return GitFilterRepoAdapter(str(temp_git_repo))


def add_commit(repo_path: Path, filename: str, content: str, message: str) -> None:
    """Convenience for tests that need an extra commit on top of a fixture."""
    (repo_path / filename).write_text(content)
    _commit(repo_path, message)


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


@pytest.fixture
def multiline_commit_repo():
    """Create a git repo with multi-line commit messages (subject + body)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "multiline_repo"
        repo_path.mkdir()
        _init_repo(repo_path)

        (repo_path / "app.py").write_text("print('app')")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "feat: add app\n\nThis adds the main application.\nWith multiple lines in body."],
            cwd=repo_path, capture_output=True,
        )

        yield repo_path
