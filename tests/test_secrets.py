"""Unit tests for secret detection (pure-Python, no external IO)."""

import pytest

from git_filter_repo_mcp.secrets import (
    get_file_risk_level,
    is_sensitive_file,
    redact_secret,
    scan_content,
)


class TestScanContent:
    """``scan_content`` pattern detection and deduplication."""

    # (pattern_name, sample_content) — each sample is crafted to trigger
    # exactly one specific pattern. Add new patterns here when they are
    # introduced in ``secrets.py``.
    DETECTIONS = [
        ("aws_access_key", "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"),
        ("aws_access_key", "AWS_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE"),
        ("aws_secret_key", "AWS_SECRET_ACCESS_KEY=" + "a" * 40),
        ("github_oauth", "token = 'gho_" + "x" * 36 + "'"),
        ("github_token", "token = 'ghp_" + "x" * 36 + "'"),
        ("openai_api_key", "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        ("openai_api_key", 'OPENAI_API_KEY="sk-proj-' + "x" * 48 + '"'),
        ("private_key", "-----BEGIN RSA PRIVATE KEY-----\nxxx\n-----END RSA PRIVATE KEY-----"),
        (
            "jwt_token",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        ),
        ("anthropic_api_key", "API_KEY=sk-ant-api03-" + "x" * 50),
        ("slack_token", "SLACK_TOKEN=xoxb-1234567890-123456789012-" + "a" * 24),
        (
            "slack_webhook",
            "https://hooks.slack.com/services/T12345678/B12345678/" + "a" * 24,
        ),
        ("stripe_key", "STRIPE_SECRET=sk_live_" + "a" * 24),
        ("stripe_test_key", "STRIPE_SECRET=sk_test_" + "a" * 24),
        ("google_api_key", "GOOGLE_API_KEY=AIza" + "a" * 35),
        ("firebase_key", "FIREBASE_KEY=AAAA" + "a" * 7 + ":" + "b" * 140),
        ("basic_auth", "https://user:password@example.com/path"),
        ("password_in_url", "https://example.com/reset?password=secret123"),
        ("generic_secret", 'api_key = "abcdefghijklmnop"'),
        ("env_secret", "APP_TOKEN=abc123"),
    ]

    @pytest.mark.parametrize("pattern_name,content", DETECTIONS, ids=[p for p, _ in DETECTIONS])
    def test_pattern_detected(self, pattern_name: str, content: str) -> None:
        findings = scan_content(content, "file.txt", "abc123")
        assert any(f.pattern_name == pattern_name for f in findings), (
            f"expected pattern {pattern_name!r} in {[f.pattern_name for f in findings]}"
        )

    def test_anthropic_key_not_misclassified_as_openai(self) -> None:
        """Both keys start with ``sk-`` — make sure the openai pattern
        excludes the anthropic prefix."""
        content = "API_KEY=sk-ant-api03-" + "x" * 50
        findings = scan_content(content, "config.py", "abc123")
        assert not any(f.pattern_name == "openai_api_key" for f in findings)

    def test_openai_project_key_not_downgraded_to_env_secret(self) -> None:
        content = 'OPENAI_API_KEY="sk-proj-' + "x" * 48 + '"'
        findings = scan_content(content, "config.py", "abc123")
        assert any(f.pattern_name == "openai_api_key" and f.severity == "high" for f in findings)
        assert not any(f.pattern_name == "env_secret" for f in findings)

    @pytest.mark.parametrize(
        "content,expected",
        [
            ("AWS_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE", "aws_access_key"),
            ("AWS_SECRET_ACCESS_KEY=" + "a" * 40, "aws_secret_key"),
        ],
    )
    def test_aws_specific_patterns_not_downgraded_to_env_secret(
        self, content: str, expected: str,
    ) -> None:
        findings = scan_content(content, "config.py", "abc123")
        assert any(f.pattern_name == expected and f.severity == "high" for f in findings)
        assert not any(f.pattern_name == "env_secret" for f in findings)

    def test_github_oauth_uses_specific_pattern(self) -> None:
        findings = scan_content("token='gho_" + "x" * 36 + "'", "config.py", "abc123")
        assert [f.pattern_name for f in findings] == ["github_oauth"]

    def test_no_false_positive_on_clean_code(self) -> None:
        content = 'def hello():\n    print("Hello, World!")\n    return 42\n'
        findings = scan_content(content, "main.py", "abc123")
        assert not [f for f in findings if f.severity == "high"]

    def test_overlapping_patterns_deduplicated(self) -> None:
        """Each character span should only be reported once even when
        multiple patterns could match it."""
        content = 'api_key = "sk-proj-' + "x" * 50 + '"'
        findings = scan_content(content, "config.py", "abc123")
        spans = [(f.line_number, f.matched_text) for f in findings]
        assert len(spans) == len(set(spans))


class TestSensitiveFiles:
    """``is_sensitive_file`` filename classifier."""

    @pytest.mark.parametrize(
        "path",
        [".env", ".env.local", ".env.production", "credentials.json",
         "secrets.json", "id_rsa", "server.key", "cert.pem", ".ENV",
         "SERVER.KEY", "CERT.PEM",
         # Regression: Windows-style separator must not block matching.
         "src\\firebase-adminsdk-abc.json", "configs\\.env",
         "src\\FIREBASE-ADMINSdk-abc.JSON"],
    )
    def test_sensitive(self, path: str) -> None:
        assert is_sensitive_file(path) is True

    @pytest.mark.parametrize("path", ["main.py", "README.md", "package.json"])
    def test_not_sensitive(self, path: str) -> None:
        assert is_sensitive_file(path) is False


class TestRiskLevel:
    """``get_file_risk_level`` ranking."""

    @pytest.mark.parametrize(
        "path,level",
        [
            (".env", "high"),
            ("id_rsa", "high"),
            ("server.pem", "high"),
            ("SERVER.PEM", "high"),
            # config.json itself is in SENSITIVE_FILES (high); use another
            # .json/.yml to exercise the medium branch.
            ("app_settings.json", "medium"),
            ("APP_SETTINGS.JSON", "medium"),
            ("data.yml", "medium"),
            ("main.py", "low"),
            ("index.js", "low"),
            # Regression: a dotted directory must not mask the actual file ext.
            ("dir.config/noextfile", "low"),
            ("dir.env/script.py", "low"),
        ],
    )
    def test_risk_level(self, path: str, level: str) -> None:
        assert get_file_risk_level(path) == level


class TestRedactSecret:
    """``redact_secret`` output shape and determinism."""

    def test_short_secret_fully_masked(self) -> None:
        result = redact_secret("abc")
        assert result.startswith("[REDACTED:") and result.endswith("]")

    @pytest.mark.parametrize(
        "secret,expected_prefix",
        [
            ("sk-1234567890abcdef", "sk-"),
            ("ghp_xxxxxxxxxxxxxxxxxxxxxxxx", "ghp"),
            ("AKIAIOSFODNN7EXAMPLE", "AKIA"),
            ("verylongsecretkey123456", "***"),  # unknown -> fully masked
        ],
    )
    def test_long_secret_prefix(self, secret: str, expected_prefix: str) -> None:
        result = redact_secret(secret)
        assert result.startswith(expected_prefix)
        assert "[" in result  # Hash suffix present

    def test_hash_is_deterministic(self) -> None:
        secret = "my_secret_key_12345"
        assert redact_secret(secret) == redact_secret(secret)

    def test_different_secrets_get_different_hashes(self) -> None:
        assert redact_secret("secret_one_12345") != redact_secret("secret_two_12345")
