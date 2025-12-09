"""Tests for secret detection."""

from git_filter_repo_mcp.secrets import (
    get_file_risk_level,
    is_sensitive_file,
    redact_secret,
    scan_content,
)


class TestSecretPatterns:
    """Test secret pattern detection."""

    def test_aws_access_key(self):
        content = "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"
        findings = scan_content(content, "config.py", "abc123")
        assert len(findings) >= 1
        assert any(f.pattern_name == "aws_access_key" for f in findings)

    def test_github_token(self):
        content = "token = 'ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'"
        findings = scan_content(content, "config.py", "abc123")
        assert len(findings) >= 1
        assert any(f.pattern_name == "github_token" for f in findings)

    def test_openai_api_key(self):
        content = "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        findings = scan_content(content, ".env", "abc123")
        assert len(findings) >= 1
        assert any(f.pattern_name == "openai_api_key" for f in findings)

    def test_private_key(self):
        content = "-----BEGIN RSA PRIVATE KEY-----\nxxx\n-----END RSA PRIVATE KEY-----"
        findings = scan_content(content, "id_rsa", "abc123")
        assert len(findings) >= 1
        assert any(f.pattern_name == "private_key" for f in findings)

    def test_jwt_token(self):
        content = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        findings = scan_content(content, "auth.py", "abc123")
        assert len(findings) >= 1
        assert any(f.pattern_name == "jwt_token" for f in findings)

    def test_no_false_positive_normal_code(self):
        content = """
def hello():
    print("Hello, World!")
    return 42
"""
        findings = scan_content(content, "main.py", "abc123")
        # Should not detect secrets in normal code
        high_severity = [f for f in findings if f.severity == "high"]
        assert len(high_severity) == 0


class TestSensitiveFiles:
    """Test sensitive file detection."""

    def test_env_file(self):
        assert is_sensitive_file(".env") is True
        assert is_sensitive_file(".env.local") is True
        assert is_sensitive_file(".env.production") is True

    def test_credential_files(self):
        assert is_sensitive_file("credentials.json") is True
        assert is_sensitive_file("secrets.json") is True

    def test_key_files(self):
        assert is_sensitive_file("id_rsa") is True
        assert is_sensitive_file("server.key") is True
        assert is_sensitive_file("cert.pem") is True

    def test_normal_files(self):
        assert is_sensitive_file("main.py") is False
        assert is_sensitive_file("README.md") is False
        assert is_sensitive_file("package.json") is False


class TestRiskLevel:
    """Test file risk level assessment."""

    def test_high_risk(self):
        assert get_file_risk_level(".env") == "high"
        assert get_file_risk_level("id_rsa") == "high"
        assert get_file_risk_level("server.pem") == "high"

    def test_medium_risk(self):
        # config.json is in SENSITIVE_FILES, so use other json files
        assert get_file_risk_level("app_settings.json") == "medium"
        assert get_file_risk_level("data.yml") == "medium"

    def test_low_risk(self):
        assert get_file_risk_level("main.py") == "low"
        assert get_file_risk_level("index.js") == "low"


class TestRedaction:
    """Test secret redaction."""

    def test_redact_short_secret(self):
        result = redact_secret("abc")
        # Short secrets are fully redacted with hash
        assert result.startswith("[REDACTED:")
        assert result.endswith("]")

    def test_redact_long_secret_with_known_prefix(self):
        result = redact_secret("sk-1234567890abcdef")
        # Known prefix (sk-) is shown
        assert result.startswith("sk-")
        assert "***" in result
        assert "[" in result  # Contains hash

    def test_redact_long_secret_unknown_prefix(self):
        result = redact_secret("verylongsecretkey123456")
        # Unknown prefix is fully masked
        assert result.startswith("***")
        assert "[" in result  # Contains hash

    def test_redact_consistent_hash(self):
        # Same secret should produce same hash
        secret = "my_secret_key_12345"
        result1 = redact_secret(secret)
        result2 = redact_secret(secret)
        assert result1 == result2

    def test_redact_different_hash(self):
        # Different secrets should produce different hashes
        result1 = redact_secret("secret_one_12345")
        result2 = redact_secret("secret_two_12345")
        assert result1 != result2
