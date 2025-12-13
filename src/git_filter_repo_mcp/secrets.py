"""Secret detection utilities for git history scanning."""

import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern


@dataclass
class SecretPattern:
    """Pattern for detecting secrets."""

    name: str
    pattern: Pattern
    description: str
    severity: str = "high"  # high, medium, low


# Common secret patterns
SECRET_PATTERNS: list[SecretPattern] = [
    SecretPattern(
        name="aws_access_key",
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        description="AWS Access Key ID",
        severity="high",
    ),
    SecretPattern(
        name="aws_secret_key",
        pattern=re.compile(
            r"(?i)(aws_secret|secret_key|secret_access)['\"]?\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
        ),
        description="AWS Secret Key",
        severity="high",
    ),
    SecretPattern(
        name="github_token",
        pattern=re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
        description="GitHub Token",
        severity="high",
    ),
    SecretPattern(
        name="github_oauth",
        pattern=re.compile(r"gho_[A-Za-z0-9]{36}"),
        description="GitHub OAuth Token",
        severity="high",
    ),
    SecretPattern(
        name="openai_api_key",
        pattern=re.compile(r"sk-[A-Za-z0-9]{48,}"),
        description="OpenAI API Key",
        severity="high",
    ),
    SecretPattern(
        name="anthropic_api_key",
        pattern=re.compile(r"sk-ant-[A-Za-z0-9-]{40,}"),
        description="Anthropic API Key",
        severity="high",
    ),
    SecretPattern(
        name="slack_token",
        pattern=re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}"),
        description="Slack Token",
        severity="high",
    ),
    SecretPattern(
        name="slack_webhook",
        pattern=re.compile(
            r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"
        ),
        description="Slack Webhook URL",
        severity="medium",
    ),
    SecretPattern(
        name="stripe_key",
        pattern=re.compile(r"sk_live_[A-Za-z0-9]{24,}"),
        description="Stripe Live Key",
        severity="high",
    ),
    SecretPattern(
        name="stripe_test_key",
        pattern=re.compile(r"sk_test_[A-Za-z0-9]{24,}"),
        description="Stripe Test Key",
        severity="low",
    ),
    SecretPattern(
        name="google_api_key",
        pattern=re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        description="Google API Key",
        severity="high",
    ),
    SecretPattern(
        name="firebase_key",
        pattern=re.compile(r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}"),
        description="Firebase Cloud Messaging Key",
        severity="high",
    ),
    SecretPattern(
        name="private_key",
        pattern=re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        description="Private Key File",
        severity="high",
    ),
    SecretPattern(
        name="jwt_token",
        pattern=re.compile(r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+"),
        description="JWT Token",
        severity="medium",
    ),
    SecretPattern(
        name="basic_auth",
        pattern=re.compile(r"https?://[^/:@\s]+:[^/@\s]+@[^/\s]+"),
        description="URL with Basic Auth Credentials",
        severity="high",
    ),
    SecretPattern(
        name="password_in_url",
        pattern=re.compile(r"[?&]password=[^&\s]+"),
        description="Password in URL Parameter",
        severity="high",
    ),
    SecretPattern(
        name="generic_secret",
        pattern=re.compile(
            r"(?i)(api[_-]?key|secret|password|token|credential)['\"]?\s*[=:]\s*['\"][A-Za-z0-9+/=]{16,}['\"]"
        ),
        description="Generic Secret Assignment",
        severity="medium",
    ),
    SecretPattern(
        name="env_secret",
        pattern=re.compile(
            r"(?i)^[A-Z_]*(SECRET|KEY|TOKEN|PASSWORD|CREDENTIAL)[A-Z_]*\s*=\s*['\"]?[^\s'\"]+['\"]?",
            re.MULTILINE,
        ),
        description="Environment Variable Secret",
        severity="medium",
    ),
]

# Files commonly containing secrets
SENSITIVE_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "secrets.json",
    "config.json",
    "settings.json",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "service-account.json",
    "firebase-adminsdk*.json",
    ".htpasswd",
    "wp-config.php",
    "database.yml",
    "secrets.yml",
]


@dataclass
class SecretFinding:
    """A detected secret in the repository."""

    pattern_name: str
    description: str
    severity: str
    file_path: str
    commit_hash: str
    line_number: int | None
    matched_text: str  # Redacted version
    context: str | None = None


def redact_secret(text: str, visible_chars: int = 4) -> str:
    """Redact a secret with hash identifier for tracking."""
    # Generate short hash for tracking without exposing secret content
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

    if len(text) <= 8:
        # Short secrets: fully redact with hash
        return f"[REDACTED:{text_hash}]"

    # Longer secrets: show type hint (first 2-4 chars) + hash
    # Only show prefix if it's a known safe prefix pattern
    safe_prefixes = ["sk-", "ghp", "gho", "AKIA", "xox", "eyJ"]
    prefix = ""
    for safe in safe_prefixes:
        if text.startswith(safe):
            prefix = safe
            break

    if prefix:
        return f"{prefix}***[{text_hash}]"
    return f"***[{text_hash}]"


def scan_content(
    content: str,
    file_path: str = "",
    commit_hash: str = "",
) -> list[SecretFinding]:
    """Scan content for secrets."""
    if not isinstance(content, str):
        return []

    findings = []

    for pattern in SECRET_PATTERNS:
        for match in pattern.pattern.finditer(content):
            matched_text = match.group(0)

            # Get line number
            line_number = content[: match.start()].count("\n") + 1

            # Get context (surrounding text)
            start = max(0, match.start() - 20)
            end = min(len(content), match.end() + 20)
            context = content[start:end].replace("\n", " ")

            findings.append(
                SecretFinding(
                    pattern_name=pattern.name,
                    description=pattern.description,
                    severity=pattern.severity,
                    file_path=file_path,
                    commit_hash=commit_hash,
                    line_number=line_number,
                    matched_text=redact_secret(matched_text),
                    context=redact_secret(context, 10),
                )
            )

    return findings


def is_sensitive_file(file_path: str) -> bool:
    """Check if a file path matches sensitive file patterns."""
    name = Path(file_path).name

    for pattern in SENSITIVE_FILES:
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(file_path, pattern):
            return True

    return False


def get_file_risk_level(file_path: str) -> str:
    """Get risk level for a file path."""
    if is_sensitive_file(file_path):
        return "high"

    # Check extensions
    high_risk_extensions = [".pem", ".key", ".p12", ".pfx", ".env"]
    medium_risk_extensions = [".json", ".yml", ".yaml", ".xml", ".conf", ".cfg"]

    ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""

    if ext in high_risk_extensions:
        return "high"
    if ext in medium_risk_extensions:
        return "medium"

    return "low"
