"""
Slurm MCP Configuration
=======================

Environment-based configuration for the Slurm MCP server.

Environment variables:
    SLURM_BASE_URL:     Slurm REST API base URL (default: http://localhost:6820)
    SLURM_JWT_TOKEN:    JWT bearer token
    SLURM_API_VERSION:  API version string (default: v0.0.42)
    SLURM_TIMEOUT:      Request timeout in seconds (default: 30)
"""

import base64
import json

from pydantic import Field
from pydantic_settings import BaseSettings


def _jwt_username(token: str) -> str:
    """Extract the Slurm user name (sun claim) from a JWT without verification."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)  # fix padding
        return json.loads(base64.b64decode(payload)).get("sun", "")
    except Exception:
        return ""


class SlurmConfig(BaseSettings):
    """Configuration loaded from environment variables."""

    base_url: str = Field(default="http://localhost:6820")
    jwt_token: str = Field(default="")
    api_version: str = Field(default="v0.0.42")
    timeout: int = Field(default=30)

    @property
    def username(self) -> str:
        """Current user extracted from the JWT token's sun claim."""
        return _jwt_username(self.jwt_token)

    model_config = {
        "env_prefix": "SLURM_",
        "case_sensitive": False,
        "extra": "ignore",
    }


config = SlurmConfig()
