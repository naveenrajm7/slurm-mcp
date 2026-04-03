"""
Slurm MCP Configuration
=======================

Environment-based configuration for the Slurm MCP server.

Environment variables:
    SLURM_BASE_URL:     Slurm REST API base URL. Used as fallback when no X-Slurm-URL
                        header is present (stdio/dev mode). Optional in SSE/k8s mode.
    SLURM_JWT_TOKEN:    User JWT for Slurm authentication.
                        Obtain with: scontrol token username=<you>
                        Used as a fallback when no Authorization header is present (stdio mode).
    SLURM_API_VERSION:  API version string (default: v0.0.42)
    SLURM_TIMEOUT:      Request timeout in seconds (default: 30)
    SLURM_VERIFY_SSL:   Verify SSL certificates when calling Slurm REST API (default: true).
                        Set to false to disable — use only when the CA chain is not trusted.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class SlurmConfig(BaseSettings):
    """Configuration loaded from environment variables."""

    base_url: str = Field(default="")  # optional: overridden per-request via X-Slurm-URL header
    jwt_token: str = Field(default="")    # user JWT — stdio fallback when no Authorization header
    api_version: str = Field(default="v0.0.42")
    timeout: int = Field(default=30)
    verify_ssl: bool = Field(default=True)

    model_config = {
        "env_prefix": "SLURM_",
        "case_sensitive": False,
        "extra": "ignore",
    }


config = SlurmConfig()
