"""
Slurm Authentication
====================

The MCP server is unauthenticated — Slurm's own JWT validation is the security
boundary, exactly as Grafana's service account token is the boundary there.

Two token sources, tried in order:

  1. HTTP Authorization header (SSE/HTTP transport, multi-user):
       Authorization: Bearer <jwt>
     Each client supplies their own JWT; the server passes it straight through.

  2. SLURM_JWT_TOKEN env var (stdio transport, local dev):
     A single static JWT configured in .mcp.json env block.

In both cases the username is decoded from the JWT payload (no signature
verification needed here — Slurm rejects invalid tokens on its own).
"""

import base64
import json

from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token, get_http_headers

from slurm_mcp_server.config import config
from slurm_mcp_server.slurm_client import SlurmRestClient


class SlurmBearerAuthProvider(TokenVerifier):
    """
    Accepts any non-empty Bearer token as a Slurm JWT.

    Slurm validates the JWT itself — we just need FastMCP to process the
    Authorization header so it is available via get_http_headers() in tools.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None
        return AccessToken(token=token, client_id=token, scopes=[], expires_at=None)


def _username_from_token(token: str) -> str:
    """Decode the JWT payload to extract the Slurm username ('sun' claim)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4 % 4)
        claims = json.loads(base64.b64decode(payload))
        return str(claims.get("sun") or claims.get("sub") or "unknown")
    except Exception:
        return "unknown"


def get_slurm_client() -> tuple[str, SlurmRestClient]:
    """
    Return a (username, SlurmRestClient) for the current request.

    Both the Slurm URL and JWT are resolved per-request, with env fallbacks
    for stdio/dev mode:

      X-Slurm-URL header       → per-user cluster URL  (falls back to SLURM_BASE_URL)
      Authorization: Bearer ... → per-user JWT          (falls back to SLURM_JWT_TOKEN)
    """
    headers = get_http_headers()

    base_url = headers.get("x-slurm-url", "") or config.base_url
    if not base_url:
        raise ToolError(
            "No Slurm URL available. Set SLURM_BASE_URL in the environment "
            "or pass X-Slurm-URL in the request header."
        )

    # Prefer the token from the FastMCP auth context (set by SlurmBearerAuthProvider),
    # fall back to env var for stdio/dev mode.
    access_token = get_access_token()
    if access_token and access_token.token:
        token = access_token.token
    elif config.jwt_token:
        token = config.jwt_token
    else:
        raise ToolError(
            "No Slurm JWT available. Set SLURM_JWT_TOKEN in the environment "
            "or supply a Bearer token in the Authorization header.\n"
            "Get a JWT with: scontrol token username=<you>"
        )

    username = _username_from_token(token)
    return username, SlurmRestClient(base_url, config, username, token)
