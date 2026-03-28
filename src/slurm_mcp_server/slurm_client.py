"""
Slurm REST Client
=================

Thin async HTTP client for the Slurm REST API.  All tools in server.py call
through this client so auth, base URL construction, and error handling live in
one place.
"""

from typing import Any

import httpx

from slurm_mcp_server.config import SlurmConfig


class SlurmRestClient:
    """
    Async HTTP client for the Slurm REST API.

    Usage:
        client = SlurmRestClient(config)
        jobs = await client.get("jobs")
        job  = await client.get(f"job/{job_id}")
    """

    def __init__(self, config: SlurmConfig) -> None:
        headers: dict[str, str] = {}
        if config.jwt_token:
            headers["X-SLURM-USER-TOKEN"] = config.jwt_token
        if config.username:
            headers["X-SLURM-USER-NAME"] = config.username

        self._version = config.api_version
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout,
        )

    def _url(self, resource: str) -> str:
        """Build versioned Slurm REST URL: /slurm/<version>/<resource>/"""
        resource = resource.strip("/")
        return f"/slurm/{self._version}/{resource}/"

    async def get(
        self,
        resource: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        GET a Slurm resource and return the parsed JSON body.

        Args:
            resource: Resource path without version prefix (e.g. "jobs", "node/n1").
            params:   Optional query parameters forwarded to the API.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        url = self._url(resource)
        response = await self._http.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._http.aclose()
