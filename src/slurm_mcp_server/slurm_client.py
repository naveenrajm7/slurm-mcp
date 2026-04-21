"""
Slurm REST Client
=================

Thin async HTTP client for the Slurm REST API.  All tools in server.py call
through this client so auth, base URL construction, and error handling live in
one place.

Each instance is scoped to one user's JWT — instantiated per-request via
auth.get_slurm_client().
"""

from typing import Any

import httpx

from slurm_mcp_server.config import SlurmConfig


class SlurmRestClient:
    """
    Async HTTP client for the Slurm REST API.

    Accepts a per-request username and JWT token rather than reading from
    the shared config, so each user's requests are authenticated with their
    own credentials and subject to Slurm's per-user access controls.

    Usage (in a tool):
        username, slurm = get_slurm_client()
        jobs = await slurm.get("jobs")
    """

    def __init__(self, base_url: str, config: SlurmConfig, username: str, token: str) -> None:
        self._version = config.api_version
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "X-SLURM-USER-NAME": username,
                "X-SLURM-USER-TOKEN": token,
            },
            timeout=config.timeout,
            verify=config.verify_ssl,
        )

    def _url(self, resource: str) -> str:
        """Build versioned Slurm REST URL: /slurm/<version>/<resource>/"""
        resource = resource.strip("/")
        return f"/slurm/{self._version}/{resource}/"

    def _db_url(self, resource: str) -> str:
        """Build versioned SlurmDB REST URL: /slurmdb/<version>/<resource>/"""
        resource = resource.strip("/")
        return f"/slurmdb/{self._version}/{resource}/"

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
        if response.is_error:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} from {url}: {response.text}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def get_db(
        self,
        resource: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        GET a SlurmDB resource and return the parsed JSON body.

        Args:
            resource: Resource path without version prefix (e.g. "jobs").
            params:   Optional query parameters forwarded to the API.
        """
        url = self._db_url(resource)
        response = await self._http.get(url, params=params)
        if response.is_error:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} from {url}: {response.text}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def aclose(self) -> None:
        await self._http.aclose()
