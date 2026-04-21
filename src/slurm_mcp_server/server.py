"""
Alola Slurm MCP Server
======================

MCP tools for querying the Slurm workload manager REST API.

Tools return structured dicts (not formatted strings) so the LLM can reason
across multiple calls — e.g. find all RUNNING jobs on gfx1151 nodes.

Slurm REST API notes (OpenAPI v0.0.42):
- Server-side filtering is minimal: only update_time (UNIX timestamp) and flags.
- All other filtering (by state, partition, user, node, GPU arch) is done
  client-side here, keeping the tool interface clean and consistent.
- Slurm encodes many numbers as {"set": true, "number": 42}; utils.normalise_*
  unwraps these before returning to the LLM.
"""

import os
from typing import Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from slurm_mcp_server.auth import SlurmBearerAuthProvider, get_slurm_client
from slurm_mcp_server.config import config
from slurm_mcp_server.utils import (
    normalise_db_job,
    normalise_job,
    normalise_node,
    normalise_partition,
    normalise_reservation,
)

mcp = FastMCP(name="slurm-mcp-server", auth=SlurmBearerAuthProvider())


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@mcp.tool(
    description="""
    List jobs on the cluster, sorted most-recent first.

    Returns a list of job objects.  By default each object contains:
    job_id, name, user_name, job_state, partition, nodes, node_count, cpus,
    submit_time, start_time, end_time, time_limit, work_dir,
    standard_output, standard_error, tres_req_str, tres_alloc_str,
    comment, reason, exit_code.

    Filtering (all optional, applied client-side):
    - user:      Return only jobs owned by this username.
    - mine:      If True, return only your own jobs (resolved from JWT token).
    - states:    List of job state strings to include.
                 Common values: RUNNING, PENDING, COMPLETED, FAILED,
                 CANCELLED, TIMEOUT, NODE_FAIL, PREEMPTED, SUSPENDED,
                 COMPLETING, OUT_OF_MEMORY.
                 Example: ["RUNNING", "PENDING"]
    - partition: Return only jobs in this partition.
    - node:      Return only jobs allocated to this node name.
    - limit:     Maximum number of jobs to return (default 20, max 500).

    Field projection:
    - fields: List of field names to include in each returned object.
              Pass [] or omit to get the default summary fields.
              Pass ["job_id", "name", "job_state", "nodes"] for a minimal view.

    Two-step pattern for cross-resource queries:
      # Find all RUNNING jobs on nodes with gfx1151 GPUs:
      nodes = slurm_list_nodes(gres_filter="gpu:nvidia:1")
      node_names = [n["name"] for n in nodes]
      jobs = slurm_list_jobs(states=["RUNNING"], node=node_names[0])

    Timestamps are returned as ISO-8601 strings (e.g. "2025-03-14T09:00:00").
    """
)
async def slurm_list_jobs(
    user: Optional[str] = None,
    mine: bool = False,
    states: Optional[list[str]] = None,
    partition: Optional[str] = None,
    node: Optional[str] = None,
    limit: int = 20,
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """List jobs on the cluster with optional client-side filtering."""
    try:
        username, slurm = get_slurm_client()
        data = await slurm.get("jobs")
        jobs: list[dict] = data.get("jobs", [])

        # Client-side filters — 'mine' uses the authenticated user's username
        effective_user = username if mine else user
        if effective_user:
            jobs = [j for j in jobs if j.get("user_name") == effective_user]
        if states:
            upper = [s.upper() for s in states]
            jobs = [
                j for j in jobs
                if any(s in upper for s in (j.get("job_state") or []))
            ]
        if partition:
            jobs = [j for j in jobs if j.get("partition") == partition]
        if node:
            jobs = [j for j in jobs if node in (j.get("nodes") or "")]

        # Sort most-recent first, apply limit
        jobs.sort(key=lambda j: (j.get("submit_time") or {}).get("number", 0), reverse=True)
        jobs = jobs[: max(1, min(limit, 500))]

        return [normalise_job(j, fields) for j in jobs]

    except Exception as e:
        raise ToolError(f"Failed to list jobs: {e!r}") from e


@mcp.tool(
    description="""
    Get detailed information about a single job by its ID.

    Returns a single job object with extended fields beyond the list summary,
    including: batch_host, command, current_working_directory, gres_detail,
    job_resources, licenses, mail_user, num_tasks, tasks_per_node,
    array_job_id, array_task_id, priority, qos, requeue, resv_name,
    sockets_per_node, cores_per_socket, threads_per_core,
    minimum_memory_per_node, and all summary fields.

    Args:
        job_id: The integer Slurm job ID.
        fields: Optional list of field names to return.
                Omit to get the full default detail set.

    Examples:
        # Get full details for job 12345
        slurm_get_job(12345)

        # Get just the resource allocation
        slurm_get_job(12345, fields=["job_id", "nodes", "tres_alloc_str", "gres_detail"])
    """
)
async def slurm_get_job(
    job_id: int,
    fields: Optional[list[str]] = None,
) -> dict:
    """Get detailed information about a single job."""
    try:
        _, slurm = get_slurm_client()
        data = await slurm.get(f"job/{job_id}")
        jobs: list[dict] = data.get("jobs", [])
        if not jobs:
            raise ToolError(f"Job {job_id} not found.")
        from slurm_mcp_server.utils import JOB_DETAIL_FIELDS
        effective_fields = fields if fields is not None else JOB_DETAIL_FIELDS
        return normalise_job(jobs[0], effective_fields)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to get job {job_id}: {e!r}") from e


# ---------------------------------------------------------------------------
# SlurmDB jobs (accounting history)
# ---------------------------------------------------------------------------

@mcp.tool(
    description="""
    Query SlurmDB for historical job records (completed, failed, cancelled, etc.).

    Unlike slurm_list_jobs / slurm_get_job — which query the live scheduler and
    only see jobs currently in the queue — this tool queries the Slurm accounting
    database (slurmdbd) and returns jobs from any point in history.

    Returns a list of job objects.  By default each object contains:
    job_id, name, user, group, account, state, partition, nodes, node_count,
    cpus, submit, start, end, elapsed, time_limit, exit_code, work_dir,
    tres_req_str, tres_alloc_str, qos, priority, reservation, cluster, comment,
    constraints.  Pass fields=["script"] or fields=["environment"] to include
    those large fields only when needed.

    All filtering parameters are forwarded server-side to slurmdbd (CSV strings
    where noted); client-side limit is applied after.

    Args:
        users:       CSV user list (e.g. "alice,bob").
        job_name:    CSV job name list.
        state:       CSV state list (e.g. "COMPLETED,FAILED").
        partition:   CSV partition name list.
        account:     CSV account list.
        cluster:     CSV cluster list.
        nodes:       Ranged node string (e.g. "gpu[01-04]").
        start_time:  Include jobs that started at or after this UNIX timestamp.
        end_time:    Include jobs that ended at or before this UNIX timestamp.
        qos:         CSV QOS name list.
        reservation: CSV reservation name list.
        skip_steps:  If True, omit per-step detail from each job record.
        show_batch_script:   If True, include the job's batch script.
        show_job_environment: If True, include the job's environment variables.
        limit:       Maximum number of jobs to return (default 20, max 500).
        fields:      Field names to include per job. Omit for default summary.

    Examples:
        # All FAILED jobs for user alice in the last week
        slurmdb_list_jobs(users="alice", state="FAILED", start_time=1700000000)

        # Completed GPU jobs on a specific partition
        slurmdb_list_jobs(state="COMPLETED", partition="gpu", limit=50)
    """
)
async def slurmdb_list_jobs(
    users: Optional[str] = None,
    job_name: Optional[str] = None,
    state: Optional[str] = None,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    cluster: Optional[str] = None,
    nodes: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    qos: Optional[str] = None,
    reservation: Optional[str] = None,
    skip_steps: bool = True,
    show_batch_script: bool = False,
    show_job_environment: bool = False,
    limit: int = 20,
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """Query SlurmDB for historical job accounting records."""
    try:
        _, slurm = get_slurm_client()

        params: dict = {}
        if users:
            params["users"] = users
        if job_name:
            params["job_name"] = job_name
        if state:
            params["state"] = state
        if partition:
            params["partition"] = partition
        if account:
            params["account"] = account
        if cluster:
            params["cluster"] = cluster
        if nodes:
            params["node"] = nodes
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if qos:
            params["qos"] = qos
        if reservation:
            params["reservation"] = reservation
        if skip_steps:
            params["skip_steps"] = "true"
        if show_batch_script:
            params["show_batch_script"] = "true"
        if show_job_environment:
            params["show_job_environment"] = "true"

        data = await slurm.get_db("jobs", params=params)
        jobs: list[dict] = data.get("jobs", [])

        jobs = jobs[: max(1, min(limit, 500))]
        return [normalise_db_job(j, fields) for j in jobs]

    except Exception as e:
        raise ToolError(f"Failed to query SlurmDB jobs: {e!r}") from e


@mcp.tool(
    description="""
    Get detailed information for a specific job from SlurmDB by its ID.

    Unlike slurm_get_job — which queries the live scheduler and only sees jobs
    currently in the queue — this tool queries the Slurm accounting database
    (slurmdbd) and returns full details for any job including completed, failed,
    and cancelled jobs.

    Returns the full job record by default, including: job_id, name, user, group,
    account, state, partition, nodes, node_count, cpus, submit, start, end,
    elapsed, time_limit, exit_code, work_dir, tres_req_str, tres_alloc_str,
    qos, priority, reservation, cluster, comment, constraints, submit_line,
    script, environment, and more.

    Args:
        job_id: The integer Slurm job ID.
        fields: Optional list of field names to return.
                Omit to get the full detail set.

    Examples:
        # Get full details for job 12345
        slurmdb_get_job(12345)

        # Get just the submit line and script
        slurmdb_get_job(12345, fields=["job_id", "name", "submit_line", "script"])
    """
)
async def slurmdb_get_job(
    job_id: int,
    fields: Optional[list[str]] = None,
) -> dict:
    """Get full accounting details for a specific job from SlurmDB."""
    try:
        _, slurm = get_slurm_client()
        data = await slurm.get_db(f"job/{job_id}")
        jobs: list[dict] = data.get("jobs", [])
        if not jobs:
            raise ToolError(f"Job {job_id} not found in SlurmDB.")
        # Pass [] to normalise_db_job so project() returns all fields
        return normalise_db_job(jobs[0], fields if fields is not None else [])
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to get SlurmDB job {job_id}: {e!r}") from e


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@mcp.tool(
    description="""
    List nodes in the cluster with optional filtering.

    Returns a list of node objects.  By default each object contains:
    name, state, reason, partitions, cpus, real_memory, free_memory,
    cores, threads, sockets, gres, gres_used, tres_fmt_str, tres_used,
    operating_system, arch, version, address, hostname,
    alloc_cpus, idle_cpus, alloc_memory.

    Filtering (all optional, applied client-side):
    - partition:   Return only nodes belonging to this partition name.
    - states:      List of node state strings to include.
                   Common values: idle, allocated, mixed, down, drain,
                   draining, fail, failing, future, unknown, completing.
                   Example: ["idle", "mixed"]
    - gres_filter: Return only nodes whose gres field contains this string.
                   Use this to find nodes with specific GPU types:
                   - "gpu"               → any GPU node
                   - "gpu:nvidia"        → any NVIDIA GPU node
                   - "gpu:nvidia:4"      → nodes with exactly 4 NVIDIA GPUs
                   Example: gres_filter="gpu:nvidia"

    Field projection:
    - fields: List of field names to return per node.
              Omit to get the default summary fields.
              Example: ["name", "state", "gres", "partitions"]

    Examples:
        # All nodes
        slurm_list_nodes()

        # Idle GPU nodes only
        slurm_list_nodes(states=["idle"], gres_filter="gpu")

        # Nodes in the 'gpu' partition — minimal view
        slurm_list_nodes(partition="gpu", fields=["name", "state", "gres", "free_memory"])
    """
)
async def slurm_list_nodes(
    partition: Optional[str] = None,
    states: Optional[list[str]] = None,
    gres_filter: Optional[str] = None,
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """List nodes with optional client-side filtering."""
    try:
        _, slurm = get_slurm_client()
        data = await slurm.get("nodes")
        nodes: list[dict] = data.get("nodes", [])

        if partition:
            nodes = [n for n in nodes if partition in (n.get("partitions") or [])]
        if states:
            lower = [s.lower() for s in states]
            nodes = [
                n for n in nodes
                if any(s in lower for s in (n.get("state") or []))
            ]
        if gres_filter:
            nodes = [n for n in nodes if gres_filter in (n.get("gres") or "")]

        return [normalise_node(n, fields) for n in nodes]

    except Exception as e:
        raise ToolError(f"Failed to list nodes: {e!r}") from e


@mcp.tool(
    description="""
    Get detailed information about a single node by name.

    Args:
        node_name: The node hostname as it appears in Slurm (e.g. "n1", "gpu-01").
        fields:    Optional list of field names to return.

    Returns a single node object with all available fields.

    Examples:
        slurm_get_node("node-01")
        slurm_get_node("gpu-01", fields=["name", "state", "gres", "gres_used", "free_memory"])
    """
)
async def slurm_get_node(
    node_name: str,
    fields: Optional[list[str]] = None,
) -> dict:
    """Get detailed information about a single node."""
    try:
        _, slurm = get_slurm_client()
        data = await slurm.get(f"node/{node_name}")
        nodes: list[dict] = data.get("nodes", [])
        if not nodes:
            raise ToolError(f"Node '{node_name}' not found.")
        return normalise_node(nodes[0], fields)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to get node '{node_name}': {e!r}") from e


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

@mcp.tool(
    description="""
    List all Slurm partitions (queues).

    Know your partition names before filtering jobs — use this tool first when
    the user asks about queues, partitions, or where to submit jobs.

    Returns a list of partition objects.  By default each object contains:
    name, state, total_nodes, total_cpus, nodes, node_sets,
    maximum_nodes, minimum_nodes, maximum_cpus_per_node,
    default_memory_per_node, maximum_memory_per_node,
    default_time, maximum_time, minimum_time, preemption_mode,
    priority_job_factor, priority_tier, over_subscribe,
    allow_groups, allow_accounts, allow_qos,
    deny_accounts, deny_qos, tres_fmt_str, flags.

    Filtering (optional, client-side):
    - states: List of partition state strings. Common values: up, down, drain, inactive.

    Field projection:
    - fields: List of field names to return.

    Examples:
        # All partitions
        slurm_list_partitions()

        # Active partitions only, minimal view
        slurm_list_partitions(states=["up"], fields=["name", "total_nodes", "maximum_time", "nodes"])
    """
)
async def slurm_list_partitions(
    states: Optional[list[str]] = None,
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """List all Slurm partitions."""
    try:
        _, slurm = get_slurm_client()
        data = await slurm.get("partitions")
        partitions: list[dict] = data.get("partitions", [])

        if states:
            lower = [s.lower() for s in states]
            partitions = [
                p for p in partitions
                if str(p.get("state", "")).lower() in lower
            ]

        return [normalise_partition(p, fields) for p in partitions]

    except Exception as e:
        raise ToolError(f"Failed to list partitions: {e!r}") from e


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------

@mcp.tool(
    description="""
    List scheduled reservations on the cluster.

    Use this to check for maintenance windows or reserved time blocks before
    submitting jobs or advising users on scheduling.

    Returns a list of reservation objects.  By default each object contains:
    name, state, start_time, end_time, duration, nodes, node_count, core_count,
    accounts, users, flags, features, tres_str, partition, comment, groups.

    Timestamps are ISO-8601 strings.

    Field projection:
    - fields: List of field names to return.

    Examples:
        slurm_list_reservations()
        slurm_list_reservations(fields=["name", "start_time", "end_time", "nodes", "users"])
    """
)
async def slurm_list_reservations(
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """List all Slurm reservations."""
    try:
        _, slurm = get_slurm_client()
        data = await slurm.get("reservations")
        reservations: list[dict] = data.get("reservations", [])
        return [normalise_reservation(r, fields) for r in reservations]
    except Exception as e:
        raise ToolError(f"Failed to list reservations: {e!r}") from e


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    port = int(os.environ.get("MCP_PORT", "0"))
    transport = os.environ.get("MCP_TRANSPORT", "sse")
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    if port:
        mcp.run(transport=transport, host=host, port=port, show_banner=False)
    else:
        mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
