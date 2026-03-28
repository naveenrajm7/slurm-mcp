"""
Response Utilities
==================

Helpers for unwrapping Slurm's dict-wrapped numeric values and timestamps,
and for projecting only the fields an LLM caller cares about.
"""

from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Slurm value unwrappers
# ---------------------------------------------------------------------------

def _unwrap(value: Any, default: Any = None) -> Any:
    """
    Unwrap a Slurm dict-wrapped scalar.

    Slurm REST responses encode many numbers as {"set": true, "number": 42}
    or {"set": false, "number": 0}.  This helper returns the plain value.
    """
    if isinstance(value, dict):
        if not value.get("set", True):
            return default
        return value.get("number", default)
    return value


def _fmt_ts(ts: Any) -> str | None:
    """Convert a Slurm UNIX timestamp (int or dict-wrapped) to ISO-8601."""
    ts = _unwrap(ts)
    if ts and ts > 0:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
    return None


def _fmt_duration(seconds: Any) -> str | None:
    """Convert seconds to HH:MM:SS string."""
    seconds = _unwrap(seconds)
    if not seconds or seconds <= 0:
        return None
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Field projection
# ---------------------------------------------------------------------------

def project(obj: dict, fields: list[str] | None) -> dict:
    """
    Return only the requested keys from *obj*.

    If *fields* is None or empty, return the full object.
    Missing keys are silently omitted (not returned as None).
    """
    if not fields:
        return obj
    return {k: obj[k] for k in fields if k in obj}


def project_list(objects: list[dict], fields: list[str] | None) -> list[dict]:
    """Apply :func:`project` to every item in a list."""
    if not fields:
        return objects
    return [project(o, fields) for o in objects]


# ---------------------------------------------------------------------------
# Job normalisation
# ---------------------------------------------------------------------------

# Fields surfaced by default when fields=None is passed to list_jobs.
# Deliberately small to keep responses context-window-friendly.
JOB_SUMMARY_FIELDS = [
    "job_id",
    "name",
    "user_name",
    "job_state",
    "partition",
    "nodes",
    "node_count",
    "cpus",
    "submit_time",
    "start_time",
    "end_time",
    "time_limit",
    "work_dir",
    "standard_output",
    "standard_error",
    "tres_req_str",
    "tres_alloc_str",
    "comment",
    "reason",
    "exit_code",
]

# Fields surfaced by default for a single job (get_job).  Wider than summary.
JOB_DETAIL_FIELDS = JOB_SUMMARY_FIELDS + [
    "batch_host",
    "command",
    "current_working_directory",
    "environment",
    "job_resources",
    "gres_detail",
    "licenses",
    "mail_user",
    "mail_type",
    "mcs_label",
    "num_tasks",
    "tasks_per_node",
    "array_job_id",
    "array_task_id",
    "array_task_string",
    "priority",
    "qos",
    "requeue",
    "restart_cnt",
    "resv_name",
    "script",
    "sockets_per_board",
    "sockets_per_node",
    "cores_per_socket",
    "threads_per_core",
    "minimum_cpus_per_node",
    "minimum_memory_per_node",
    "minimum_tmp_disk_per_node",
]


def normalise_job(raw: dict, fields: list[str] | None = None) -> dict:
    """
    Flatten a raw Slurm job object into a clean dict.

    Unwraps dict-wrapped numbers, converts timestamps, and applies field
    projection.  When *fields* is None, uses JOB_SUMMARY_FIELDS.
    """
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and set(val.keys()) <= {"set", "number", "infinite"}:
            out[key] = _unwrap(val)
        else:
            out[key] = val

    # Convert common timestamps
    for ts_field in ("submit_time", "start_time", "end_time", "eligible_time",
                     "accrue_time", "last_sched_evaluation", "preemptable_time"):
        if ts_field in out:
            out[ts_field] = _fmt_ts(out[ts_field])

    # Unwrap exit_code
    if "exit_code" in out and isinstance(out["exit_code"], dict):
        rc = out["exit_code"]
        out["exit_code"] = rc.get("return_code", {}).get("number") if isinstance(rc.get("return_code"), dict) else rc.get("return_code")

    effective_fields = fields if fields is not None else JOB_SUMMARY_FIELDS
    return project(out, effective_fields)


# ---------------------------------------------------------------------------
# Node normalisation
# ---------------------------------------------------------------------------

NODE_SUMMARY_FIELDS = [
    "name",
    "state",
    "reason",
    "reason_changed_at",
    "reason_set_by_user",
    "partitions",
    "cpus",
    "real_memory",
    "free_memory",
    "cores",
    "threads",
    "sockets",
    "gres",
    "gres_used",
    "tres_fmt_str",
    "tres_used",
    "operating_system",
    "arch",
    "version",
    "address",
    "hostname",
    "alloc_cpus",
    "idle_cpus",
    "alloc_memory",
    "comment",
    "extra",
]


def normalise_node(raw: dict, fields: list[str] | None = None) -> dict:
    """Flatten a raw Slurm node object."""
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and set(val.keys()) <= {"set", "number", "infinite"}:
            out[key] = _unwrap(val)
        else:
            out[key] = val

    for ts_field in ("reason_changed_at", "last_busy", "boot_time", "slurmd_start_time"):
        if ts_field in out:
            out[ts_field] = _fmt_ts(out[ts_field])

    effective_fields = fields if fields is not None else NODE_SUMMARY_FIELDS
    return project(out, effective_fields)


# ---------------------------------------------------------------------------
# Partition normalisation
# ---------------------------------------------------------------------------

PARTITION_SUMMARY_FIELDS = [
    "name",
    "state",
    "total_nodes",
    "total_cpus",
    "nodes",
    "node_sets",
    "maximum_nodes",
    "minimum_nodes",
    "maximum_cpus_per_node",
    "default_memory_per_node",
    "maximum_memory_per_node",
    "default_time",
    "maximum_time",
    "minimum_time",
    "preemption_mode",
    "priority_job_factor",
    "priority_tier",
    "over_subscribe",
    "allow_groups",
    "allow_accounts",
    "allow_qos",
    "deny_accounts",
    "deny_qos",
    "tres_fmt_str",
    "flags",
]


def normalise_partition(raw: dict, fields: list[str] | None = None) -> dict:
    """Flatten a raw Slurm partition object."""
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and set(val.keys()) <= {"set", "number", "infinite"}:
            out[key] = _unwrap(val)
        else:
            out[key] = val

    effective_fields = fields if fields is not None else PARTITION_SUMMARY_FIELDS
    return project(out, effective_fields)


# ---------------------------------------------------------------------------
# Reservation normalisation
# ---------------------------------------------------------------------------

RESERVATION_SUMMARY_FIELDS = [
    "name",
    "state",
    "start_time",
    "end_time",
    "duration",
    "nodes",
    "node_count",
    "core_count",
    "license_count",
    "accounts",
    "users",
    "flags",
    "features",
    "tres_str",
    "partition",
    "comment",
    "groups",
    "burst_buffer",
    "watts",
    "max_start_delay",
    "node_inx",
    "core_specializations",
    "purge_completed",
]


def normalise_reservation(raw: dict, fields: list[str] | None = None) -> dict:
    """Flatten a raw Slurm reservation object."""
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and set(val.keys()) <= {"set", "number", "infinite"}:
            out[key] = _unwrap(val)
        else:
            out[key] = val

    for ts_field in ("start_time", "end_time"):
        if ts_field in out:
            out[ts_field] = _fmt_ts(out[ts_field])

    effective_fields = fields if fields is not None else RESERVATION_SUMMARY_FIELDS
    return project(out, effective_fields)
