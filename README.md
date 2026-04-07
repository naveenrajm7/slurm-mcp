# Slurm MCP Server

A read-only [Model Context Protocol](https://modelcontextprotocol.io/) server for the Slurm workload manager. It enables you to query jobs, nodes, partitions, and reservations on your cluster directly via LLMs that support MCP.

## Tools

| Tool | Description |
|------|-------------|
| `slurm_list_jobs` | List jobs with optional filtering by user, state, partition, or node |
| `slurm_get_job` | Get full details for a single job by ID |
| `slurm_list_nodes` | List nodes with optional filtering by partition, state, or GPU type |
| `slurm_get_node` | Get full details for a single node by name |
| `slurm_list_partitions` | List all partitions (queues) and their limits |
| `slurm_list_reservations` | List scheduled reservations and maintenance windows |

All tools support a `fields` parameter to limit which fields are returned, reducing token usage for large clusters.

## Usage

1. Obtain a Slurm JWT token:

    ```bash
    # Generate a token valid for 24 hours
    scontrol token lifespan=86400
    ```

2. Install dependencies:

    ```bash
    uv sync
    ```

3. Verify the server runs:

    ```bash
    SLURM_BASE_URL=http://your-slurm-host:6820 \
    SLURM_JWT_TOKEN=<your-jwt-token> \
    uv run slurm-mcp
    ```

4. Add to your LLM client. See below for examples with Claude Code.

### Claude Code

#### Stdio Transport (local)

```bash
claude mcp add --transport stdio slurm \
  --env SLURM_BASE_URL=http://your-slurm-host:6820 \
  --env SLURM_JWT_TOKEN=<your-jwt-token> \
  -- uv --directory /path/to/slurm run slurm-mcp
```

Or add directly to `.mcp.json`:

```json
{
  "mcpServers": {
    "slurm": {
      "command": "/path/to/slurm/.venv/Scripts/slurm-mcp.exe",
      "args": [],
      "env": {
        "SLURM_BASE_URL": "http://your-slurm-host:6820",
        "SLURM_JWT_TOKEN": "<your-jwt-token>"
      }
    }
  }
}
```

#### SSE Transport (remote / k8s)

When the server is deployed in Kubernetes:

```bash
claude mcp add --transport sse slurm http://slurm-mcp.your-cluster/sse
```

### Example queries

```text
> Show me all running jobs on the cluster
> Which jobs are pending in the gpu partition?
> What are the resource details for job 12345?
> List all idle GPU nodes
> Which nodes have GPUs?
> Are there any maintenance reservations this week?
> What partitions are available and what are their time limits?
> Show me all my jobs
```

## Field Filtering (Token Optimization)

All tools accept a `fields` parameter to return only the fields you need:

```python
# Without fields — full default summary per job
slurm_list_jobs(states=["RUNNING"])

# With fields — minimal view
slurm_list_jobs(states=["RUNNING"], fields=["job_id", "name", "nodes", "tres_alloc_str"])
```

**Common field patterns:**

- **Job overview:** `["job_id", "name", "user_name", "job_state", "partition", "nodes"]`
- **GPU allocation:** `["job_id", "name", "tres_alloc_str", "gres_detail", "nodes"]`
- **Node GPU check:** `["name", "state", "gres", "gres_used", "free_memory"]`
- **Partition limits:** `["name", "maximum_time", "total_nodes", "allow_accounts"]`

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SLURM_BASE_URL` | `http://localhost:6820` | Yes | Slurm REST API base URL |
| `SLURM_JWT_TOKEN` | — | Yes | JWT bearer token |
| `SLURM_API_VERSION` | `v0.0.42` | No | API version string |
| `SLURM_TIMEOUT` | `30` | No | Request timeout in seconds |
| `MCP_PORT` | — | No | Set to enable SSE transport on this port (e.g. `8000`) |

### Example `.env` file

```env
SLURM_BASE_URL=http://your-slurm-host:6820
SLURM_JWT_TOKEN=eyJhbGci...
SLURM_API_VERSION=v0.0.42
SLURM_TIMEOUT=30
```

## Docker

```bash
# Build
docker build -t slurm-mcp .

# Run (SSE transport, port 8000 is the default)
docker run --rm \
  -e SLURM_BASE_URL=http://your-slurm-host:6820 \
  -e SLURM_JWT_TOKEN=<your-jwt-token> \
  -p 8000:8000 \
  slurm-mcp
```

The server will be accessible at `http://localhost:8000/sse` for MCP clients.

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: slurm-mcp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: slurm-mcp
  template:
    metadata:
      labels:
        app: slurm-mcp
    spec:
      containers:
        - name: slurm-mcp
          image: slurm-mcp:latest
          ports:
            - containerPort: 8000
          env:
            - name: SLURM_BASE_URL
              value: "http://your-slurm-host:6820"
            - name: SLURM_JWT_TOKEN
              valueFrom:
                secretKeyRef:
                  name: slurm-mcp-secret
                  key: jwt-token
```

```bash
kubectl create secret generic slurm-mcp-secret \
  --from-literal=jwt-token=<your-jwt-token>
```

## Slurm REST API Notes

- Server-side filtering is limited to `update_time` (UNIX timestamp) and `flags`. All filtering by state, partition, user, node, and GPU type is applied client-side in this server.
- Slurm encodes many numeric values as `{"set": true, "number": 42}`. This server unwraps those before returning results to the LLM.
- JWT tokens expire. Refresh with `scontrol token lifespan=<seconds>` and update `SLURM_JWT_TOKEN`.
