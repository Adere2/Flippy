import base64
import json
import os
import subprocess
import tempfile
from typing import Optional

from langchain_core.tools import tool

from src.tools.fuzzball_auth import ApiException, fuzzball, get_api_instance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_status(status: str) -> str:
    """Strip the STAGE_STATUS_ prefix for cleaner display."""
    if not status:
        return "UNKNOWN"
    return status.replace("STAGE_STATUS_", "")


def _fmt_time(dt) -> str:
    """Format a datetime object to a readable string."""
    if not dt:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Workflow Management Tools
# ---------------------------------------------------------------------------


@tool
def list_workflows(
    status: Optional[str] = None, scope: str = "SCOPE_USER", limit: int = 10
) -> str:
    """
    List Fuzzball workflows.

    Args:
        status: Filter by status (e.g., 'STARTED', 'FINISHED', 'FAILED', 'CANCELED').
        scope: 'SCOPE_USER' for your workflows, 'SCOPE_GROUP' for account-wide.
        limit: Number of workflows to return (default 10).
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.WorkflowServiceApi(client)

        # Build requested_statuses list if status is provided
        # (the old `status` param is deprecated; use `requested_statuses` instead)
        requested_statuses = None
        if status:
            if not status.startswith("STAGE_STATUS_"):
                status = f"STAGE_STATUS_{status}"
            requested_statuses = [status]

        response = api.list_workflows(
            scope=scope,
            requested_statuses=requested_statuses,
            page_size=limit,
            order_by="createTime desc",
        )
        workflows = getattr(response, "workflows", []) or []

        if not workflows:
            return "No workflows found matching the criteria."

        # Prepare data for aligned table
        headers = ["ID", "Name", "User", "Status", "Created"]
        rows = []

        # Calculate max width for each column to align nicely (min 3 chars for dashes)
        widths = [max(len(h), 3) for h in headers]

        for wf in workflows:
            wf_id = f"`{getattr(wf, 'id', 'N/A')}`"
            name = getattr(wf, "name", "") or ""
            user_val = (
                getattr(wf, "email", None) or getattr(wf, "user_id", "N/A") or "N/A"
            )
            wf_status = f"**{_fmt_status(getattr(wf, 'status', ''))}**"
            created = _fmt_time(getattr(wf, "create_time", None))

            row = [wf_id, name, user_val, wf_status, created]
            rows.append(row)

            # Update widths
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        # Build the Markdown table
        lines = [f"## Workflows (Scope: {scope})"]

        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        lines.append(f"| {header_line} |")

        # Separator
        sep_line = " | ".join("-" * w for w in widths)
        lines.append(f"| {sep_line} |")

        # Rows
        for row in rows:
            row_line = " | ".join(val.ljust(w) for val, w in zip(row, widths))
            lines.append(f"| {row_line} |")

        return "\n".join(lines)

    except ApiException as e:
        return f"API error {e.status} while listing workflows: {e.reason}"
    except Exception as e:
        return f"Failed to list workflows: {str(e)}"


@tool
def get_workflow(workflow_id: str) -> str:
    """
    Retrieve full details of a specific workflow, including its Fuzzfile (specification).
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.WorkflowServiceApi(client)
        wf = api.get_workflow(workflow_id)

        lines = [f"## Workflow: {getattr(wf, 'name', 'Unnamed')} (`{workflow_id}`)"]
        lines.append(f"- **Status:** {_fmt_status(getattr(wf, 'status', ''))}")
        lines.append(f"- **User ID:** {getattr(wf, 'user_id', 'N/A')}")
        lines.append(f"- **Created:** {_fmt_time(getattr(wf, 'create_time', None))}")
        lines.append(f"- **Started:** {_fmt_time(getattr(wf, 'start_time', None))}")
        lines.append(f"- **Ended:** {_fmt_time(getattr(wf, 'end_time', None))}")

        error = getattr(wf, "error", None)
        if error:
            lines.append(f"- **Error:** {error}")

        # Extract Fuzzfile - try raw_specification first, then base64 specification
        spec_content = getattr(wf, "raw_specification", "")
        if not spec_content:
            spec_bytes = getattr(wf, "specification", None)
            if spec_bytes:
                try:
                    # Often the 'bytes' field in the OpenAPI SDK is a base64 string or already decoded
                    if isinstance(spec_bytes, str):
                        decoded = base64.b64decode(spec_bytes).decode("utf-8")
                    else:
                        decoded = spec_bytes.decode("utf-8")

                    # If it's JSON-encoded Fuzzfile, prettify it as YAML-like
                    try:
                        parsed = json.loads(decoded)
                        spec_content = json.dumps(parsed, indent=2)
                    except json.JSONDecodeError:
                        spec_content = decoded
                except Exception:
                    spec_content = (
                        "[Binary specification data present but could not be decoded]"
                    )

        if spec_content:
            lines.append("\n### Fuzzfile Specification")
            lines.append(f"```yaml\n{spec_content.strip()}\n```")

        return "\n".join(lines)

    except ApiException as e:
        detail = getattr(e, "body", "")
        # Handle cases where Fuzzball returns 500 for Not Found
        if e.status == 500 and "not found" in str(detail).lower():
            return f"❌ Workflow '{workflow_id}' was not found in the current Fuzzball context/account."

        return f"API error {e.status} for workflow '{workflow_id}': {e.reason}\nDetail: {detail}"
    except Exception as e:
        return f"Failed to get workflow: {str(e)}"


@tool
def get_workflow_status(workflow_id: str) -> str:
    """
    Get the live status of a workflow and a breakdown of its individual jobs/stages.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.WorkflowServiceApi(client)
        status_resp = api.get_workflow_status(workflow_id)

        overall_status = _fmt_status(getattr(status_resp, "workflow_status", ""))
        lines = [f"## Workflow Status: {overall_status} (`{workflow_id}`)"]

        # The breakdown of jobs is inside the 'status' (WorkflowPlan) field's 'stages' property
        plan = getattr(status_resp, "status", None)
        jobs = getattr(plan, "stages", []) if plan else []

        if jobs:
            lines.append("\n### Job Breakdown")
            lines.append("| Job Name | Status | Type | Started | Ended |")
            lines.append("|---|---|---|---|---|")
            for job in jobs:
                name = getattr(job, "name", "N/A")
                job_status = _fmt_status(getattr(job, "status", ""))
                kind = str(getattr(job, "kind", "N/A")).replace("STAGE_KIND_", "")
                start = _fmt_time(getattr(job, "start_time", None))
                end = _fmt_time(getattr(job, "end_time", None))
                lines.append(
                    f"| {name} | **{job_status}** | {kind} | {start} | {end} |"
                )

                job_error = getattr(job, "error", None)
                if job_error:
                    lines.append(f"  - *Error in {name}: {job_error}*")
        else:
            lines.append("\nNo job details available yet.")

        return "\n".join(lines)

    except ApiException as e:
        return f"API error {e.status} fetching status for '{workflow_id}': {e.reason}"
    except Exception as e:
        return f"Failed to get workflow status: {str(e)}"


@tool
def validate_workflow(fuzzfile_yaml: str) -> str:
    """
    Validate a Fuzzfile YAML string using the 'fuzzball' CLI.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
            tmp.write(fuzzfile_yaml)
            tmp_path = tmp.name

        # fuzzball workflow validate <file>
        result = subprocess.run(
            ["fuzzball", "workflow", "validate", tmp_path],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return "✅ Fuzzfile is valid."
        else:
            return f"❌ Validation failed:\n{result.stderr}\n{result.stdout}"

    except Exception as e:
        return f"Failed to run validation CLI: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@tool
def start_workflow(fuzzfile_yaml: str, name: Optional[str] = None) -> str:
    """
    Submit and start a new workflow using the 'fuzzball' CLI.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
            tmp.write(fuzzfile_yaml)
            tmp_path = tmp.name

        cmd = ["fuzzball", "workflow", "start", tmp_path, "--json"]
        if name:
            cmd.extend(["--name", name])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                out = json.loads(result.stdout)
                wf_id = out.get("id") or out.get("workflow_id")
                if wf_id:
                    return f"✅ Workflow started successfully!\n- **Workflow ID:** `{wf_id}`\n- Use `get_workflow_status` to track progress."
                else:
                    return f"✅ Workflow started successfully!\nOutput: {result.stdout}"
            except json.JSONDecodeError:
                return f"✅ Workflow started successfully!\nOutput: {result.stdout}"
        else:
            return f"❌ Failed to start workflow:\n{result.stderr}\n{result.stdout}"

    except Exception as e:
        return f"Failed to run start CLI: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@tool
def stop_workflow(workflow_id: str) -> str:
    """
    Stop (cancel) a running workflow.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.WorkflowServiceApi(client)
        api.stop_workflow(workflow_id)
        return f"🛑 Workflow `{workflow_id}` has been requested to stop."
    except ApiException as e:
        return (
            f"API error {e.status} while stopping workflow '{workflow_id}': {e.reason}"
        )
    except Exception as e:
        return f"Failed to stop workflow: {str(e)}"


@tool
def restart_workflow(workflow_id: str) -> str:
    """
    Restart a workflow by retrieving its original specification and submitting it as a new run.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.WorkflowServiceApi(client)

        # 1. Get the existing workflow to get the spec
        old_wf = api.get_workflow(workflow_id)
        spec = getattr(old_wf, "raw_specification", None)
        name = getattr(old_wf, "name", "Restarted Workflow")

        # Try to decode binary specification if raw is missing
        if not spec:
            spec_bytes = getattr(old_wf, "specification", None)
            if spec_bytes:
                try:
                    if isinstance(spec_bytes, str):
                        decoded = base64.b64decode(spec_bytes).decode("utf-8")
                    else:
                        decoded = spec_bytes.decode("utf-8")

                    # Try to see if it is valid YAML/JSON
                    spec = decoded
                except Exception:
                    pass

        if not spec:
            return f"❌ Cannot restart workflow `{workflow_id}`: Original specification not found."

        # 2. Start a new one
        return start_workflow.invoke(
            {"fuzzfile_yaml": spec, "name": f"Restart of {name}"}
        )

    except ApiException as e:
        return f"API error {e.status} during restart attempt: {e.reason}"
    except Exception as e:
        return f"Failed to restart workflow: {str(e)}"


@tool
def get_workflow_logs(workflow_id: str, job_name: str, tail: int = 100) -> str:
    """
    Retrieve logs for a specific job within a workflow.

    Args:
        workflow_id: The ID of the workflow.
        job_name: The name of the job (container) within the workflow.
        tail: Number of lines to retrieve from the end of the logs (default 100).
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.WorkflowGatewayServiceApi(client)

        # Note: log response is often a stream. We'll attempt to collect the result.
        # Based on SDK docs, workflow_gateway_log returns StreamResultOfWorkflowGatewayLogResponse
        response = api.workflow_gateway_log(
            workflow_id, name=job_name, show_tail=tail, show_follow=False
        )

        # print(response)
        log_content = ""

        # The response is a single object (StreamResultOfWorkflowGatewayLogResponse) when show_follow=False
        try:
            result = getattr(response, "result", None)
            if result and hasattr(result, "output"):
                output = result.output
                if output:
                    # Output is typically a base64 encoded string
                    if isinstance(output, str):
                        try:
                            # Try to decode base64
                            decoded_bytes = base64.b64decode(output)
                            log_content = decoded_bytes.decode(
                                "utf-8", errors="replace"
                            )
                        except Exception:
                            # If not base64 or failed, just use as is
                            log_content = output
                    elif isinstance(output, bytes):
                        log_content = output.decode("utf-8", errors="replace")
                    else:
                        log_content = str(output)
        except Exception as err:
            return f"Error processing log response: {str(err)}"
        if not log_content.strip():
            return f"No logs found for job `{job_name}` in workflow `{workflow_id}`."

        return f"### Logs for job `{job_name}`\n```\n{log_content}\n```"

    except ApiException as e:
        return f"API error {e.status} fetching logs: {e.reason}"
    except Exception as e:
        return f"Failed to retrieve logs: {str(e)}"


# ---------------------------------------------------------------------------
# Smoke-test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Testing tool for looking at job logs.
    # print("--- testing get_workflow_logs ---")
    # print(
    #     get_workflow_logs.invoke(
    #         {
    #             "workflow_id": "0007581a-0c7e-4db6-8ceb-1175c2fde782",
    #             "job_name": "geogrid",
    #             "tail": 3,
    #         }
    #     )
    # )
    #

    # Testing tool for stopping workflow
    # print("--- testing stop_workflow ---")
    # print(stop_workflow.invoke({"workflow_id": "4395e80b-922d-452b-a3cb-fae3f5dfcb0a"}))
    #
    # Testing tool for listing workflows
    print("--- testing list_workflows ---")
    print(list_workflows.invoke({'status': 'STARTED', 'scope': 'SCOPE_GROUP'}))
    # Testing tool for validating fuzzfile
    # print("--- testing start_workflow ---")
    # print(start_workflow.invoke({"fuzzfile_yaml": valid_fuzzfile, "name": "Agent Test WF"}))
    # test retrying a workflow by ID (make sure to replace with an actual workflow ID from your context)
    # print("--- testing restart_workflow ---")
    # print(restart_workflow.invoke({"workflow_id": "57b5962b-6668-40c8-a91c-d2055f7342de"}))
    # # test checking workflow status and logs
    # print("--- testing get_workflow_status ---")
    # print(get_workflow_status.invoke({"workflow_id": "57b5962b-6668-40c8-a91c-d2055f7342de"}))
    # print("--- testing get_workflow_logs ---")
    # print(get_workflow_logs.invoke({"workflow_id": "57b5962b-6668-40c8-a91c-d2055f7342de", "job_name": "generate-data", "tail": 10}))
