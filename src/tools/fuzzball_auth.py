import subprocess
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Optional SDK import – centralised here so every tool file can import
# `fuzzball` and `ApiException` from this module instead of repeating the
# try/except block themselves.
# ---------------------------------------------------------------------------


class _FallbackApiException(Exception):
    """Stub for fuzzball.ApiException when the SDK is not installed.

    Keeping `.status` and `.reason` as class attributes means that every
    ``except ApiException as e:`` block in the tool files can safely
    reference ``e.status`` and ``e.reason`` regardless of whether the real
    SDK is present.
    """

    status: int = 0
    reason: str = "fuzzball SDK not installed"


try:
    import fuzzball
    from fuzzball.exceptions import ApiException
except ImportError:
    fuzzball = None  # type: ignore[assignment]
    ApiException = _FallbackApiException  # type: ignore[assignment, misc]
    print("Warning: fuzzball SDK not installed. Fuzzball tools will not work.")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def trigger_cli_login() -> None:
    """Executes the Fuzzball CLI login command interactively.

    Blocks until the browser-based login flow completes, piping output
    directly to the user's terminal.
    """
    print("\n🔑 Fuzzball authentication required or token expired.")
    print("⏳ Triggering CLI login. Please check your browser...\n")
    try:
        subprocess.run(["fuzzball", "context", "login"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Fuzzball CLI login failed with exit code {e.returncode}.")
    except FileNotFoundError:
        raise RuntimeError("Fuzzball CLI is not installed or not found in PATH.")


def load_fuzzball_config() -> dict:
    """Safely loads the Fuzzball config YAML using pathlib."""
    config_path = Path("~/.config/fuzzball/config.yaml").expanduser()

    if not config_path.exists():
        raise FileNotFoundError("Could not locate Fuzzball config.")

    with config_path.open("r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse Fuzzball config at {config_path}: {exc}")


def get_api_instance(skip_auth: bool = False):
    """Initialise and return an authenticated Fuzzball API client.

    Args:
        skip_auth: When ``True`` the CLI login step is skipped (useful when
                   a valid token is already present in the config file).

    Returns:
        A configured ``fuzzball.ApiClient`` ready to be passed to any
        ``fuzzball.*ServiceApi`` constructor.
    """
    if not skip_auth:
        trigger_cli_login()

    config = load_fuzzball_config()

    api_config = fuzzball.Configuration()
    active_context = config.get("activeContext")

    for context in config.get("contexts", []):
        if context.get("name") == active_context:
            api_config.host = "https://{}/v3".format(context["address"])
            # The OpenAPI SDK uses access_token natively instead of api_key dicts
            api_config.access_token = context["auth"]["credentials"]["token"]

    client = fuzzball.ApiClient(api_config)
    client.set_default_header("Authorization", f"Bearer {api_config.access_token}")

    return client


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_api_instance(skip_auth=False)
    api = fuzzball.WorkflowServiceApi(client)

    try:
        workflows = api.list_workflows()
        for wf in workflows.workflows:
            print(wf.id, getattr(wf, "name", "Unknown"), wf.status)
    except ApiException as e:
        print(f"API error {e.status}: {e.reason}")
