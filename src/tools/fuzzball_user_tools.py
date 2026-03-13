from langchain_core.tools import tool

from src.tools.fuzzball_auth import ApiException, fuzzball, get_api_instance

# ---------------------------------------------------------------------------
# Agent Tools – User Identity & Platform Version
# ---------------------------------------------------------------------------


@tool
def get_user_profile() -> str:
    """
    Retrieve the current authenticated user's Fuzzball profile.
    Use this tool when the user asks "who am I?", "what is my username?",
    "what is my email?", or any question about their own identity on the platform.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.UserServiceApi(client)
        profile = api.get_user_profile()

        lines = ["## Your Fuzzball Profile"]
        if getattr(profile, "username", None):
            lines.append(f"- **Username:** {profile.username}")
        if getattr(profile, "email", None):
            lines.append(f"- **Email:** {profile.email}")
        if getattr(profile, "name", None):
            lines.append(f"- **Display Name:** {profile.name}")
        if getattr(profile, "id", None):
            lines.append(f"- **User ID:** {profile.id}")

        return (
            "\n".join(lines)
            if len(lines) > 1
            else "Profile retrieved but contained no displayable fields."
        )

    except ApiException as e:
        return f"API error {e.status} while fetching user profile: {e.reason}"
    except Exception as e:
        return f"Failed to retrieve user profile: {str(e)}"


@tool
def get_fuzzball_version() -> str:
    """
    Retrieve the version of the currently connected Fuzzball server/agent.
    Use this tool when the user asks "what version is Fuzzball?", "what version
    am I running?", or to confirm the platform is reachable.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.VersionServiceApi(client)
        response = api.version_get()

        lines = ["## Fuzzball Server Version"]
        # Field names vary by SDK build – use getattr defensively
        for field in ("version", "git_commit", "build_date", "build_time"):
            value = getattr(response, field, None)
            if value:
                label = field.replace("_", " ").title()
                lines.append(f"- **{label}:** {value}")

        return "\n".join(lines) if len(lines) > 1 else str(response)

    except ApiException as e:
        return f"API error {e.status} while fetching server version: {e.reason}"
    except Exception as e:
        return f"Failed to retrieve Fuzzball version: {str(e)}"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("--- get_user_profile ---")
    print(get_user_profile.invoke({}))

    print("\n--- get_fuzzball_version ---")
    print(get_fuzzball_version.invoke({}))
