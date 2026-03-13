from langchain_core.tools import tool

from src.tools.fuzzball_auth import ApiException, fuzzball, get_api_instance

# ---------------------------------------------------------------------------
# Agent Tools – Organization
# ---------------------------------------------------------------------------


@tool
def get_organization() -> str:
    """
    Retrieve details about the current user's Fuzzball organization.
    Use this tool when the user asks "what organization am I in?", "tell me about
    my org", "what is my organization name?", or any question about the top-level
    organization on the Fuzzball platform.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.OrganizationServiceApi(client)
        org = api.get_organization()

        lines = ["## Your Fuzzball Organization"]
        for field, label in [
            ("name", "Name"),
            ("id", "ID"),
            ("description", "Description"),
            ("created_at", "Created"),
        ]:
            value = getattr(org, field, None)
            if value:
                formatted = f"`{value}`" if field == "id" else str(value)
                lines.append(f"- **{label}:** {formatted}")

        return (
            "\n".join(lines)
            if len(lines) > 1
            else "Organization retrieved but contained no displayable fields."
        )

    except ApiException as e:
        return f"API error {e.status} while fetching organization: {e.reason}"
    except Exception as e:
        return f"Failed to retrieve organization: {str(e)}"


@tool
def list_organization_members() -> str:
    """
    List all members of the current user's Fuzzball organization.
    Use this tool when the user asks "who is in my org?", "list all organization
    members", "who are my colleagues on this platform?", or wants to audit
    who has access to the Fuzzball installation.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.OrganizationServiceApi(client)
        response = api.list_organization_members()

        members = getattr(response, "users", []) or []

        if not members:
            return "No members found in the organization."

        lines = [f"## Organization Members ({len(members)} found)\n"]
        for user in members:
            username = getattr(user, "username", None)
            email = getattr(user, "email", None)
            user_id = getattr(user, "id", None)
            display = username or email or user_id or "Unknown User"
            entry = f"- **{display}**"
            if user_id and display != user_id:
                entry += f" (`{user_id}`)"
            lines.append(entry)
            if email and display != email:
                lines.append(f"  - Email: {email}")

        return "\n".join(lines)

    except ApiException as e:
        return f"API error {e.status} while listing organization members: {e.reason}"
    except Exception as e:
        return f"Failed to list organization members: {str(e)}"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("--- get_organization ---")
    print(get_organization.invoke({}))

    print("\n--- list_organization_members ---")
    print(list_organization_members.invoke({}))
