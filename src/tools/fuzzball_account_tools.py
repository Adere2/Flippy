from langchain_core.tools import tool

from src.tools.fuzzball_auth import ApiException, fuzzball, get_api_instance

# ---------------------------------------------------------------------------
# Agent Tools – Accounts (Groups)
# ---------------------------------------------------------------------------


@tool
def list_accounts() -> str:
    """
    List all Fuzzball accounts (groups) that the current user has access to.
    Use this tool when the user asks "what accounts do I have?", "what groups
    am I in?", "list my accounts", or needs to find an account ID before
    performing account-specific operations.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.AccountServiceApi(client)
        response = api.list_accounts()

        accounts = getattr(response, "accounts", []) or []

        if not accounts:
            return "No accounts found for the current user."

        lines = [f"## Fuzzball Accounts ({len(accounts)} found)\n"]
        for acct in accounts:
            acct_id = getattr(acct, "id", "N/A")
            name = getattr(acct, "name", "Unnamed")
            description = getattr(acct, "description", None)
            lines.append(f"### {name}")
            lines.append(f"- **ID:** `{acct_id}`")
            if description:
                lines.append(f"- **Description:** {description}")

        return "\n".join(lines)

    except ApiException as e:
        return f"API error {e.status} while listing accounts: {e.reason}"
    except Exception as e:
        return f"Failed to list accounts: {str(e)}"


@tool
def get_account(account_id: str) -> str:
    """
    Retrieve detailed information about a specific Fuzzball account (group) by its ID.
    Use this tool when the user asks for details about a particular account,
    such as its name, description, or configuration. Requires an account ID,
    which can be found using the list_accounts tool first.

    Args:
        account_id: The unique identifier of the account to retrieve.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.AccountServiceApi(client)
        acct = api.get_account(account_id)

        name = getattr(acct, "name", "Unknown")
        lines = [f"## Account: {name}"]
        for field, label in [
            ("id", "ID"),
            ("description", "Description"),
            ("created_at", "Created"),
            ("updated_at", "Last Updated"),
        ]:
            value = getattr(acct, field, None)
            if value:
                formatted = f"`{value}`" if field == "id" else str(value)
                lines.append(f"- **{label}:** {formatted}")

        return "\n".join(lines)

    except ApiException as e:
        return f"API error {e.status} while fetching account '{account_id}': {e.reason}"
    except Exception as e:
        return f"Failed to retrieve account '{account_id}': {str(e)}"


@tool
def list_account_members(account_id: str) -> str:
    """
    List all members of a specific Fuzzball account (group) by its ID.
    Use this tool when the user asks "who is in account X?", "list the members
    of this group", or needs to audit account membership. Use list_accounts
    first if you do not know the account ID.

    Args:
        account_id: The unique identifier of the account whose members to list.
    """
    try:
        client = get_api_instance(skip_auth=True)
        api = fuzzball.AccountServiceApi(client)
        response = api.list_account_members(account_id)

        members = getattr(response, "users", []) or []

        if not members:
            return f"No members found for account `{account_id}`."

        lines = [f"## Members of Account `{account_id}` ({len(members)} found)\n"]
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
        return f"API error {e.status} while listing members of account '{account_id}': {e.reason}"
    except Exception as e:
        return f"Failed to list account members for '{account_id}': {str(e)}"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("--- list_accounts ---")
    print(list_accounts.invoke({}))

    print("\n--- get_account (replace with a real ID) ---")
    print(get_account.invoke({"account_id": "your-account-id-here"}))
