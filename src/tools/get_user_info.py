import base64
import json

from langchain_core.tools import tool

from src.tools.fuzzball_auth import (
    ApiException,
    fuzzball,
    get_api_instance,
    load_fuzzball_config,
)


def _decode_token_account_id(client) -> str | None:
    """Helper to decode the 'account_id' claim from the SDK's JWT token."""
    try:
        token = client.configuration.access_token
        if not token:
            return None

        # JWT format: header.payload.signature
        parts = token.split(".")
        if len(parts) < 2:
            return None

        payload = parts[1]
        # Add padding for base64 decoding
        payload += "=" * ((4 - len(payload) % 4) % 4)
        decoded_bytes = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded_bytes)
        return claims.get("account_id")
    except Exception:
        return None


@tool
def get_user_info() -> str:
    """
    Comprehensive tool to retrieve all identity and environment details for the current user.
    Use this when the user asks "who am I?", "what account am I in?", "show my context",
    or for a summary of their Fuzzball platform identity.

    Returns:
        A formatted Markdown string containing User Profile, Organization,
        Active Contexts, Accounts/Groups, and Fuzzball Version.
    """
    try:
        client = get_api_instance(skip_auth=True)
        config = load_fuzzball_config()
        active_context_name = config.get("activeContext", "None")

        # Get the active account ID directly from the token used by the SDK
        active_account_id = _decode_token_account_id(client)

        lines = ["# Fuzzball User Information\n"]

        # 1. User Profile & User ID
        try:
            user_api = fuzzball.UserServiceApi(client)
            profile = user_api.get_user_profile()
            lines.append("## Identity")
            lines.append(f"- **Username:** {getattr(profile, 'username', 'N/A')}")
            lines.append(f"- **Display Name:** {getattr(profile, 'name', 'N/A')}")
            lines.append(f"- **Email:** {getattr(profile, 'email', 'N/A')}")
            lines.append(f"- **User ID:** `{getattr(profile, 'id', 'N/A')}`\n")
        except Exception as e:
            lines.append(f"## Identity\n- _Error fetching profile: {e}_\n")

        # 2. Organization Name
        try:
            org_api = fuzzball.OrganizationServiceApi(client)
            org = org_api.get_organization()
            lines.append("## Organization")
            lines.append(f"- **Name:** {getattr(org, 'name', 'N/A')}")
            lines.append(f"- **Org ID:** `{getattr(org, 'id', 'N/A')}`\n")
        except Exception as e:
            lines.append(f"## Organization\n- _Error fetching organization: {e}_\n")

        # 3. Contexts (from local config)
        lines.append("## Contexts")
        contexts = config.get("contexts", [])
        if not contexts:
            lines.append("- No contexts found in configuration.\n")

        for ctx in contexts:
            name = ctx.get("name", "Unknown")
            address = ctx.get("address", "Unknown")

            if name == active_context_name:
                is_active = " ✅ *(Active)*"
            else:
                is_active = ""

            lines.append(f"- **{name}** ({address}){is_active}")
        lines.append("")

        # 4. Accounts / Groups
        try:
            acct_api = fuzzball.AccountServiceApi(client)
            acct_resp = acct_api.list_accounts()
            accounts = getattr(acct_resp, "accounts", []) or []

            lines.append("## Accounts / Groups")
            if not accounts:
                lines.append("- No accounts found for this user.\n")
            else:
                for acct in accounts:
                    acct_name = getattr(acct, "name", "Unnamed")
                    acct_id = getattr(acct, "id", "N/A")
                    is_active = " ✅ *(Active)*" if acct_id == active_account_id else ""
                    lines.append(f"- **{acct_name}** (ID: `{acct_id}`){is_active}")
                lines.append("")
        except Exception as e:
            lines.append(f"## Accounts / Groups\n- _Error fetching accounts: {e}_\n")

        # 5. Fuzzball Version
        try:
            version_api = fuzzball.VersionServiceApi(client)
            version_resp = version_api.version_get()
            version = getattr(version_resp, "version", "Unknown")
            lines.append("## Platform")
            lines.append(f"- **Fuzzball Version:** {version}")
        except Exception as e:
            lines.append(f"## Platform\n- _Error fetching version: {e}_\n")

        return "\n".join(lines)

    except ApiException as e:
        return f"API error {e.status} while retrieving user info: {e.reason}"
    except Exception as e:
        return f"Failed to retrieve comprehensive user info: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print(get_user_info.invoke({}))
