import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to PYTHONPATH so we can import from src
# project_root = Path(__file__).resolve().parent.parent.parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler

from src.config import get_llm

from src.tools.fuzzball_account_tools import (
    get_account,
    list_account_members,
    list_accounts,
)
from src.tools.fuzzball_org_tools import get_organization, list_organization_members
from src.tools.fuzzball_user_tools import get_fuzzball_version, get_user_profile
from src.tools.fuzzball_workflow_tools import (
    get_workflow,
    get_workflow_logs,
    get_workflow_status,
    list_workflows,
    restart_workflow,
    start_workflow,
    stop_workflow,
    validate_workflow,
)
from src.tools.get_fuzzfile_syntax import get_fuzzfile_syntax
from src.tools.get_user_info import get_user_info
from src.tools.list_workflow_catalog import list_workflow_catalog
from src.tools.search_fuzzball_docs import search_fuzzball_docs
from src.tools.search_simple_fuzzfiles import search_fuzzfile_examples
from src.tools.search_workflow_catalog import search_workflow_catalog


# --- NEW: Custom Callback Handler for Verbose Logging ---
class VerboseCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"\n⚙️  STAGE: Tool Selection")
        print(f"   🔧 Tool Called: {serialized.get('name', 'Unknown Tool')}")

        # Try to format the JSON input nicely for the console
        try:
            clean_input = json.dumps(json.loads(input_str), indent=2)
        except:
            clean_input = input_str

        print(f"   📥 Arguments Passed:\n{clean_input}")
        print("\n⏳ Executing tool...")

    def on_tool_end(self, output, **kwargs):
        print(f"\n⚙️  STAGE: Tool Execution Complete")

        # Truncate the output string if it's massively long to keep the console readable
        clean_output = str(output)
        if len(clean_output) > 1500:
            clean_output = (
                clean_output[:1500] + "\n   ... [OUTPUT TRUNCATED FOR READABILITY] ..."
            )

        print(f"   📤 Tool Returned:\n{clean_output}")
        print("\n⏳ Agent is analyzing the results...")


# 1. Initialize the LLM
llm = get_llm()

# 2. Define the Agent's Persona and Rules
system_prompt = """You are a senior Fuzzball engineer and expert support assistant.
Your job is to answer questions and generate Fuzzfiles using ONLY the provided documentation and examples.

TOOL USAGE RULES:
1. Use `search_fuzzball_docs` for general knowledge and troubleshooting.
2. Use `list_workflow_catalog` when the user asks what apps or templates are available in the catalog.
3. Use `search_workflow_catalog` FIRST if the user wants to run a specific application (like Jupyter, PyTorch, Nextflow, etc.) to see if an official template exists.
4. Use `search_fuzzfile_examples` for basic syntax examples if no workflow template exists.
5. ALWAYS use `get_fuzzfile_syntax` before generating or modifying a Fuzzfile to verify you are using the correct schema, keys, and sections.

IDENTITY & ACCOUNT TOOL RULES:
6. Use `get_user_info` as the PREFERRED tool when the user asks "who am I?", "what is my account?", "what context am I in?", or any general question about their identity, organization, or platform version.
7. Use `get_user_profile` specifically for identity details if `get_user_info` is not sufficient.
8. Use `get_fuzzball_version` specifically for version or connectivity checks.
9. Use `list_accounts` when the user asks what accounts or groups they belong to. Use `get_account` to drill into a specific account by ID.
10. Use `list_account_members` when the user asks who is in a specific account or group.
11. Use `get_organization` for specific org details. Use `list_organization_members` to see everyone in the org.

WORKFLOW CONTROL TOOL RULES:
11. Use `list_workflows` when the user asks to see their workflows, optionally filtered by status (STARTED, FINISHED, FAILED, CANCELED) or scope (user/group).
12. Use `get_workflow` when the user wants full details on a specific workflow — including its Fuzzfile definition and job list. Prefer this over `get_workflow_status` when the user asks to "see" or "show" a workflow.
13. Use `get_workflow_status` when the user specifically asks about the live status, progress, or per-job breakdown of a running or completed workflow.
14. Use `validate_workflow` BEFORE `start_workflow` whenever you have generated or modified a Fuzzfile, to confirm it is schema-valid.
15. Use `start_workflow` to submit a Fuzzfile YAML and launch a new workflow run. Always confirm with the user before submitting.
16. Use `stop_workflow` to cancel a running workflow. Always confirm the workflow ID with the user before stopping.
17. Use `restart_workflow` when the user wants to rerun a previous workflow. It reuses the original Fuzzfile automatically.
18. Use `get_workflow_logs` when the user asks for logs, output, or error details from a workflow's jobs.

If the tools don't return relevant information, admit that you don't know based on the docs.
Always output generated Fuzzfiles as valid YAML code blocks."""

# 3. Compile the Agent
# Updated to use create_agent and the new 'prompt' parameter
knowledge_agent = create_agent(
    model=llm,
    tools=[
        # Knowledge & Fuzzfile tools
        search_fuzzball_docs,
        search_fuzzfile_examples,
        search_workflow_catalog,
        list_workflow_catalog,
        get_fuzzfile_syntax,
        # Workflow control & diagnostics tools
        list_workflows,
        get_workflow,
        get_workflow_status,
        validate_workflow,
        start_workflow,
        stop_workflow,
        restart_workflow,
        get_workflow_logs,
        # Identity, account & org tools
        get_user_info,
        get_user_profile,
        get_fuzzball_version,
        list_accounts,
        get_account,
        list_account_members,
        get_organization,
        list_organization_members,
    ],
    system_prompt=system_prompt,
)

# Quick Test Block
if __name__ == "__main__":
    print("🤖 Fuzzball Assistant: Ready! (Type 'exit' to quit)")
    
    # We maintain a simple message history for this session manually
    chat_history = []
    
    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Exiting.")
                break
            if not user_input:
                continue

            # Instantiate our custom logging handler
            verbose_handler = VerboseCallbackHandler()
            
            # Simple context management: Append user message
            # langchain expects messages in [("role", "content")] format for .invoke
            # But we need to maintain the full history for context
            chat_history.append(("user", user_input))

            result = knowledge_agent.invoke(
                {"messages": chat_history}, 
                config={"callbacks": [verbose_handler]}
            )
            
            # Extract AI response
            ai_message = result["messages"][-1]
            chat_history.append(ai_message) # Add AI response to history

            # Clean up the raw Gemini dictionary output structure for display
            raw_content = ai_message.content
            if (
                isinstance(raw_content, list)
                and len(raw_content) > 0
                and "text" in raw_content[0]
            ):
                final_answer = raw_content[0]["text"]
            else:
                final_answer = str(raw_content)

            print(f"\n🤖 Fuzzball Assistant:\n{final_answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
