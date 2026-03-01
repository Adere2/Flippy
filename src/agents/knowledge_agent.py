import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI

# Add src to the path so we can import our modules securely
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from tools.get_fuzzfile_syntax import get_fuzzfile_syntax
from tools.list_workflow_catalog import list_workflow_catalog
from tools.search_fuzzball_docs import search_fuzzball_docs
from tools.search_simple_fuzzfiles import search_fuzzfile_examples
from tools.search_workflow_catalog import search_workflow_catalog


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
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)

# 2. Define the Agent's Persona and Rules
system_prompt = """You are a senior Fuzzball engineer and expert support assistant.
Your job is to answer questions and generate Fuzzfiles using ONLY the provided documentation and examples.

TOOL USAGE RULES:
1. Use `search_fuzzball_docs` for general knowledge and troubleshooting.
2. Use `list_workflow_catalog` when the user asks what apps or templates are available in the catalog.
3. Use `search_workflow_catalog` FIRST if the user wants to run a specific application (like Jupyter, PyTorch, Nextflow, etc.) to see if an official template exists.
4. Use `search_fuzzfile_examples` for basic syntax examples if no workflow template exists.
5. ALWAYS use `get_fuzzfile_syntax` before generating or modifying a Fuzzfile to verify you are using the correct schema, keys, and sections.

If the tools don't return relevant information, admit that you don't know based on the docs.
Always output generated Fuzzfiles as valid YAML code blocks."""

# 3. Compile the Agent
# Updated to use create_agent and the new 'prompt' parameter
knowledge_agent = create_agent(
    model=llm,
    tools=[
        search_fuzzball_docs,
        search_fuzzfile_examples,
        search_workflow_catalog,
        list_workflow_catalog,
        get_fuzzfile_syntax,
    ],
    system_prompt=system_prompt,
)

# Quick Test Block
if __name__ == "__main__":
    test_question = """
    what in the workflow catalgo?
    """
    print(f"👤 User: {test_question}\n")

    # Instantiate our custom logging handler
    verbose_handler = VerboseCallbackHandler()

    # Pass the handler into the config so LangChain hooks into it
    result = knowledge_agent.invoke(
        {"messages": [("user", test_question)]}, config={"callbacks": [verbose_handler]}
    )

    # Clean up the raw Gemini dictionary output structure
    raw_content = result["messages"][-1].content
    if (
        isinstance(raw_content, list)
        and len(raw_content) > 0
        and "text" in raw_content[0]
    ):
        final_answer = raw_content[0]["text"]
    else:
        final_answer = str(raw_content)

    print(f"\n🤖 Fuzzball Assistant:\n{final_answer}")
