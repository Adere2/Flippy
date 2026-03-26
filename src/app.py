import json
import uuid

import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agents.knowledge_agent import agent_executor


def format_tool_output(content, max_chars: int = 1200) -> str:
    if isinstance(content, (dict, list)):
        try:
            text = json.dumps(content, indent=2)
        except TypeError:
            text = str(content)
    else:
        text = str(content)

    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


st.set_page_config(page_title="Flippy Assistant", page_icon="🤖")
st.title("🤖 Flippy: Fuzzball HPC Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"

# Display chat history on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if prompt := st.chat_input("Ask me about Fuzzball or to run a command..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # LangGraph's checkpointer maintains the full conversation state, so we only
    # need to pass the newest user message to the graph input.
    latest_message = HumanMessage(content=prompt)

    with st.chat_message("assistant"):
        # We use a list to track all response parts so we can save full history later
        full_response_content = ""
        run_config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Placeholder management for interleaving text and tools
        current_text_placeholder = st.empty()
        current_text_block = ""

        # Track active tool UI elements
        # Map: tool_call_id -> {
        #   "status": st.status object,
        #   "args_placeholder": st.empty object,
        #   "args_buffer": str,
        #   "name": str
        # }
        active_tools = {}

        try:
            for msg, _ in agent_executor.stream(
                {"messages": [latest_message]},
                config=run_config,
                stream_mode="messages",
            ):
                # 1. Handle Tool Calls (Accumulate args and show status)
                if isinstance(msg, AIMessageChunk) and msg.tool_call_chunks:
                    # If we were writing text, finish that block so the tool appears below it
                    if current_text_block:
                        current_text_placeholder = None
                        current_text_block = ""

                    for chunk in msg.tool_call_chunks:
                        # Some models allow parallel tool calls, so we rely on index/id
                        # If 'id' is present, it's the start of a new call definition
                        # or a continuation where the ID is repeated.
                        # LangChain guarantees 'index' is stable.
                        tc_id = chunk.get("id")

                        # Note: In streaming chunks, 'id' might only appear in the first chunk
                        # But typically LangChain aggregators handle this. In raw stream,
                        # we might need to rely on 'index' if 'id' is missing, but
                        # usually for UI 'id' is best if available.
                        # If 'id' is missing but we have an active tool at this index, we'd continue.
                        # For simplicity, we assume 'id' is provided or we match by index if needed.
                        # Here we will rely on 'id' presence or map index to ID if we implemented
                        # a more complex aggregator.
                        # HOWEVER: Most providers send ID in first chunk.

                        if tc_id and tc_id not in active_tools:
                            # Start a new Status container
                            name = chunk.get("name") or "unknown_tool"
                            status_container = st.status(
                                f"🔧 `{name}`", expanded=True, state="running"
                            )
                            with status_container:
                                st.write("**Input:**")
                                args_ph = st.empty()

                            active_tools[tc_id] = {
                                "status": status_container,
                                "args_placeholder": args_ph,
                                "args_buffer": "",
                                "name": name,
                            }

                        # If we have args, append them
                        # We need to find which tool this chunk belongs to.
                        # If chunk has ID, use it. If not, it's a continuation of the last one?
                        # Standard LC generic stream: Usually ID is on first chunk.
                        # We'll use the ID if present.
                        if tc_id and chunk["args"]:
                            tool_data = active_tools[tc_id]
                            tool_data["args_buffer"] += chunk["args"]

                            # Live update args
                            raw = tool_data["args_buffer"]
                            try:
                                pretty = json.dumps(json.loads(raw), indent=2)
                                tool_data["args_placeholder"].code(
                                    pretty, language="json"
                                )
                            except:
                                tool_data["args_placeholder"].code(raw, language="json")

                # 2. Handle Tool Execution Results (Update status to complete)
                elif isinstance(msg, ToolMessage):
                    tc_id = msg.tool_call_id
                    if tc_id in active_tools:
                        tool_data = active_tools[tc_id]
                        status = tool_data["status"]

                        # Render output inside the status container
                        status.write("**Output:**")
                        status.code(format_tool_output(msg.content))
                        status.update(
                            label=f"`{tool_data['name']}`",
                            state="complete",
                            expanded=False,
                        )

                        # Ensure next text block starts fresh below this tool
                        current_text_placeholder = None
                        current_text_block = ""

                # 3. Stream AI Text Response
                elif isinstance(msg, AIMessageChunk) and msg.content:
                    # If we don't have a place to write text (e.g. just finished a tool), create one
                    if current_text_placeholder is None:
                        current_text_placeholder = st.empty()

                    content_chunk = msg.content

                    # Handle potential list content (Gemini/Anthropic)
                    text_to_add = ""
                    if isinstance(content_chunk, list):
                        for part in content_chunk:
                            if isinstance(part, str):
                                text_to_add += part
                            elif isinstance(part, dict) and "text" in part:
                                text_to_add += part["text"]
                    else:
                        text_to_add = str(content_chunk)

                    full_response_content += text_to_add
                    current_text_block += text_to_add
                    current_text_placeholder.markdown(current_text_block + "▌")

            # Final cleanup: remove cursor from last text block
            if current_text_placeholder:
                current_text_placeholder.markdown(current_text_block)

            # Save to history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response_content}
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
