import streamlit as st
# Import your agent executor from your existing file
# Note: Ensure your knowledge_agent.py exports an 'agent_executor' or similar compiled agent object!
from agents.knowledge_agent import agent_executor 
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Flippy Assistant", page_icon="🤖")
st.title("🤖 Flippy: Fuzzball HPC Assistant")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask me about Fuzzball or to run a command..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Flippy is thinking (and using tools)..."):
            try:
                # Convert session history to LangChain message objects
                lc_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_messages.append(AIMessage(content=msg["content"]))

                response = agent_executor.invoke({"messages": lc_messages})
                raw_content = response["messages"][-1].content
                if isinstance(raw_content, list) and raw_content and "text" in raw_content[0]:
                    output_text = raw_content[0]["text"]
                else:
                    output_text = str(raw_content)
                
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"Error running agent: {e}")