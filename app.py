import os
import streamlit as st
from typing import Annotated, TypedDict
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- 1. Setup Environment ---
# Load API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# --- 2. Define Tools ---
# We'll use Tavily for web searches.
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@tool
def tavily_search(query: str):
    """
    A search tool that uses Tavily to find information on the web.
    Use this for any questions about recent events, facts, or information
    not in your knowledge base.
    """
    try:
        results = tavily.search(query=query, max_results=3)
        return "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results['results']])
    except Exception as e:
        return f"Error running search: {e}"

# --- 3. Define Agent State ---
# This is the "memory" of our agent. It's a list of messages.
# `add_messages` is a special reducer that appends new messages to the list.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- 4. Define Graph Nodes ---
# These are the functions that will be the "steps" in our graph.

def call_model(state: AgentState):
    """
    The "agent" node. This calls the LLM, which can either
    respond directly or call a tool.
    """
    # Create the OpenAI model, binding our search tool to it
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    # Use the simpler .bind_tools() and pass the decorated function
    model_with_tools = model.bind_tools([tavily_search])

    # `state['messages']` contains the full conversation history
    response = model_with_tools.invoke(state['messages'])
    
    # The response is an AIMessage. We add it to the state.
    # It will either be a final answer or a tool call.
    return {"messages": [response]}

def call_tool(state: AgentState):
    """
    The "tool" node. This checks the last message for a tool call.
    If it finds one, it runs the tool and returns the result.
    """
    last_message = state['messages'][-1]
    
    # If the last message is a tool call
    if last_message.tool_calls:
        # Note: We only handle the first tool call for simplicity
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call['name']
        
        if tool_name == "tavily_search":
            query = tool_call['args']['query']
            # Run the search
            tool_response = tavily_search.run(query)
            
            # Create a ToolMessage with the result and add it to the state
            return {"messages": [ToolMessage(content=tool_response, tool_call_id=tool_call['id'])]}
    
    # If no tool call, just return
    return {}

# --- 5. Define Conditional Edge ---
# This function decides which node to go to next.
def should_continue(state: AgentState):
    """
    The "router." This checks the last message to decide the next step.
    """
    last_message = state['messages'][-1]
    
    # If the last message is a tool call, we go to the 'action' node
    if last_message.tool_calls:
        return "action"
    # Otherwise, we end the graph
    else:
        return END

# --- 6. Assemble the Graph ---
# Create a new StateGraph with our AgentState
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", call_model) # The agent
workflow.add_node("action", call_tool)  # The tool executor

# Set the entry point
workflow.set_entry_point("agent")

# Add the conditional edge
workflow.add_conditional_edges(
    "agent",         # Start from the 'agent' node
    should_continue, # Use our router function
    {
        "action": "action", # If it returns "action", go to the 'action' node
        END: END            # If it returns END, finish.
    }
)

# Add the loop edge
# After the 'action' node runs, always go back to the 'agent' node
workflow.add_edge("action", "agent")

# Compile the graph into a runnable app
app_runnable = workflow.compile()

# --- 7. Streamlit UI ---

st.set_page_config(page_title="LangGraph Research Agent", page_icon="🤖")
st.title("LangGraph Research Agent 🤖")
st.caption("A simple Streamlit chatbot powered by LangGraph that can search the web.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm a research agent. How can I help you today?")
    ]

# Display chat messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("ai", avatar="🤖").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("human", avatar="👤").write(msg.content)
    # We don't display ToolMessages in the main chat

# Get user input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("human", avatar="👤").write(prompt)

    # Prepare the input for the graph
    graph_input = {"messages": st.session_state.messages}
    
        # Use a spinner while the agent is "thinking"
    with st.spinner("Thinking..."):
        # Create an empty message container for streaming
        response_container = st.chat_message("ai", avatar="🤖")
        response_placeholder = response_container.empty()
        final_response_content = ""
        
        # Stream events from the graph
        for event in app_runnable.stream(graph_input, config={"recursion_limit": 10}):
            
            # Check if the event is from the 'agent' node
            if "agent" in event:
                msg = event["agent"]["messages"][-1]
                if msg.content:
                    # Append the agent's message content
                    final_response_content += msg.content
                    response_placeholder.markdown(final_response_content + "▌")
                
                if msg.tool_calls:
                    # Display a message if the agent is calling a tool
                    tool_call = msg.tool_calls[0]
                    response_placeholder.markdown(
                        f"Calling tool: `{tool_call['name']}` "
                        f"with query: `{tool_call['args']['query']}`..."
                    )

            elif "action" in event:
                # Optionally show progress here
                pass
        
        # Update the placeholder with the final, complete response
        response_placeholder.markdown(final_response_content)

    
    # Add the final AI response to the session state
    st.session_state.messages.append(AIMessage(content=final_response_content))