"""
This is the MCP (Model Context Protocol) Enabled Agent.
It is database-agnostic, matching the Strands Agents style, and dynamically explores the schema.
Save me on your computer as: agent/agent/myagent_mcp.py
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
import re
import os
import inspect

import litellm
from datarobot_genai.core.agents import InvokeReturn, make_system_prompt
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.chat import agent_chat_completion_wrapper
from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from datarobot_genai.langgraph.llm import get_llm
from datarobot_genai.langgraph.mcp import mcp_tools_context
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from openai.types.chat import CompletionCreateParams

# Import MCP libraries to connect to stdio servers and run session handshakes
from mcp import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

if TYPE_CHECKING:
    from ragas import MultiTurnSample

litellm.modify_params = True
_PLACEHOLDER_MODELS = frozenset({"unknown"})

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that plans and writes content based on the "
            "user's topic. You have access to an internal Neo4j Knowledge Graph via your tools. "
            "Chat history is provided via {chat_history} (it may be empty).",
        ),
        (
            "user",
            "The topic is {topic}. Make sure you find any interesting and "
            f"relevant information given the current year is {datetime.now().year}.",
        ),
    ]
)


async def run_mcp_query(cypher_query: str) -> str:
    """
    Launches the official neo4j-mcp server as an ephemeral stdio subprocess via uvx.
    Connects directly to the public Companies demo database (matching the Strands agent example).
    """
    params = StdioServerParameters(
        command="uvx",
        args=["neo4j-mcp"],
        env={
            "NEO4J_URI": "neo4j+s://demo.neo4jlabs.com:7687",
            "NEO4J_USERNAME": "companies",
            "NEO4J_PASSWORD": "companies",
            "NEO4J_DATABASE": "companies",
            "PATH": os.environ.get("PATH", "")  # Essential for correct lookup of uvx on macOS/Linux
        }
    )
    
    try:
        print(f"\n[MCP CLIENT]: Tunneling Cypher query directly via MCP stdio: {cypher_query}", flush=True)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Queries tools made available by the MCP server
                tools_list = await session.list_tools()
                available_tools = [t.name for t in tools_list.tools]
                target_tool = "read-cypher" if "read-cypher" in available_tools else available_tools[0]
                
                # Calls the tool exposing the structured parameters required by the standard protocol
                response = await session.call_tool(
                    target_tool,
                    arguments={"query": cypher_query}
                )
                
                text_output = "".join([
                    chunk.text for chunk in response.content 
                    if hasattr(chunk, 'text')
                ])
                return text_output
    except Exception as e:
        return f"MCP subprocess execution failed: {str(e)}"


def graph_factory_mcp(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    
    # AGNOSTIC PROMPT (STRANDS AGENTS STYLE)
    # The agent does not know the schema beforehand; it will use dynamic exploration to understand node structure.
    planner_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            make_system_prompt(
                "You are a helper for querying graph databases. Use the available tools to answer questions.\n"
                "\n"
                "To execute a Cypher query on the graph database, you must format your query inside a JSON block exactly like this:\n"
                "{{\n"
                "  \"cypher\": \"YOUR_QUERY_HERE\"\n"
                "}}\n"
                "Do not guess node labels or property keys. If you do not know the database schema or structure, "
                "first execute an exploratory Cypher query (e.g., matching a few nodes or calling system schema visualizations) "
                "to understand the schema before answering the user question."
            )
        ),
        ("placeholder", "{messages}")
    ])
    planner_chain = planner_prompt | llm

    # Planner node with automatic forwarding of the extracted query to the MCP stdio subprocess
    async def planner_node(state: MessagesState) -> dict:
        messages = state["messages"]
        response = await planner_chain.ainvoke({"messages": messages})
        text_content = getattr(response, "content", "")

        cypher_match = re.search(r'"cypher"\s*:\s*"((?:[^"\\]|\\.)*)"', text_content)
        if not cypher_match:
            backticks = chr(96) * 3
            pattern = rf"{backticks}(?:cypher|sql)?\s*(.*?)\s*{backticks}"
            cypher_match = re.search(pattern, text_content, re.DOTALL)

        if cypher_match:
            raw_query = cypher_match.group(1)
            extracted_query = raw_query.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').strip()
            
            # Execute the query routing the execution to the MCP stdio server
            db_result = await run_mcp_query(extracted_query)
            
            tool_msg = HumanMessage(
                content=(
                    f"The Cypher query was executed successfully on the Neo4j database via the MCP server.\n"
                    f"Database Results:\n{db_result}\n\n"
                    f"Analyze the results. If this was an exploratory query, use the obtained schema information to formulate your final query.\n"
                    f"Otherwise, plan and present the final outline based on the extracted data."
                )
            )
            
            # Feed the planner chain again to plan the next step
            second_response = await planner_chain.ainvoke({"messages": messages + [response, tool_msg]})
            return {"messages": [response, tool_msg, second_response]}

        return {"messages": [response]}

    # Definition of the Writer node
    writer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            make_system_prompt(
                "You are a content writer. You take the structured outline data from the planner "
                "and convert it into a publication-ready markdown blog post under 500 words."
            )
        ),
        ("placeholder", "{messages}")
    ])
    writer_chain = writer_prompt | llm

    async def writer_node(state: MessagesState) -> dict:
        response = await writer_chain.ainvoke({"messages": state["messages"]})
        return {"messages": [response]}

    def planner_to_writer_relay(state: MessagesState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            return {"messages": [HumanMessage(content=last.content)]}
        return {"messages": []}

    # Construction of the MCP-enabled linear graph
    langgraph_workflow = StateGraph(MessagesState)
    
    langgraph_workflow.add_node("planner_node", planner_node)
    langgraph_workflow.add_node("planner_to_writer_relay", planner_to_writer_relay)
    langgraph_workflow.add_node("writer_node", writer_node)
    
    langgraph_workflow.add_edge(START, "planner_node")
    langgraph_workflow.add_edge("planner_node", "planner_to_writer_relay")
    langgraph_workflow.add_edge("planner_to_writer_relay", "writer_node")
    langgraph_workflow.add_edge("writer_node", END)
    
    return langgraph_workflow


MyAgent = datarobot_agent_class_from_langgraph(graph_factory_mcp, prompt_template)


async def custompy_adaptor(
    completion_create_params: CompletionCreateParams,
) -> InvokeReturn | tuple[str, Optional["MultiTurnSample"], UsageMetrics]:
    forwarded_headers = completion_create_params.get("forwarded_headers", {})
    authorization_context = completion_create_params.get("authorization_context", {})
    mcp_config = MCPConfig(
        forwarded_headers=forwarded_headers,
        authorization_context=authorization_context,
    )
    mcp_tools_factory = lambda: mcp_tools_context(mcp_config)
    model_name = completion_create_params.get("model")
    agent = MyAgent(
        llm=get_llm(
            model_name=model_name if model_name not in _PLACEHOLDER_MODELS else None
        ),
        verbose=completion_create_params.get("verbose", True),
        timeout=completion_create_params.get("timeout", 90),
        forwarded_headers=forwarded_headers,
    )
    return await agent_chat_completion_wrapper(
        agent, completion_create_params, mcp_tools_factory
    )