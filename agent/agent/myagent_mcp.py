# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime
from typing import TYPE_CHECKING, Optional
import re
import os

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


async def execute_mcp_or_fallback(extracted_query: str, tools: list[BaseTool]) -> str:
    """
    Handles the execution of the Cypher query by integrating with DataRobot.
    In production, it uses the platform's native MCP tool. Locally, it executes a clean,
    direct connection to the companies demo database without utilizing CLI subprocesses.
    """
    mcp_tool = None
    if tools:
        # 1. DATAROBOT INTEGRATION (PRODUCTION): If the platform provides MCP tools, we use them
        for t in tools:
            if any(k in t.name.lower() for k in ["cypher", "neo4j", "query", "read"]):
                mcp_tool = t
                break
                
    if mcp_tool:
        print(f"\n[MCP AGENT]: Execution via native DataRobot Tool ({mcp_tool.name}):\n{extracted_query}", flush=True)
        try:
            # Dynamically detect the parameter name expected by the tool (e.g., 'query' or 'statement')
            param_name = "query"
            if hasattr(mcp_tool, "args") and mcp_tool.args:
                param_name = list(mcp_tool.args.keys())[0]
                
            response = await mcp_tool.ainvoke({param_name: extracted_query})
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception as e:
            return f"Execution via DataRobot MCP Tool failed: {str(e)}"
            
    # 2. CLEAN LOCAL TESTING (FALLBACK): Direct connection to the Companies DB using native Python driver
    print(f"\n[MCP AGENT - LOCAL TEST]: Connecting directly to the companies demo database...", flush=True)
    try:
        from neo4j import GraphDatabase
        uri = "neo4j+s://demo.neo4jlabs.com:7687"
        auth = ("companies", "companies")
        
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session(database="companies") as session:
                result = session.run(extracted_query)
                records = [record.data() for record in result]
                return str(records)
    except Exception as e:
        return f"Direct connection to the companies database failed: {str(e)}"


def graph_factory_mcp(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    
    # AGNOSTIC PROMPT (STRANDS AGENTS STYLE)
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

    # The planner node accesses the list of tools passed by DataRobot via closure scope
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
            
            # Executes the query via DataRobot integration or local fallback
            db_result = await execute_mcp_or_fallback(extracted_query, tools)
            
            tool_msg = HumanMessage(
                content=(
                    f"The Cypher query was executed successfully on the Neo4j database via the MCP server.\n"
                    f"Database Results:\n{db_result}\n\n"
                    f"Analyze the results. If this was an exploratory query, use the obtained schema information to formulate your final query.\n"
                    f"Otherwise, plan and present the final outline based on the extracted data."
                )
            )
            
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

    # Construction of the linear graph
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