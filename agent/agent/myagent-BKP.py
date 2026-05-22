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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from openai.types.chat import CompletionCreateParams

# Direct import of our custom Neo4j execution tool
from agent.neo4j_tool import query_knowledge_graph

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


def graph_factory(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    
    # Prompt instructing the planner to output Cypher within a JSON wrapper block
    # Note the escaped double-curly braces {{ and }} around the cypher block example 
    # to prevent LangChain from raising a KeyError for missing template variables.
    planner_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            make_system_prompt(
                "You are a content planner with direct access to an enterprise Neo4j Knowledge Graph.\n"
                "\n"
                "CRITICAL INSTRUCTION: If the user asks about entities, actors, movies, or directors, "
                "you MUST instantly provide a valid Cypher query inside a JSON wrapper formatting block like this:\n"
                "{{\n"
                "  \"cypher\": \"YOUR_QUERY_HERE\"\n"
                "}}\n"
                "Always check the database results first to collect accurate data."
            )
        ),
        ("placeholder", "{messages}")
    ])
    planner_chain = planner_prompt | llm

    # Explicit node wrapper with programmatic execution fallback
    async def planner_node(state: MessagesState) -> dict:
        response = await planner_chain.ainvoke({"messages": state["messages"]})
        text_content = getattr(response, "content", "")

        # Regex parsing check to extract the query from the assistant response
        cypher_match = re.search(r'"cypher"\s*:\s*"([^"]+)"', text_content)
        if not cypher_match:
            # We construct backticks dynamically to prevent any markdown rendering issues in the UI
            backticks = chr(96) * 3
            pattern = rf"{backticks}(?:cypher|sql)?\s*(.*?)\s*{backticks}"
            cypher_match = re.search(pattern, text_content, re.DOTALL)

        if cypher_match:
            # Cleanup backslashes and escape sequences
            extracted_query = cypher_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
            print(f"\n[INTERCEPTOR LOG]: Executing Cypher query on Neo4j Cluster:\n{extracted_query}\n", flush=True)
            
            # Programmatic tool caller supporting raw functions, async functions, and LangChain Tool wrapper objects
            try:
                if hasattr(query_knowledge_graph, "invoke"):
                    try:
                        db_result = query_knowledge_graph.invoke({"cypher_query": extracted_query})
                    except Exception:
                        db_result = query_knowledge_graph.invoke(extracted_query)
                elif inspect.iscoroutinefunction(query_knowledge_graph):
                    db_result = await query_knowledge_graph(extracted_query)
                else:
                    db_result = query_knowledge_graph(extracted_query)
            except Exception as e:
                db_result = f"Local execution failed: {str(e)}"
            
            tool_msg = ToolMessage(
                content=f"Database Results:\n{db_result}",
                tool_call_id="manual_intercept_id"
            )
            
            # Loop the data payload back into the planner node to synthesize the outline
            second_response = await planner_chain.ainvoke({"messages": state["messages"] + [response, tool_msg]})
            return {"messages": [response, tool_msg, second_response]}

        return {"messages": [response]}

    # Writer agent definition
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

    # Setting up the state orchestration workflow
    langgraph_workflow = StateGraph(MessagesState)
    
    langgraph_workflow.add_node("planner_node", planner_node)
    langgraph_workflow.add_node("planner_to_writer_relay", planner_to_writer_relay)
    langgraph_workflow.add_node("writer_node", writer_node)
    
    langgraph_workflow.add_edge(START, "planner_node")
    langgraph_workflow.add_edge("planner_node", "planner_to_writer_relay")
    langgraph_workflow.add_edge("planner_to_writer_relay", "writer_node")
    langgraph_workflow.add_edge("writer_node", END)
    
    return langgraph_workflow


MyAgent = datarobot_agent_class_from_langgraph(graph_factory, prompt_template)


async def custompy_adaptor(
    completion_create_params: CompletionCreateParams,
) -> InvokeReturn | tuple[str, Optional["MultiTurnSample"], UsageMetrics]:
    forwarded_headers = completion_create_params.get("forwarded_headers", {})
    authorization_context = completion_create_params.get("authorization_context", {})
    mcp_config = MCPConfig(
        forwarded_headers=forwarded_headers,
        authorization_context=authorization_context,
    )
    mcp_tools_factory = lambda: mcp_tools_context(mcp_config)  # noqa: E731
    model_name = completion_create_params.get("model")
    agent = MyAgent(
        llm=get_llm(
            model_name=model_name if model_name not in _PLACEHOLDER_MODELS else None
        ),
        verbose=completion_create_params.get("verbose", True),  # type: ignore[arg-type]
        timeout=completion_create_params.get("timeout", 90),  # type: ignore[arg-type]
        forwarded_headers=forwarded_headers,  # type: ignore[arg-type]
    )
    return await agent_chat_completion_wrapper(
        agent, completion_create_params, mcp_tools_factory
    )