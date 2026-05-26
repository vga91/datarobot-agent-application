"""
This is the central routing file (Router) that receives calls from DataRobot
and decides which agent to launch based on the user's prompt.
Save me on your computer as: agent/agent/myagent.py
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from datarobot_genai.core.agents import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from openai.types.chat import CompletionCreateParams

# Dynamic import of adapters from the separate agent modules
from agent.myagent_standard import MyAgent as MyStandardAgent, custompy_adaptor as custompy_adaptor_standard
from agent.myagent_mcp import MyAgent as MyMCPAgent, custompy_adaptor as custompy_adaptor_mcp

if TYPE_CHECKING:
    from ragas import MultiTurnSample

# Exposes a static class reference required for compilation in DataRobot
MyAgent = MyStandardAgent


async def custompy_adaptor(
    completion_create_params: CompletionCreateParams,
) -> InvokeReturn | tuple[str, Optional["MultiTurnSample"], UsageMetrics]:
    """
    Smart router to dispatch incoming requests.
    Analyzes the user's prompt content to route to the correct agent.
    """
    user_prompt = ""
    messages = completion_create_params.get("messages", [])
    
    # Identify the user's last message to decide the route
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break
            
    # Keywords indicating the need to activate the MCP environment (Companies/FinCEN Database)
    use_mcp_keywords = ["mcp", "company", "companies", "fincen", "crime", "transaction"]
    
    if any(keyword in user_prompt.lower() for keyword in use_mcp_keywords):
        print("\n[ROUTE ENGINE]: MCP keywords detected. Routing to -> MCP AGENT (Companies)\n", flush=True)
        return await custompy_adaptor_mcp(completion_create_params)
    
    # Default behavior: routing to standard agent for movies
    print("\n[ROUTE ENGINE]: Standard request. Routing to -> STANDARD AGENT (Movies)\n", flush=True)
    return await custompy_adaptor_standard(completion_create_params)
