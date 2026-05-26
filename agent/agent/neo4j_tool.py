import os
from typing import Annotated
from pydantic import Field
from langchain_neo4j import Neo4jGraph

_graph_instance = None

def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD"),
            database=os.environ.get("NEO4J_DATABASE")
        )
    return _graph_instance

# This native function structure allows the DataRobot compiler to generate the JSON schema
async def query_knowledge_graph(
    cypher_query: Annotated[str, Field(description="A valid and executable Cypher query string for Neo4j.")]
) -> str:
    """Executes a Cypher query against the Neo4j Knowledge Graph to retrieve information about entities."""
    try:
        graph = get_graph()
        result = graph.query(cypher_query)
        
        # Explicit flush to force terminal log display instantly
        print(f"\n[NEO4J DATABASE RESPONSE]: {result}\n", flush=True)
        
        if not result:
            return "No data found for this query in the graph."
        return str(result)
    except Exception as e:
        return f"Database Error: {str(e)}. Please check your Cypher syntax and try again."