import json
import logging
from typing import Optional

from minirag import azure_chat, azure_chat_with_tools
from vector_store import retrieve, format_context, load_index

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SEARCH_NIST_TOOL = {
    "type": "function",
    "function": {
        "name": "search_nist_knowledge",
        "description": "Search the NIST cybersecurity publications knowledge base to find relevant information about security frameworks, access control, cryptography, zero trust, PKI, and other cybersecurity topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant NIST publications and guidelines and extract useful information.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

TOOLS = [SEARCH_NIST_TOOL]


# Tool dispatching and execution
# Load index once at module level
_index = None
_metadata = None


def get_index():
    global _index, _metadata
    if _index is None:
        _index, _metadata = load_index()
    return _index, _metadata


def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "search_nist_knowledge":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        
        logger.info(f"[TOOL] Searching NIST knowledge: '{query}' (top_k={top_k})")
        
        index, metadata = get_index()
        results = retrieve(query, top_k=top_k, index=index, metadata=metadata)
        
        return format_context(results)
    else:
        return f"Unknown tool: {tool_name}"


# Critic Agent

CRITIC_SYSTEM_PROMPT = """You are a critic agent that evaluates whether retrieved context is sufficient to answer a user's question.

Analyze the query and retrieved context, then respond with a JSON object:
{
    "decision": "sufficient" | "insufficient" | "needs_refinement",
    "reasoning": "Brief explanation of your decision",
    "refined_query": "If needs_refinement, provide an improved search query, otherwise null"
}

Guidelines:
- "sufficient": Context directly addresses the query with relevant, specific information
- "insufficient": Context is completely unrelated or missing critical information
- "needs_refinement": Context is partially relevant but a better search query could improve results"""


def critic_evaluate(query: str, context: str) -> dict:
    prompt = f"""User Query: {query}

Retrieved Context:
{context}

Evaluate if this context sufficiently answers the query. Respond with JSON only."""

    response = azure_chat(prompt, system_prompt=CRITIC_SYSTEM_PROMPT)
    
    try:
        # Try to extract JSON from markdown code blocks if present
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  # Remove ```
        
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        
        cleaned_response = cleaned_response.strip()
        
        # Parse JSON response
        result = json.loads(cleaned_response)
        logger.info(f"[CRITIC] Decision: {result.get('decision')} - {result.get('reasoning', '')[:100]}")
        return result
    except json.JSONDecodeError as e:
        # Fallback if response isn't valid JSON
        logger.warning(f"[CRITIC] Failed to parse response: {e}")
        logger.warning(f"[CRITIC] Raw response (first 500 chars): {response[:500]}")
        return {"decision": "sufficient", "reasoning": "Parse error fallback", "refined_query": None}


# Main Agent Orchestrator
AGENT_SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in NIST cybersecurity publications and guidelines.

When answering questions:
1. Use the search_nist_knowledge tool to find relevant information from NIST publications
2. Synthesize information from multiple sources when appropriate
3. Always cite your sources using the format [Source Name, Page X]
4. If information is not found, clearly state that

Be accurate, thorough, and cite specific documents when possible."""


# three iterations should be sufficient for most queries and when the critic think it is sufficient, it breaks the loop and give the response (augumented response)
def agentic_rag(user_query: str, max_iterations: int = 3) -> str:
    logger.info(f"[AGENT] Starting agentic RAG for: '{user_query}'")
    
    messages = [{"role": "user", "content": user_query}]
    context = None
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"[AGENT] Iteration {iteration}/{max_iterations}")
        
        # Step 1: Send to GPT-4o with retriever tool
        response = azure_chat_with_tools(messages, tools=TOOLS, system_prompt=AGENT_SYSTEM_PROMPT)
        
        # Step 2: Check if tool was called
        tool_calls = response.get("tool_calls", [])
        
        if tool_calls:
            # Execute tool calls
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                # Execute tool
                tool_result = execute_tool(tool_name, arguments)
                context = tool_result
                
                # Add assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                
                # Add tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
            
            # Step 3: Critic evaluation
            if context:
                critic_result = critic_evaluate(user_query, context)
                decision = critic_result.get("decision", "sufficient")
                
                if decision == "sufficient":
                    logger.info("[AGENT] Critic approved context, generating answer")
                    break
                elif decision == "needs_refinement" and critic_result.get("refined_query"):
                    # Retry with refined query
                    refined_query = critic_result["refined_query"]
                    logger.info(f"[AGENT] Refining search with: '{refined_query}'")
                    
                    # Add refinement request
                    messages.append({
                        "role": "user",
                        "content": f"The previous search wasn't quite right. Please search again with this refined query: {refined_query}"
                    })
                else:
                    # Insufficient - try broader search
                    logger.info("[AGENT] Context insufficient, trying broader search")
                    messages.append({
                        "role": "user",
                        "content": "The retrieved information wasn't sufficient. Please try a broader or different search."
                    })
        else:
            # No tool call - model responded directly
            logger.info("[AGENT] Model responded without tool call")
            return response.get("content", "I couldn't find relevant information.")
    
    # Step 4: Generate final answer with context
    final_prompt = f"""Based on the retrieved information, provide a comprehensive answer to the user's question.
    
User Question: {user_query}

Retrieved Context:
{context if context else "No context retrieved."}

Provide a detailed answer with proper citations."""

    final_response = azure_chat_with_tools(
        messages + [{"role": "user", "content": final_prompt}],
        system_prompt=AGENT_SYSTEM_PROMPT
    )
    
    return final_response.get("content", "Unable to generate response.")


# CLI Interface


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is zero trust architecture according to NIST?"
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    answer = agentic_rag(query)
    
    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}")
    print(answer)
