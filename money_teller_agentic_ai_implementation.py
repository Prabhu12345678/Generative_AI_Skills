import pandas as pd
import json
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Define Bank Operations as Single-Input or Clean String-Parsed Tools
# Local LLMs handle a single string input much better than multiple native args
@tool("check_account_balance")
def check_account_balance(account_number: str) -> float:
    """Returns the current balance for a given bank account number. Input must be just the account number string."""
    mock_balances = {"ACC-1001": 4520.50, "ACC-1002": 125.00}
    return mock_balances.get(account_number.strip(), 0.0)

@tool("transfer_funds")
def transfer_funds(transfer_details: str) -> str:
    """Transfers funds between accounts. Input format must be a comma-separated string exactly like: source_account,target_account,amount"""
    try:
        # Safely parse the comma-separated text from the local model
        source, target, amt = [x.strip() for x in transfer_details.split(",")]
        return f"Success: Transferred ${float(amt)} from {source} to {target}."
    except Exception as e:
        return "Error parsing inputs. Please provide input exactly as: source_account,target_account,amount"

@tool("query_bank_procedures")
def query_bank_procedures(query: str) -> str:
    """Queries the local FAISS database for official banking policies and procedures."""
    docs = retriever.invoke(query)
    return docs[0].page_content if docs else "No specific policy found."

# Group tools
tools = [check_account_balance, transfer_funds, query_bank_procedures]

# 2. Initialize Local LLM and Embedding Model
# CRITICAL: Add a system prompt wrapper directly to the Ollama class to force strict output rules on Llama 3.2
llm = Ollama(
    model="llama3.2", 
    temperature=0.0,
    system="""You are a strict banking assistant. 
    You must always answer using the exact ReAct format:
    Thought: [your reasoning]
    Action: [tool name]
    Action Input: [tool parameter]
    Observation: [tool output]
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: [your final human-readable response to the user]
    
    If you already know the answer or have completed the action, skip straight to:
    Thought: I now know the final answer
    Final Answer: [your response]
    Do not repeat yourself or call tools unnecessarily."""
)

embeddings = OllamaEmbeddings(model="all-minilm")

# 3. Setup Local Knowledge Base using FAISS
documents = [
    "Procedure for international wire transfers: Identify customer, check daily limit, get manager approval for amounts over $10000.",
    "Customer authentication procedure: Ask for two forms of ID, verify date of birth and current address."
]
faiss_db = FAISS.from_texts(documents, embeddings)
retriever = faiss_db.as_retriever(search_kwargs={"k": 1})

# 4. Initialize Agent using Python 3.8 Compatible Syntax
# Switching to ZERO_SHOT_REACT_DESCRIPTION because it uses highly scannable string inputs instead of buggy JSON formatting
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4 # Safeguard loop budget
)

# 5. Execute Queries
print("--- Test 1: Policy Retrieval ---")
query_1 = "What is the procedure for verifying a customer at the teller counter?"
response_1 = agent_executor.run(query_1)
print(f"Final Answer: {response_1}\n")

print("--- Test 2: Multi-Input Tool Execution ---")
# Prompt explicitly defines the formatted style your tool expects
query_2 = "Transfer $150 from ACC-1001 to ACC-1002. Call the tool using format: ACC-1001,ACC-1002,150"
response_2 = agent_executor.run(query_2)
print(f"Final Answer: {response_2}")
