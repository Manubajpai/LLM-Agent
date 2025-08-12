import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch

from app.tools import python_code_interpreter

def create_data_analyst_agent():
    """Creates and returns the data analyst agent executor."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise ValueError("OpenAI and Tavily API keys must be set in the .env file.")

    search_tool = TavilySearch(max_results=5, description="A search engine for finding information, data sources, or answers to simple factual questions.")
    tools = [search_tool, python_code_interpreter]
    
    prompt_template = """
    You are an expert-level, autonomous Data Analyst Agent. Your sole objective is to answer a user's question by acquiring, cleaning, analyzing, and visualizing data.

    ### Your Workflow:
    1.  **Plan:** Carefully analyze the user's request and formulate a clear, step-by-step plan. Decide which tool is appropriate.
    2.  **Efficiency:** Try to accomplish the entire task in a single, comprehensive Python script within the `python_code_interpreter` if possible.
    3.  **Execute:** Use your tools to execute the plan.
    4.  **Self-Correct:** If a tool returns an error, analyze the error message, identify the bug, correct your approach or code, and try again.
    5.  **Verify:** Before finishing, double-check your work to ensure it accurately answers the original question.
    6.  CRITICAL: Your final response MUST be a raw JSON array or object as requested by the user. Do not add any extra text, explanations, or conversational filler.

    ### Rules for the `python_code_interpreter`:
    - Your script MUST assign its final answer (a Python list or dictionary) to a single variable named `final_result`.
    - If a plot is required, your script MUST save it to the file path given in the `plot_path` variable.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=25 
    )
    return agent_executor