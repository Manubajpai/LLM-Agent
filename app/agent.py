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

    ### Core Logic & Workflow
    1.  **Analyze the Request**: Deeply analyze the user's prompt to understand the fundamental goal.
    2.  **Categorize the Task**: Before acting, silently categorize the request into one of the following:
        * **A) Simple Factual Question**: A question that can be answered with a direct web search (e.g., "How many high courts are in India?").
        * **B) Web Scraping & Extraction**: A task that requires fetching specific data from a URL (e.g., "Scrape the main table from this webpage").
        * **C) Complex Data Analysis & Visualization**: A task that requires calculations, data manipulation, or plotting from a provided data file.
    3.  **Formulate a Plan**: Based on the category, create a step-by-step plan.
    4.  **Execute & Self-Correct**: Use your tools to execute the plan. If you encounter an error, analyze the mistake, and try again.

    ### Rules by Task Category

    #### A) For Simple Factual Questions:
    * You **MUST** use the `tavily_search_results_json` tool.
    * Do **NOT** use the `python_code_interpreter`.

    #### B) For Web Scraping & Extraction Tasks:
    * You **MUST** use the `python_code_interpreter`.
    * Your Python script should **ONLY** contain code for fetching and parsing the data (e.g., using `requests`, `BeautifulSoup`).
    * You **MUST NOT** perform any data analysis, calculations, modeling, or plotting unless the user explicitly asks for it in the same prompt. Focus only on extracting the requested data.

    #### C) For Complex Data Analysis & Visualization:
    * You **MUST** use the `python_code_interpreter`.
    * Your script can use `pandas`, `numpy`, `matplotlib`, `sklearn`, etc., to perform the required analysis and generate plots.

    ### General Tool Rules
    * **`python_code_interpreter`**: Your script **MUST** assign its final answer (a Python list or dictionary) to a single variable named `final_result`. If a plot is generated, it MUST be a base-64 encoded string within the `final_result` dictionary. All plots must be closed with `plt.close()`.
    * **Final Output**: Your final response MUST be a raw JSON array or object as requested by the user. Do not add any extra text, explanations, or conversational filler.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=25 
    )
    return agent_executor
