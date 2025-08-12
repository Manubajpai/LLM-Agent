# AI Data Analyst Agent

This project is a sophisticated, general-purpose AI Data Analyst Agent. It exposes a public API that can accept complex, natural language questions and perform a complete data analysis workflow, including data acquisition, cleaning, analysis, and visualization, to provide a structured answer.

## 🚀 Key Features

* **General-Purpose Reasoning:** Built to handle a wide variety of unforeseen data analysis questions, not just pre-defined tasks.
* **Multi-Tool Capability:** Intelligently chooses between two powerful tools:
    * **Web Search:** Utilizes the Tavily API for real-time, fact-based queries and to find data sources.
    * **Code Interpreter:** A secure environment to write and execute complex Python scripts for in-depth analysis.
* **Advanced Data Analysis:** Leverages a suite of Python libraries (`pandas`, `numpy`, `matplotlib`, `statsmodels`) to perform tasks like:
    * Live web scraping of HTML tables.
    * Data cleaning and manipulation.
    * Statistical calculations (e.g., correlations, regressions).
    * Querying large datasets (e.g., with `duckdb`).
    * Generating data visualizations like scatter plots and pie charts.
* **Self-Correction:** The agent is prompted to analyze its own errors and rewrite its code, making it resilient to bugs and unexpected data formats.
* **Robust Architecture:** Built with a professional tech stack, including a FastAPI backend and LangChain for agent orchestration, and is designed to be deployed as a public web service.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **AI Framework:** LangChain
* **LLM:** OpenAI GPT-4o
* **Tools:** Tavily Search API, Python Code Interpreter
* **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `requests`, `duckdb`, `lxml`, `statsmodels`
* **Server:** Uvicorn, Gunicorn

## ⚙️ Setup and Installation

Follow these steps to run the project locally.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
    cd YOUR_REPOSITORY
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of the project and add your API keys:
    ```
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="tvly-..."
    ```

## 🏃‍♀️ Running the Application

To start the local development server, run the following command in your terminal:
```bash
uvicorn app.main:app --reload
