import os
import pandas as pd
from langchain.tools import tool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import tempfile
import requests
from bs4 import BeautifulSoup
import lxml
import statsmodels
import duckdb
import networkx as nx

@tool
def python_code_interpreter(code: str) -> dict:
    """
    Executes a python script in a sandboxed environment and returns the result.
    The script MUST assign its final answer (a dictionary or a list) to a variable called `final_result`.
    If a plot is generated, the script MUST save it to a temporary file path.
    """
    # Create a unique temporary file path for the plot for this session
    plot_path = os.path.join(tempfile.gettempdir(), "plot.png")
    if os.path.exists(plot_path):
        os.remove(plot_path)

    local_vars = {
        "pd": pd, "plt": plt, "np": np, "io": io, "base64": base64,
        "requests": requests, "BeautifulSoup": BeautifulSoup, "duckdb": duckdb,
        "plot_path": plot_path, "nx": nx, "statsmodels": statsmodels, "lxml": lxml,
        "final_result": None, "tempfile": tempfile, "os": os
    }
    
    try:
        exec(code, local_vars)
        if 'final_result' in local_vars:
            was_plot_created = os.path.exists(plot_path) and os.path.getsize(plot_path) > 0
            
            return {
                "result": local_vars['final_result'],
                "plot_created": was_plot_created
            }
        else:
            raise ValueError("The script did not assign a value to the 'final_result' variable.")

    except Exception as e:
        plt.close('all')
        if os.path.exists(plot_path):
            os.remove(plot_path)
        return {"error": f"Error executing code: {type(e).__name__} - {str(e)}"}