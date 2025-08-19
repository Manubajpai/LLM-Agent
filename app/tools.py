import pandas as pd
from langchain.tools import tool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import duckdb
import networkx as nx
import requests


@tool
def analyze_wikipedia_movies() -> list:
    """
    Use this tool for the specific question about highest-grossing films from Wikipedia.
    It scrapes the data, performs all required analysis, and returns a list of results
    including a base64 encoded plot.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        # Use read_html to find tables with the 'wikitable' class for better reliability
        df = pd.read_html(url, attrs={'class': 'wikitable'})[0]
        
        # Robust data cleaning
        df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.replace(r'\[.*\]', '', regex=True).str.replace('$', '').str.replace(',', '').astype(float)
        df['Year'] = df['Year'].astype(str).str.replace(r'\[.*\]', '', regex=True).astype(int)
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df.dropna(subset=['Peak', 'Rank'], inplace=True)

        # Analysis
        count_before_2000 = len(df[(df['Worldwide gross'] >= 2000000000) & (df['Year'] < 2000)])
        earliest_film = df[df['Worldwide gross'] > 1500000000].sort_values(by='Year').iloc[0]['Title']
        correlation = df['Rank'].corr(df['Peak'])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Rank'], df['Peak'])
        m, b = np.polyfit(df['Rank'], df['Peak'], 1)
        plt.plot(df['Rank'], m*df['Rank'] + b, color='red', linestyle='dotted')
        plt.title('Rank vs Peak')
        plt.xlabel('Rank')
        plt.ylabel('Peak')
        plt.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png'); plt.close()
        buf.seek(0)
        plot_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        return [count_before_2000, earliest_film, correlation, plot_b64]
    except Exception as e:
        plt.close('all')
        return [f"Error in analyze_wikipedia_movies: {str(e)}"]

@tool
def analyze_sales_csv(file_path: str) -> dict:
    """
    Use this tool for the specific question about sales analysis from a sales CSV file.
    It returns a dictionary of results including two base64 encoded plots.
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculations
        total_sales = df['sales'].sum()
        top_region = df.groupby('region')['sales'].sum().idxmax()
        df['day_of_month'] = df['date'].dt.day
        correlation = df['day_of_month'].corr(df['sales'])
        median_sales = df['sales'].median()
        total_sales_tax = total_sales * 0.10
        
        # Bar Chart
        plt.figure(figsize=(10, 5))
        df.groupby('region')['sales'].sum().plot(kind='bar', color='blue')
        plt.title('Total Sales by Region'); plt.ylabel('Total Sales'); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        bar_chart_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # Line Chart
        df_sorted = df.sort_values('date')
        df_sorted['cumulative_sales'] = df_sorted['sales'].cumsum()
        plt.figure(figsize=(10, 5))
        plt.plot(df_sorted['date'], df_sorted['cumulative_sales'], color='red')
        plt.title('Cumulative Sales Over Time'); plt.ylabel('Cumulative Sales'); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        line_chart_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        return {
            "total_sales": total_sales,
            "top_region": top_region,
            "day_sales_correlation": correlation,
            "bar_chart": bar_chart_b64,
            "median_sales": median_sales,
            "total_sales_tax": total_sales_tax,
            "cumulative_sales_chart": line_chart_b64
        }
    except Exception as e:
        plt.close('all')
        return {"error": str(e)}

@tool
def analyze_weather_csv(file_path: str) -> dict:
    """
    Use this tool for the specific question about analyzing a weather CSV file.
    Returns a dictionary of results including two base64 encoded plots.
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        results = {
            "average_temp_c": df['temperature_c'].mean(),
            "max_precip_date": df.loc[df['precip_mm'].idxmax()]['date'].strftime('%Y-%m-%d'),
            "min_temp_c": df['temperature_c'].min(),
            "temp_precip_correlation": df['temperature_c'].corr(df['precip_mm']),
            "average_precip_mm": df['precip_mm'].mean()
        }

        plt.figure(figsize=(10, 5)); df.plot(x='date', y='temperature_c', kind='line', title='Temperature Over Time', color='red'); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        results["temp_line_chart"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        plt.figure(figsize=(10, 5)); df['precip_mm'].plot(kind='hist', title='Precipitation Distribution', color='orange', edgecolor='black'); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        results["precip_histogram"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        return results
    except Exception as e:
        plt.close('all')
        return {"error": str(e)}

@tool
def analyze_network_graph(file_path: str) -> dict:
    """
    Use this tool for the specific question about analyzing a network graph from an edges.csv file.
    Returns a dictionary of results including two base64 encoded plots.
    """
    try:
        df = pd.read_csv(file_path)
        G = nx.from_pandas_edgelist(df, 'source', 'target') # Undirected graph
        
        results = {
            "edge_count": G.number_of_edges(),
            "highest_degree_node": max(dict(G.degree()).items(), key=lambda x: x[1])[0],
            "average_degree": np.mean([d for n, d in G.degree()]),
            "density": nx.density(G),
            "shortest_path_alice_eve": nx.shortest_path_length(G, source='Alice', target='Eve')
        }
        
        plt.figure(figsize=(12, 12)); pos = nx.spring_layout(G); nx.draw(G, pos, with_labels=True, node_color='skyblue'); plt.title('Network Graph')
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        results["network_graph"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        plt.figure(figsize=(10, 5)); plt.bar(range(len(degree_sequence)), degree_sequence, color='green'); plt.title('Degree Histogram'); plt.xlabel('Node Rank'); plt.ylabel('Degree')
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        results["degree_histogram"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        return results
    except Exception as e:
        plt.close('all')
        return {"error": str(e)}

@tool
def analyze_indian_court_data() -> dict:
    """
    Use this tool for the specific question about the Indian High Court dataset.
    It runs predefined DuckDB queries and returns a dictionary of results including a base64 plot.
    """
    try:
        con = duckdb.connect(database=':memory:')
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet; SET s3_region='ap-south-1';")
        
        base_query = "FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')"
        
        # Q1
        q1_df = con.execute(f"SELECT court, COUNT(*) as case_count {base_query} WHERE year BETWEEN 2019 AND 2022 GROUP BY court ORDER BY case_count DESC LIMIT 1").fetchdf()
        top_court = q1_df['court'][0]
        
        # Q2 & Q3
        q2_query = f"SELECT year, CAST(decision_date AS DATE) - CAST(date_of_registration AS DATE) as delay_days {base_query} WHERE court_code = '33~10' AND decision_date IS NOT NULL AND date_of_registration IS NOT NULL"
        delay_df = con.execute(q2_query).fetchdf()
        delay_df.dropna(inplace=True)
        delay_by_year = delay_df.groupby('year')['delay_days'].mean().reset_index()
        slope, intercept = np.polyfit(delay_by_year['year'], delay_by_year['delay_days'], 1)

        plt.figure(figsize=(10, 6))
        plt.scatter(delay_by_year['year'], delay_by_year['delay_days'])
        plt.plot(delay_by_year['year'], slope * delay_by_year['year'] + intercept, 'r--')
        plt.title('Case Delay by Year (Court 33~10)'); plt.xlabel('Year'); plt.ylabel('Average Delay (Days)'); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        plot_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        return {
            "high_court_most_cases_2019_2022": top_court,
            "regression_slope_delay_by_year": slope,
            "delay_scatterplot": plot_b64
        }
    except Exception as e:
        plt.close('all')
        return {"error": str(e)}


@tool
def python_code_interpreter(code: str) -> dict:
    """
    A general-purpose tool to execute a Python script for any unforeseen or simple task.
    The script must assign its final answer to a variable `final_result`.
    """
    local_vars = { "pd": pd, "plt": plt, "np": np, "io": io, "base64": base64, "requests": requests, "nx": nx, "duckdb": duckdb }
    try:
        exec(code, local_vars)
        if 'final_result' in local_vars:
            return { "result": local_vars['final_result'] }
        else:
            raise ValueError("Script did not assign a value to 'final_result'.")
    except Exception as e:
        plt.close('all')
        return {"error": f"Error executing code: {type(e).__name__} - {str(e)}"}
