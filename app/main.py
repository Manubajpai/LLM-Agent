import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.agent import create_data_analyst_agent
import tempfile

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses an LLM agent to analyze data.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    agent_executor = create_data_analyst_agent()
except Exception as e:
    print(f"Startup Error: {e}")
    agent_executor = None

@app.post("/api/", tags=["Analysis"])
async def analyze_data( question_file: UploadFile = File(...), data_file: UploadFile = File(None) ):
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    try:
        user_prompt = (await question_file.read()).decode("utf-8")
        if data_file:
            temp_dir = tempfile.gettempdir()
            temp_data_path = os.path.join(temp_dir, data_file.filename)
            with open(temp_data_path, "wb") as f:
                f.write(await data_file.read())
            user_prompt += f"\n\nCRITICAL: Use the data from the CSV file at: {temp_data_path}"

        response = agent_executor.invoke({"input": user_prompt})
        agent_output = response.get("output")

        final_response = {}
        if isinstance(agent_output, dict):
            final_response = agent_output.get("result", {})
            if agent_output.get("plot_created"):
                final_response["plot_url"] = "/api/latest-plot"
        else:
            final_response = {"message": str(agent_output)}

        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/api/latest-plot", tags=["Analysis"])
async def get_latest_plot():
    """Serves the most recently generated plot image."""
    plot_path = os.path.join(tempfile.gettempdir(), "plot.png")
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found.")