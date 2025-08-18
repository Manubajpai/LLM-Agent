import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.agent import create_data_analyst_agent
import tempfile

app = FastAPI( title="Data Analyst Agent API", version="1.0.0" )
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
            data_content = (await data_file.read()).decode("utf-8")
           
            user_prompt += f"\n\nHere is the data for your analysis, provided in CSV format:\n---\n{data_content}\n---"
        
        response = agent_executor.invoke({"input": user_prompt})
        agent_output = response.get("output")
        
       
        if isinstance(agent_output, dict):
            final_response = agent_output.get("result", {})
            if agent_output.get("plot_created"):
                final_response["plot_url"] = "/api/latest-plot"
            return JSONResponse(content=final_response)
        else:
            return JSONResponse(content={"message": str(agent_output)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/api/latest-plot", tags=["Analysis"])
async def get_latest_plot():
    plot_path = os.path.join(tempfile.gettempdir(), "latest_plot.png")
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found.")
