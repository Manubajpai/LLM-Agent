import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
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
async def analyze_data(request: Request):
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    try:
        # Manually parse the multipart form data to handle dynamic field names
        form_data = await request.form()
        
        question_file = None
        data_file = None

        # Find the question and data files by their content type or filename extension
        for key, value in form_data.items():
            if isinstance(value, UploadFile):
                if value.filename.endswith('.txt'):
                    question_file = value
                elif value.filename.endswith('.csv'):
                    data_file = value
        
        if not question_file:
            raise HTTPException(status_code=400, detail="Missing required question file (must be a .txt file).")

        user_prompt = (await question_file.read()).decode("utf-8")
        
        params = {"input": user_prompt}
        if data_file:
            # Read the data content and pass it directly into the prompt
            data_content = (await data_file.read()).decode("utf-8")
            params["input"] += f"\n\nHere is the data for your analysis, provided in CSV format:\n---\n{data_content}\n---"

        response = agent_executor.invoke(params)
        return JSONResponse(content=response.get("output"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
