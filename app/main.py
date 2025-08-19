import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
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
async def analyze_data( question_file: UploadFile = File(...), data_file: UploadFile = File(None) ):
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    try:
        user_prompt = (await question_file.read()).decode("utf-8")
        
        params = {"input": user_prompt}
        if data_file:
            # Save the uploaded data file to a temporary path that tools can access
            temp_dir = tempfile.gettempdir()
            temp_data_path = os.path.join(temp_dir, data_file.filename)
            with open(temp_data_path, "wb") as f:
                f.write(await data_file.read())
            # Add the file path to the prompt for the specialist tools
            params["input"] += f"\nFile is at path: {temp_data_path}"

        response = agent_executor.invoke(params)
        return JSONResponse(content=response.get("output"))

    except Exception as e:
        # Clean up temp file on error if it exists
        if 'temp_data_path' in locals() and os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# done
