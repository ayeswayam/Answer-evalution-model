import os
import sys
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the evaluation system
from answer_evaluation_model import AnswerEvaluationSystem, evaluate_single_answer, evaluate_answers

app = FastAPI(title="Answer Evaluation System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize evaluator
evaluator = AnswerEvaluationSystem()

class AnswerRequest(BaseModel):
    student_answer: str
    model_answer: str

class BatchEvaluationRequest(BaseModel):
    student_answers: list[str]
    correct_answers: list[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to Answer Evaluation System API"}

@app.post("/evaluate")
def evaluate_answer(request: AnswerRequest):
    """Evaluate a single answer"""
    try:
        result = evaluate_single_answer(request.student_answer, request.model_answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/evaluate-batch")
def evaluate_batch_answers(request: BatchEvaluationRequest):
    """Evaluate a batch of answers"""
    try:
        results = evaluate_answers(
            request.student_answers, 
            request.correct_answers
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {str(e)}")

@app.post("/evaluate-csv")
async def evaluate_batch_from_csv(file: UploadFile = File(...)):
    """Evaluate a batch of answers from a CSV file"""
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check if required columns exist
        required_columns = ['student_answer', 'model_answer']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV file must contain columns: {', '.join(required_columns)}"
            )
            
        # Process each answer
        results = []
        for _, row in df.iterrows():
            result = evaluate_single_answer(row['student_answer'], row['model_answer'])
            # Add question_id if available
            if 'question_id' in row:
                result['question_id'] = row['question_id']
            results.append(result)
            
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV evaluation error: {str(e)}")

# Add port binding for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
