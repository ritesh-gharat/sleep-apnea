from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Sleep Apnea Analysis API"}

@app.post("/analyze")
async def analyze(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    symptoms: str = Form("Snoring, daytime sleepiness, morning headaches..."),
    file: UploadFile = File(...)
):
    try:
        df = pd.read_csv(file.file)  # Read uploaded CSV file
    except Exception:
        return {"error": "Invalid or unreadable CSV file"}

    return {
        "patient": {"name": name, "age": age, "gender": gender, "symptoms": symptoms},
        "file_name": file.filename,
        "analysis_result": "Processing data... (Replace with actual analysis)"
    }
