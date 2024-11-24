from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import pandas as pd
import io
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the logistic regression model pipeline
model = joblib.load('logreg_model.joblib')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(
    temperature: float = Form(...),
    ph: float = Form(...),
    ps_data: UploadFile = File(...)
):
    # Validate file type
    if not ps_data.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an excel file (xlsx).")
    
    # Read CSV file
    try:
        contents = await ps_data.read()
        data = pd.read_excel(io.BytesIO(contents), names=["Potential(V)", "Current (uA)"])
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading CSV file.")
    
    data = data.drop(0)
    data['pH'] = float(ph)
    data['Temperature'] = float(temperature)
    data['Interaction_Current_Potential'] = data['Current (uA)'] * data['Potential(V)']
    
    # Make a prediction
    prediction = model.predict(data)
   
    # Return the prediction
    return {"potential": list(data['Potential(V)']), "current": list(data["Current (uA)"]), "prediction": list(prediction), "temperature": temperature, "ph": ph}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)