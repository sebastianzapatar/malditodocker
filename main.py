# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("main")

# Create input/output pydantic models
input_model = create_model("main_input", **{'Carat Weight': 1.590000033378601, 'Cut': 'Ideal', 'Color': 'H', 'Clarity': 'VVS2', 'Polish': 'ID', 'Symmetry': 'ID', 'Report': 'AGSL'})
output_model = create_model("main_output", prediction=5169)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
