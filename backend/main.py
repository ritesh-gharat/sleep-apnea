from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from processing import process_ecg
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model
model = tf.keras.models.load_model('model/sleep_apnea_model')

@app.post("/predict")
async def predict(
    age: int = Form(...),
    sex: int = Form(...),
    ecg_file: UploadFile = File(...)
):
    try:
        # Log received inputs
        print(f"Received inputs: age={age}, sex={sex}, ecg_file={ecg_file.filename}")

        # Read file content
        file_content = await ecg_file.read()
        
        # Check if file is empty
        if len(file_content) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "The uploaded file is empty."}
            )
        
        # Convert to numpy array
        try:
            # Parse the file content as string and then convert to numpy array
            file_str = file_content.decode('utf-8')
            ecg_data = np.array([float(line.strip()) for line in file_str.split('\n') if line.strip()])
            print(f"ECG data shape: {ecg_data.shape}")
            
            # Check if array contains data
            if ecg_data.size == 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No numeric data found in the uploaded file."}
                )
                
        except UnicodeDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": "The file must be a text file with one value per line."}
            )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "The file contains non-numeric values. Please ensure all lines contain valid numbers."}
            )

        # Process ECG data
        try:
            rri_features = process_ecg(ecg_data)
            print(f"Processed RRI features shape: {rri_features.shape}")
        except Exception as e:
            print(f"Error in signal processing: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to process ECG signal: {str(e)}"}
            )

        # Prepare model input
        demographic_features = [age, sex]
        inputs = {
            "input_1": np.expand_dims(np.column_stack([rri_features, np.zeros_like(rri_features)]), axis=0),  # Add batch dimension
            "input_2": np.array([demographic_features])
        }

        # Log input shapes for debugging
        print(f"Input shapes: input_1={inputs['input_1'].shape}, input_2={inputs['input_2'].shape}")

        # Make prediction
        prediction = model.predict(inputs)
        print(f"Prediction result: {prediction}")

        return {"apnea_probability": float(prediction[0][0])}
    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid input format. Ensure the CSV file contains a single column of ECG signal values."
            }
        )
    except Exception as e:
        # Log the error and return a 400 response
        print(f"Error during prediction: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid input or processing error"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
