import streamlit as st
import requests
import numpy as np
import plotly.express as px
import io

API_ENDPOINT = "http://localhost:8000/predict"

st.title("Sleep Apnea Risk Assessment")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100)
sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])

# ECG File Upload
ecg_file = st.file_uploader("Upload ECG Data (CSV)", type=["csv"])
ecg_data = None

if ecg_file:
    # Preview ECG
    try:
        # Read the file content and convert to numpy array
        ecg_content = ecg_file.read()
        ecg_file.seek(0)  # Reset file pointer to beginning
        
        # Parse the content
        ecg_str = ecg_content.decode('utf-8')
        ecg_data = np.array([float(line.strip()) for line in ecg_str.split('\n') if line.strip()])
        
        # Create plot
        fig = px.line(ecg_data, title="ECG Signal Preview")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error previewing file: {e}")

if st.button("Analyze"):
    if ecg_file and age and sex is not None:
        try:
            # Create a fresh copy of the file
            if ecg_data is not None:
                # Create a new file-like object from data
                csv_content = '\n'.join([str(val) for val in ecg_data])
                files = {"ecg_file": (ecg_file.name, csv_content, "text/csv")}
            else:
                # Reset the file pointer to the beginning
                ecg_file.seek(0)
                files = {"ecg_file": (ecg_file.name, ecg_file, "text/csv")}
                
            data = {"age": age, "sex": sex[1]}

            response = requests.post(API_ENDPOINT, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                risk = result["apnea_probability"]
                st.metric("Apnea Risk Probability", f"{risk*100:.1f}%")

                # Interpretation
                if risk > 0.7:
                    st.error("High risk of sleep apnea - Consult a specialist")
                elif risk > 0.4:
                    st.warning("Moderate risk - Consider sleep study")
                else:
                    st.success("Low risk - Maintain healthy sleep habits")
            else:
                st.error(f"Analysis failed: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the API: {e}")
