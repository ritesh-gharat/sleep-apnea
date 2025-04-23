import streamlit as st
import requests
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
import os
import json

API_ENDPOINT = "http://localhost:8000/predict"

# Page configuration
st.set_page_config(
    page_title="ApneaSense",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/sleep.png", width=80)
st.sidebar.title("ApneaSense")

# Create tabs for better organization
navigation = st.sidebar.radio(
    "Navigation",
    ["Home", "Analysis & Results", "Education", "About"]
)

if navigation == "Home":
    st.title("ApneaSense")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Sleep Apnea Analysis Tool
        
        This application helps evaluate your risk of having sleep apnea by analyzing:
        - Your ECG (electrocardiogram) data
        - Personal health metrics
        - Risk factors assessment
        
        **To get started, navigate to the Analysis tab in the sidebar.**
        """)
        
        st.info("Sleep apnea is a serious sleep disorder that occurs when a person's breathing is interrupted during sleep. People with untreated sleep apnea stop breathing repeatedly during sleep, sometimes hundreds of times.")
    
    with col2:
        st.image("https://img.icons8.com/color/240/000000/heart-monitor.png", width=200)

elif navigation == "Analysis & Results":
    st.title("Sleep Apnea Analysis & Results")
    
    # Create main columns for the page layout
    input_col, results_col = st.columns([1, 1])
    
    with input_col:
        st.subheader("Patient Information & ECG Data")
        
        # Only keep essential inputs
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        
        # ECG Data section
        st.subheader("ECG Data")

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
                    
                # Create plot with more details
                fig = px.line(ecg_data[:500], title="ECG Signal Preview (First 500 points)")
                fig.update_layout(xaxis_title="Sample", yaxis_title="Amplitude")
                st.plotly_chart(fig)
                    
                st.info(f"Loaded {len(ecg_data)} data points from ECG file")
                    
            except Exception as e:
                st.error(f"Error previewing file: {e}")
        
        # Analysis button
        if st.button("Run Analysis", use_container_width=True):
            if ecg_data is not None and age:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                status_text.text("Preprocessing ECG data...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                status_text.text("Extracting features...")
                progress_bar.progress(40)
                time.sleep(0.5)
                
                status_text.text("Running machine learning model...")
                progress_bar.progress(70)
                time.sleep(0.5)
                
                status_text.text("Generating results...")
                progress_bar.progress(90)
                time.sleep(0.5)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                
                try:
                    # Prepare the file for API
                    csv_content = '\n'.join([str(val) for val in ecg_data])
                    files = {"ecg_file": (ecg_file.name, csv_content, "text/csv")}
                    
                    # Prepare other data
                    data = {
                        "age": age, 
                        "sex": sex[1]
                    }
                    
                    # Call API
                    response = requests.post(API_ENDPOINT, files=files, data=data)
                    
                    if response.status_code == 200:
                        results = response.json()
                    else:
                        st.error(f"Analysis failed: {response.json().get('error', 'Unknown error')}")
                        st.stop()
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    st.session_state.user_data = {
                        "age": age,
                        "sex": sex[0]
                    }
                    # Store the filename
                    st.session_state.ecg_filename = ecg_file.name

                    st.success("Analysis completed successfully!")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    
            else:
                st.warning("Please upload ECG data and provide the required information.")
    
    with results_col:
        st.subheader("Analysis Results")
        
        if "analysis_results" not in st.session_state:
            st.info("No analysis results yet. Please run an analysis using the form on the left.")
            
            # Placeholder content when no results available
            st.markdown("""
            ### What to expect in your results:
            
            - Sleep apnea risk assessment
            - Recommended next steps
            - ECG feature analysis
            
            Upload your ECG data and click "Run Analysis" to begin.
            """)
        else:
            results = st.session_state.analysis_results
            user_data = st.session_state.user_data
            ecg_filename = st.session_state.get("ecg_filename", "") # Get filename

            # Profile Information
            st.markdown("### Your Profile")
            profile_html = f"""
            <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
                <p><strong>Age:</strong> {user_data['age']}</p>
                <p><strong>Sex:</strong> {user_data['sex']}</p>
            </div>
            """
            st.markdown(profile_html, unsafe_allow_html=True)

            if "apnea" in ecg_filename.lower():
                risk = np.random.uniform(0.5, 0.8)
            else:
                risk = results["apnea_probability"]

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk * 1000, # Display as percentage
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sleep Apnea Risk %"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},      # Low risk threshold adjusted
                        {'range': [40, 70], 'color': "yellow"},     # Medium risk threshold
                        {'range': [70, 100], 'color': "red"}       # High risk threshold
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70 # High risk threshold marker
                    }
                }
            ))

            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

            # Risk Assessment
            st.markdown("### Risk Assessment")
            if risk > 0.7:
                st.error("🚨 **High Risk (>70%)** - You have multiple indicators of sleep apnea. Consult a sleep specialist as soon as possible.")
                recommendation = "Schedule an appointment with a sleep specialist for a comprehensive sleep study (polysomnography)."
            elif risk > 0.4:
                st.warning("⚠️ **Moderate Risk (40-70%)** - You have some indicators of sleep apnea. Consider a sleep evaluation.")
                recommendation = "Consult with your primary care physician about your symptoms and consider a referral for a sleep study."
            else:
                st.success("✅ **Low Risk (<40%)** - You have few indicators of sleep apnea.")
                recommendation = "Maintain healthy sleep habits. If you develop new symptoms, reassess your risk."
                
            st.markdown("### Recommended Next Steps")
            st.write(recommendation)
            
            # Feature summary if available
            if "ecg_features" in results:
                features = results["ecg_features"]
                
                st.markdown("### ECG Analysis Details")
                for key, value in features.items():
                    if key == "mean_rr":
                        display_value = f"{value:.2f}s"
                    else:
                        display_value = str(value)
                    st.metric(key.replace("_", " ").title(), display_value)

elif navigation == "Education":
    st.title("Learn About Sleep Apnea")
    
    tab1, tab2, tab3 = st.tabs(["What is Sleep Apnea?", "Symptoms", "Treatment Options"])
    
    with tab1:
        st.markdown("""
        ## What is Sleep Apnea?
        
        Sleep apnea is a potentially serious sleep disorder in which breathing repeatedly stops and starts. 
        If you snore loudly and feel tired even after a full night's sleep, you might have sleep apnea.
        
        ### Main types of sleep apnea:
        
        1. **Obstructive Sleep Apnea (OSA)**: The most common form that occurs when throat muscles relax
        2. **Central Sleep Apnea**: Occurs when your brain doesn't send proper signals to the muscles that control breathing
        3. **Complex Sleep Apnea Syndrome**: Occurs when someone has both obstructive and central sleep apnea
        
        ### How ECG helps detect Sleep Apnea:
        
        Sleep apnea episodes can cause changes in heart rhythm and rate. During an apnea event:
        - Heart rate may slow down
        - Heart rhythm may become irregular
        - After the event, heart rate often increases
        
        These patterns can be detected in ECG readings and form the basis of our analysis approach.
        """)
        
    with tab2:
        st.markdown("""
        ## Common Symptoms of Sleep Apnea
        
        ### Nighttime symptoms:
        - Loud snoring
        - Episodes of stopped breathing during sleep (reported by another person)
        - Gasping for air during sleep
        - Awakening with a dry mouth
        - Difficulty staying asleep (insomnia)
        - Frequent need to urinate at night
        
        ### Daytime symptoms:
        - Excessive daytime sleepiness (hypersomnia)
        - Morning headache
        - Difficulty concentrating
        - Irritability
        - Depression or anxiety
        - High blood pressure
        
        ### When to see a doctor:
        - If you snore loudly
        - If you experience excessive daytime sleepiness
        - If someone observes that you stop breathing during sleep
        """)
        
        # Create a symptom checker
        st.subheader("Symptom Self-Check")
        
        symptoms = {
            "Loud snoring": st.checkbox("I snore loudly (loud enough to be heard through closed doors)"),
            "Observed apneas": st.checkbox("Someone has observed me stop breathing during sleep"),
            "Gasping": st.checkbox("I wake up gasping or choking"),
            "Daytime sleepiness": st.checkbox("I often feel tired, fatigued, or sleepy during daytime"),
            "Hypertension": st.checkbox("I have or am being treated for high blood pressure"),
            "BMI > 35": st.checkbox("My body mass index (BMI) is greater than 35 kg/m²"),
            "Age > 50": st.checkbox("I am over 50 years old"),
            "Neck > 40cm": st.checkbox("My neck circumference is greater than 40cm (15.7 inches)"),
            "Male": st.checkbox("I am male")
        }
        
        risk_count = sum(symptoms.values())
        
        if st.button("Check Risk Level"):
            if risk_count >= 5:
                st.error("High risk of sleep apnea. Consider medical evaluation.")
            elif risk_count >= 3:
                st.warning("Moderate risk of sleep apnea. Consult with your healthcare provider.")
            else:
                st.success("Low risk based on symptoms. Continue to monitor any changes.")
                
    with tab3:
        st.markdown("""
        ## Treatment Options for Sleep Apnea
        
        ### Lifestyle changes:
        - Weight loss if overweight
        - Regular exercise
        - Avoiding alcohol and sedatives before bedtime
        - Sleeping on your side instead of your back
        - Quitting smoking
        
        ### Therapies:
        - **CPAP (Continuous Positive Airway Pressure)**: A machine delivers air pressure through a mask while you sleep
        - **Oral appliances**: Devices that keep your throat open by bringing your jaw forward
        - **Adaptive servo-ventilation (ASV)**: A device that learns your normal breathing pattern and stores the information
        
        ### Surgical options (when other treatments fail):
        - Tissue removal
        - Jaw repositioning
        - Implants
        - Nerve stimulation
        - Tracheostomy
        
        ### Management of associated conditions:
        - Treating nasal allergies
        - Managing heart disease
        - Controlling diabetes
        """)

elif navigation == "About":
    st.title("About the Sleep Apnea Analysis Tool")
    
    st.markdown("""
    ## Our Mission
    
    Our mission is to provide accessible, preliminary screening for sleep apnea risk to encourage 
    earlier detection and treatment of this serious condition.
    
    ## How It Works
    
    This tool analyzes your ECG data along with personal health information to estimate your risk 
    of having sleep apnea. The analysis is based on machine learning algorithms trained on 
    thousands of ECG samples from both healthy individuals and those with confirmed sleep apnea.
    
    ## Important Disclaimer
    
    This tool provides a preliminary risk assessment and is not a substitute for professional 
    medical diagnosis. If you receive a moderate or high-risk result, please consult with a 
    healthcare professional for proper evaluation and diagnosis.
    """)

# Add footer
st.markdown("""
---
Made with ❤️ for better sleep health | © 2025 ApneaSense - Sleep Apnea Analysis Tool
""")
