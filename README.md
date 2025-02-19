# Sleep Apnea Analysis

A web-based tool for analyzing sleep apnea symptoms and patterns using machine learning.

## Features

- Web interface for data upload and analysis
- REST API backend for processing
- Machine learning model for sleep pattern detection

## Setup

1. Install dependencies:
```sh
pip install -r requirements.txt
```

2. Start backend server:
```sh
uvicorn backend.main:app --reload
```

3. Launch frontend:
```sh
streamlit run frontend/app.py
```

## Project Structure

```
├── frontend/     # Streamlit web interface
├── backend/      # FastAPI server
└── model/        # ML model and data processing
└── model/data/   # Dataset [ECG Sleep Apnea Dataset](https://physionet.org/content/apnea-ecg/1.0.0/)
```

## Requirements

- Python 3.7+
- Streamlit
- FastAPI
- Pandas
- scikit-learn

## License

MIT License