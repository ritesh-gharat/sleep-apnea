# Sleep Apnea Analysis

A web-based tool for analyzing sleep apnea symptoms and patterns using machine learning.

## Features

- Web interface for data upload and analysis
- REST API backend for processing
- Machine learning model for sleep pattern detection

## Setup

1. Create conda environment:
```sh
conda create -n sleep-apnea python=3.9
conda activate sleep-apnea
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Pre-Training Model
*  Takes around 10-20 min depends on hardware
```sh
python model/pre_proc.py 
```

4. Deep Learning
* Takes around 1-2 hrs depends on hardware (trained on Nividia 3050 6GB VRAM)
```sh
python model/train.py 
```

5. Start backend server:
```sh
uvicorn backend.main:app --reload
```

4. Launch frontend:
```sh
streamlit run frontend/app.py
```

## Project Structure

```
├── frontend/              # Streamlit web interface
├── backend/               # FastAPI server
├── backend/model/         # Trained Model
└── model-training/        # ML model and data processing
└── model-training/data/   # Dataset [ECG Sleep Apnea Dataset](https://physionet.org/content/apnea-ecg/1.0.0/)
```

## Requirements

- Python 3.9
- TensorFlow 2.10
- Streamlit
- FastAPI
- Pandas
- scikit-learn
- Miniconda/Anaconda

## Team
1. Ritesh Gharat
2. Harsh Dubey
3. Vishal Chauhan
4. Puneet Choudhary

## License

MIT License