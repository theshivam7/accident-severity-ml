# Accident Severity ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
[![Live](https://img.shields.io/badge/Live-Streamlit_Cloud-brightgreen?style=flat)](https://accident-analysis-ml.streamlit.app)

**Predict motor vehicle accident severity and emergency deployment level using Karnataka State Police FIR data.**

Built for [KSP Datathon 2024](https://hack2skill.com/hack/kspdatathon2024/) · Trained on 329,000+ real accident records (2016–2024)

[Live Demo](https://accident-analysis-ml.streamlit.app) · [Dataset](https://drive.google.com/drive/folders/1iHHYRkGRhYfO-lTaEjnrkO-QQE428WLU?usp=drive_link) · [Hackathon](https://hack2skill.com/hack/kspdatathon2024/)

</div>

---

## Overview

This project analyzes Karnataka State Police (KSP) FIR data to predict whether a motor vehicle accident will be fatal or non-fatal, and recommends an appropriate emergency deployment level. It was originally developed during the KSP Datathon 2024 and has since been rebuilt with a clean data pipeline and an interactive Streamlit dashboard.

## Features

- **Severity Prediction** — predicts Fatal or Non-Fatal given district, road type, month, and year
- **Deployment Recommendation** — suggests emergency response level (High / Standard) based on prediction
- **Insights Dashboard** — 5 interactive charts covering monthly trends, severity split, top districts, road types, and fatality rate by year
- **Dark Mode UI** — theme-aware interface built with Streamlit

## Tech Stack

| Component | Tool |
|-----------|------|
| ML Model | CatBoost 1.2.10 (gradient boosting, native categorical support) |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Evaluation | scikit-learn |
| Language | Python 3.12 |
| Deployment | Streamlit Community Cloud |

## Model

| Property | Value |
|----------|-------|
| Algorithm | CatBoostClassifier |
| Features | District, Road Type, Month, Year |
| Target | Fatal / Non-Fatal |
| Training records | 263,224 |
| Test records | 65,806 |
| Test accuracy | ~59% |

> Accuracy is inherently limited — fatal vs non-fatal outcome depends heavily on speed and collision type, which are not recorded in the KSP FIR master table. The model provides a useful baseline signal from location and time context alone.

## Dataset

Source: Karnataka State Police FIR dataset, released for KSP Datathon 2024.

Download: [Google Drive](https://drive.google.com/drive/folders/1iHHYRkGRhYfO-lTaEjnrkO-QQE428WLU?usp=drive_link) (~1.8 GB, not included in repo)

| File | Description |
|------|-------------|
| `FIR_Details_Data.csv` | 1.69M FIR records with crime group, road type, district, date |
| `VictimInfoDetails.csv` | 1.47M victim records with injury type |

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/theshivam7/accident-severity-ml.git
cd accident-severity-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add raw data files to data/
#    data/FIR_Details_Data.csv
#    data/VictimInfoDetails.csv

# 4. Preprocess
python3 preprocess.py

# 5. Train
python3 train.py

# 6. Run
streamlit run app.py
```

App opens at `http://localhost:8501`.

## Project Structure

```
accident-severity-ml/
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── data/                   # Raw CSV files (gitignored — too large)
├── app.py                  # Streamlit web application
├── preprocess.py           # Data cleaning and feature engineering
├── train.py                # Model training and evaluation
├── model.cbm               # Trained CatBoost model
├── model_meta.json         # District/road type lists for app dropdowns
├── processed_data.csv      # Cleaned dataset used by Insights tab
├── requirements.txt
├── runtime.txt             # Python 3.12 pin for Streamlit Cloud
└── README.md
```

## Team

Built at [KSP Datathon 2024](https://hack2skill.com/hack/kspdatathon2024/) by:

- Shivam Sharma
- Krishnan Lakshmi Narayana
- Bhavya Vishal
- Madhuri

## License

MIT
