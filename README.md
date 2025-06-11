# Accident Severity Prediction System

A machine learning-based web application that predicts accident severity and recommends appropriate emergency response levels based on various factors such as road type, weather conditions, and accident location.

## 🚀 Features

- Real-time accident severity prediction
- Emergency response level recommendations
- Interactive web interface
- Modern and responsive design
- Cached model loading for better performance

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: CatBoost Classifier
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Deployment**: Streamlit Cloud

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/theshivam7/KSP.git
cd KSP
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

1. Make sure you're in the project directory and your virtual environment is activated.

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## 📊 Model Training

The application uses a CatBoost Classifier for prediction. The model is automatically trained when you first run the application if no pre-trained model exists. The model is saved as `catboost_model.cbm` for future use.

## 🎯 Input Parameters

The application takes the following inputs:
- Road Type
- Weather Condition
- Accident Location
- Collision Type

## 📈 Output

The application provides:
1. Predicted accident severity level
2. Recommended deployment level
3. Detailed emergency response recommendations

## 🔒 Security

- All dependencies are updated to their latest secure versions
- Input validation and error handling implemented
- Secure model loading and prediction pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Shivam - Initial work

## 🙏 Acknowledgments

- Thanks to all contributors and users of this project
- Special thanks to the Streamlit team for their amazing framework 