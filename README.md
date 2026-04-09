# Crop Recommendation System 🌱

An end-to-end Machine Learning web application to predict the most suitable crop based on soil testing and environmental parameters.

## Project Overview
This project uses the Crop Recommendation dataset to train an ensemble of Machine Learning models. The top-performing algorithm (Random Forest) is utilized in a Flask web application, which provides users with a clean, glassmorphic UI to input their soil statistics and instantly receive an optimal crop recommendation.

### Features
* **Predictive AI**: Predicts among 20+ different crop types.
* **Modern Interface**: Custom CSS incorporating modern, aesthetic glassmorphism elements.
* **Comprehensive EDA**: Jupyter Notebook with Exploratory Data Analysis and Model Evaluation metrics (Confusion Matrices, Feature Importances).

---

## Folder Structure

```
├── app.py                     # Main Flask Application
├── build_model.py             # Script mapping EDA to ML Pipeline and exporting notebooks
├── model.pkl                  # Serialized best-performing ML model (Random Forest)
├── notebook.ipynb             # Jupyter Notebook generated programmatically containing ML experiments
├── requirements.txt           # Environment dependencies required
├── dataset/                   
│   └── Crop_recommendation.csv # Harvestify Dataset
├── static/                    # Generated Visualizations & CSS
│   ├── style.css                      
│   ├── accuracy_comparison.png        
│   ├── confusion_matrix.png           
│   ├── correlation_heatmap.png        
│   └── feature_importance.png         
└── templates/                 # HTML UI layouts
    ├── index.html                     
    └── result.html                    
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your machine.

### 2. Install Required Libraries
Install the necessary python dependencies using pip:
```bash
pip install -r requirements.txt
```

### 3. Generate ML Models (Optional)
The script `build_model.py` creates the serialized `.pkl` file and the Jupyter Notebook:
```bash
python build_model.py
```
*(Note: A `model.pkl` is already included if generated previously)*

### 4. Run the Web Application
Start the Flask server to launch the frontend interface:
```bash
python app.py
```
After executing this command, open your browser and navigate to `http://localhost:5000` (or the internal network IP provided in the console).

---

## 🔬 Model Performance
We evaluated the following classical Machine Learning models:
* **Random Forest** (Selected - Accuracy > 99%)
* **Decision Tree** 
* **Logistic Regression**
* **K-Nearest Neighbors**

Refer to `static/accuracy_comparison.png` for metrics mapped across models. The prediction pipeline leverages the serialized `model.pkl` to process novel vectors inputted via the web template.
