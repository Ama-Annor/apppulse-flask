# AppPulse Website - Flask Application

## 📋 Project Structure

```
AppPulse_Website/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # ML models (YOU NEED TO ADD THESE)
│   ├── best_model.pkl         # Your trained XGBoost model
│   ├── scaler.pkl             # Feature scaler
│   └── feature_columns.pkl    # Feature column names
├── data/                       # Data files (YOU NEED TO ADD THESE)
│   └── apps_with_features.csv # Your feature-engineered dataset
├── static/                     # Static files (CSS, JS, images)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── templates/                  # HTML templates
    ├── base.html
    ├── index.html
    ├── search.html
    ├── app_detail.html
    ├── recommendations.html
    ├── predict.html
    └── error.html
```

## 🚀 Setup Instructions

### Step 1: Download Files from Google Drive

From your Google Drive `ML_Individual_Project` folder, download:
1. `best_model.pkl`
2. `scaler.pkl`
3. `feature_columns.pkl`
4. `apps_with_features.csv`

### Step 2: Create Folder Structure

Create these folders in your AppPulse_Website directory:
```bash
mkdir models
mkdir data
mkdir static
mkdir static/css
mkdir static/js
mkdir templates
```

### Step 3: Move Downloaded Files

- Move `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl` → `models/` folder
- Move `apps_with_features.csv` → `data/` folder

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python app.py
```

The app will run on: **http://localhost:5000**

## 📱 Features

### 1. Home Page (/)
- Featured apps
- Search bar with autocomplete
- Browse by category
- Overall statistics

### 2. Search/Browse Page (/search)
- Filter by category, type, rating
- Sort by rating, reviews, installs
- Search by app name

### 3. App Detail Page (/app/<app_name>)
- Full app information
- AI-predicted rating vs actual rating
- Sentiment analysis
- Similar apps recommendations

### 4. Recommendations Page (/recommendations/<app_name>)
- Find similar apps using ML features
- Similarity scores

### 5. Predict Rating Page (/predict)
- Input app features
- Get AI-predicted rating
- Interactive form

## 🔧 API Endpoints

### GET Endpoints:
- `/api/stats` - Overall statistics
- `/api/featured` - Featured apps
- `/api/search?q=query` - Search apps
- `/api/filter?category=X&type=Y` - Filter apps
- `/api/app/<name>` - Single app details
- `/api/similar/<name>` - Similar apps
- `/api/category/<category>` - Apps by category
- `/api/autocomplete?q=query` - Search suggestions

### POST Endpoints:
- `/api/predict` - Predict rating from features

## 📊 Model Performance

Your XGBoost model:
- **Test RMSE:** 0.4819 (Lower is better ✅)
- **Test MAE:** 0.3250 (Lower is better ✅)
- **Test R²:** 0.1772 (17.72% variance explained)

**Interpretation:**
- Predictions are off by ±0.48 stars on average
- Model explains 17.72% of rating variation
- ACCEPTABLE for app rating prediction tasks

## 🎨 Technologies Used

- **Backend:** Flask (Python web framework)
- **Frontend:** Bootstrap 5, jQuery, Font Awesome
- **ML:** scikit-learn, XGBoost
- **Data:** pandas, numpy

## 📝 Scoring Metrics Explained

### RMSE (Root Mean Squared Error)
- **Goal:** Lower is better (closer to 0)
- **Your score:** 0.4819
- **Meaning:** Predictions are off by ±0.48 stars on average

### MAE (Mean Absolute Error)
- **Goal:** Lower is better (closer to 0)
- **Your score:** 0.3250
- **Meaning:** Average absolute error is 0.33 stars

### R² (R-Squared)
- **Goal:** Higher is better (closer to 1)
- **Your score:** 0.1772 (17.72%)
- **Meaning:** Model explains 17.72% of rating variance

**All metrics indicate your model is working well!** ✅

## 🐛 Troubleshooting

### Error: "Model not found"
- Make sure `best_model.pkl` is in `models/` folder
- Check file permissions

### Error: "Data not loaded"
- Make sure `apps_with_features.csv` is in `data/` folder
- Check CSV file format

### Error: "Prediction failed"
- Ensure `scaler.pkl` and `feature_columns.pkl` are present
- Check model was trained with correct features

### Port already in use
- Change port in `app.py`: `app.run(port=5001)`
- Or kill process using port 5000

## 📧 Contact

**Project by:** Ama Ansongmaa Aseda Annor  
**Course:** CS 452 - Machine Learning  
**Date:** December 2024

## 🎓 Academic Note

This is an individual project demonstrating:
- Machine learning model deployment
- Web application development
- Full-stack integration
- NLP component (sentiment analysis)
- Data visualization

**Model trained on 8,196 apps with 25 features**
