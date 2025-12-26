"""
AppPulse - Google Play Store App Intelligence & Recommender
Flask Backend Application - IMPROVED MODEL (36.34% R²)
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'apppulse-secret-key-2024'

# =============================================================================
# DATA LOADING
# =============================================================================

DATA = {}

def load_data():
    """Load all models and data files"""
    global DATA
    print("🔄 Loading AppPulse data (IMPROVED MODEL)...")
    
    # Load Improved Model (Stacking Ensemble)
    try:
        with open('models/best_model.pkl', 'rb') as f:
            DATA['model'] = pickle.load(f)
        print("✓ Improved model loaded (Stacking Ensemble - 36.34% R²)")
    except Exception as e:
        print(f"⚠ Model not found: {e}")
        DATA['model'] = None
    
    # Load Feature Selector (not used directly, but loaded for completeness)
    try:
        with open('models/feature_selector.pkl', 'rb') as f:
            DATA['feature_selector'] = pickle.load(f)
        print("✓ Feature selector loaded")
    except Exception as e:
        print(f"⚠ Feature selector not found: {e}")
        DATA['feature_selector'] = None
    
    # Load Selected Features (50 features)
    try:
        with open('models/selected_features.pkl', 'rb') as f:
            DATA['feature_columns'] = pickle.load(f)
        print(f"✓ Selected features loaded ({len(DATA['feature_columns'])} features)")
    except Exception as e:
        print(f"⚠ Selected features not found: {e}")
        DATA['feature_columns'] = None
    
    # Load Scaler
    try:
        with open('models/scaler.pkl', 'rb') as f:
            DATA['scaler'] = pickle.load(f)
        print("✓ Scaler loaded")
    except Exception as e:
        print(f"⚠ Scaler not found: {e}")
        DATA['scaler'] = None
    
    # Load Apps Data
    try:
        DATA['apps'] = pd.read_csv('data/apps_with_features.csv')
        # Ensure numeric columns are numeric
        for col in ['Rating', 'Reviews', 'Size_MB', 'Installs_Clean', 'Price_Clean']:
            if col in DATA['apps'].columns:
                DATA['apps'][col] = pd.to_numeric(DATA['apps'][col], errors='coerce')
        print(f"✓ Apps loaded ({len(DATA['apps']):,} apps)")
    except Exception as e:
        print(f"⚠ Apps data not found: {e}")
        DATA['apps'] = None
    
    print("✅ Loading complete!")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_categories():
    """Get unique app categories"""
    if DATA['apps'] is None:
        return []
    cats = DATA['apps']['Category'].dropna().unique()
    return sorted([str(c) for c in cats if str(c) != 'nan'])

def get_content_ratings():
    """Get unique content ratings"""
    if DATA['apps'] is None:
        return []
    if 'Content Rating' in DATA['apps'].columns:
        ratings = DATA['apps']['Content Rating'].dropna().unique()
        return sorted(ratings)
    return []

def predict_rating(app_data):
    """
    Predict rating for an app using improved 50-feature model
    """
    if DATA['model'] is None or DATA['feature_columns'] is None:
        return None
    
    try:
        # Build feature vector for the 50 selected features
        features = []
        
        if isinstance(app_data, pd.Series):
            # App from dataset - extract selected features
            for col in DATA['feature_columns']:
                if col in app_data.index:
                    val = app_data[col]
                    # Handle NaN
                    if pd.isna(val):
                        val = 0.0
                    features.append(float(val))
                else:
                    features.append(0.0)
        else:
            # User input dictionary
            for col in DATA['feature_columns']:
                val = app_data.get(col, 0.0)
                # Handle NaN
                if pd.isna(val):
                    val = 0.0
                features.append(float(val))
        
        # Convert to numpy array
        features = np.array(features, dtype=float).reshape(1, -1)
        
        # Verify we have the right number of features
        if features.shape[1] != len(DATA['feature_columns']):
            print(f"⚠️ Feature count mismatch: expected {len(DATA['feature_columns'])}, got {features.shape[1]}")
            return None
        
        # Make prediction (tree-based ensemble, no scaling needed)
        prediction = float(DATA['model'].predict(features)[0])
        
        # Clip to valid rating range (1.0 to 5.0)
        prediction = round(np.clip(prediction, 1.0, 5.0), 2)
        
        return prediction
    
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def app_to_dict(row):
    """Convert app row to dictionary for JSON"""
    app_dict = {
        'App': str(row.get('App', '')),
        'Category': str(row.get('Category', '')),
        'Rating': float(row['Rating']) if pd.notna(row.get('Rating')) else None,
        'Reviews': int(row['Reviews']) if pd.notna(row.get('Reviews')) else 0,
        'Size_MB': float(row['Size_MB']) if pd.notna(row.get('Size_MB')) else None,
        'Installs': int(row['Installs_Clean']) if pd.notna(row.get('Installs_Clean')) else 0,
        'Type': str(row.get('Type', '')),
        'Price': float(row['Price_Clean']) if pd.notna(row.get('Price_Clean')) else 0.0,
        'Content_Rating': str(row.get('Content Rating', '')),
        'Genres': str(row.get('Genres', '')),
        'Last_Updated': str(row.get('Last Updated', '')) if pd.notna(row.get('Last Updated')) else None,
    }
    
    # Add sentiment features if available
    if 'sentiment_polarity_mean' in row.index:
        app_dict['sentiment_polarity'] = float(row['sentiment_polarity_mean']) if pd.notna(row.get('sentiment_polarity_mean')) else None
    if 'positive_percentage' in row.index:
        app_dict['positive_percentage'] = float(row['positive_percentage']) if pd.notna(row.get('positive_percentage')) else None
    
    app_dict['predicted_rating'] = None
    
    return app_dict

def calculate_similarity(app1_features, app2_features):
    """Calculate cosine similarity"""
    try:
        v1 = np.array(app1_features, dtype=float)
        v2 = np.array(app2_features, dtype=float)
        
        # Replace NaN with 0
        v1 = np.nan_to_num(v1, 0)
        v2 = np.nan_to_num(v2, 0)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            return float(dot / (norm1 * norm2))
        return 0.0
    except:
        return 0.0

# =============================================================================
# ROUTES - PAGES
# =============================================================================

@app.route('/')
def index():
    """Home page"""
    categories = get_categories()
    total_apps = len(DATA['apps']) if DATA['apps'] is not None else 0
    return render_template('index.html', categories=categories, total_apps=total_apps)

@app.route('/search')
def search_page():
    """Search page"""
    categories = get_categories()
    content_ratings = get_content_ratings()
    return render_template('search.html', categories=categories, content_ratings=content_ratings)

@app.route('/app/<path:app_name>')
def app_detail(app_name):
    """App detail page with AI prediction"""
    if DATA['apps'] is None:
        return render_template('error.html', message="Data not loaded"), 404
    
    app = DATA['apps'][DATA['apps']['App'] == app_name]
    if len(app) == 0:
        return render_template('error.html', message="App not found"), 404
    
    app = app.iloc[0]
    app_dict = app_to_dict(app)
    
    # Get AI prediction
    predicted = predict_rating(app)
    if predicted:
        app_dict['predicted_rating'] = predicted
        print(f"✓ Predicted rating for '{app_name}': {predicted}")
    else:
        print(f"⚠️ Could not predict rating for '{app_name}'")
    
    return render_template('app_detail.html', app=app_dict)

#@app.route('/predict')
#def predict_page():
#    """Prediction page"""
#    categories = get_categories()
#    return render_template('predict.html', categories=categories)

# =============================================================================
# ROUTES - API
# =============================================================================

@app.route('/api/featured')
def api_featured():
    """Get featured apps"""
    if DATA['apps'] is None:
        return jsonify([])
    
    df = DATA['apps'].copy()
    df = df[df['Rating'].notna() & (df['Reviews'] > 100)]
    df = df.sort_values(['Rating', 'Reviews'], ascending=[False, False])
    
    featured = df.head(12)
    apps = [app_to_dict(row) for _, row in featured.iterrows()]
    
    return jsonify(apps)

@app.route('/api/search')
def api_search():
    """Search apps"""
    query = request.args.get('q', '').lower()
    
    if not query or len(query) < 2 or DATA['apps'] is None:
        return jsonify([])
    
    df = DATA['apps'].copy()
    mask = df['App'].str.lower().str.contains(query, na=False)
    results = df[mask].head(50)
    
    apps = [app_to_dict(row) for _, row in results.iterrows()]
    return jsonify(apps)

@app.route('/api/filter')
def api_filter():
    """Filter apps"""
    if DATA['apps'] is None:
        return jsonify([])
    
    category = request.args.get('category', '')
    app_type = request.args.get('type', '')
    min_rating = request.args.get('min_rating', type=float)
    sort_by = request.args.get('sort', 'rating')
    limit = request.args.get('limit', 50, type=int)
    
    df = DATA['apps'].copy()
    
    if category:
        df = df[df['Category'] == category]
    if app_type:
        df = df[df['Type'] == app_type]
    if min_rating:
        df = df[df['Rating'] >= min_rating]
    
    df = df[df['Rating'].notna()]
    
    if sort_by == 'rating':
        df = df.sort_values('Rating', ascending=False)
    elif sort_by == 'reviews':
        df = df.sort_values('Reviews', ascending=False)
    elif sort_by == 'installs':
        df = df.sort_values('Installs_Clean', ascending=False)
    
    results = df.head(limit)
    apps = [app_to_dict(row) for _, row in results.iterrows()]
    
    return jsonify(apps)

@app.route('/api/similar/<path:app_name>')
def api_similar(app_name):
    """Get similar apps"""
    if DATA['apps'] is None or DATA['feature_columns'] is None:
        return jsonify([])
    
    limit = request.args.get('limit', 6, type=int)
    
    target_app = DATA['apps'][DATA['apps']['App'] == app_name]
    if len(target_app) == 0:
        return jsonify([])
    
    target_app = target_app.iloc[0]
    target_category = target_app.get('Category')
    
    # Get target features
    target_features = []
    for col in DATA['feature_columns']:
        if col in target_app.index:
            val = target_app[col]
            target_features.append(0.0 if pd.isna(val) else float(val))
        else:
            target_features.append(0.0)
    
    # Find similar apps
    candidates = DATA['apps'][
        (DATA['apps']['Category'] == target_category) &
        (DATA['apps']['App'] != app_name)
    ].copy()
    
    if len(candidates) == 0:
        return jsonify([])
    
    # Calculate similarities
    similarities = []
    for idx, row in candidates.iterrows():
        app_features = []
        for col in DATA['feature_columns']:
            if col in row.index:
                val = row[col]
                app_features.append(0.0 if pd.isna(val) else float(val))
            else:
                app_features.append(0.0)
        sim = calculate_similarity(target_features, app_features)
        similarities.append(sim)
    
    candidates['similarity'] = similarities
    candidates = candidates.sort_values('similarity', ascending=False).head(limit)
    
    apps = []
    for _, row in candidates.iterrows():
        app_dict = app_to_dict(row)
        app_dict['similarity_score'] = round(row['similarity'] * 100, 1)
        apps.append(app_dict)
    
    return jsonify(apps)

@app.route('/api/stats')
def api_stats():
    """Get statistics"""
    if DATA['apps'] is None:
        return jsonify({})
    
    df = DATA['apps']
    
    stats = {
        'total_apps': int(len(df)),
        'total_categories': int(df['Category'].nunique()) if 'Category' in df.columns else 0,
        'avg_rating': round(float(df['Rating'].mean()), 2) if 'Rating' in df.columns else 0,
        'total_reviews': int(df['Reviews'].sum()) if 'Reviews' in df.columns else 0,
        'free_apps': int((df['Type'] == 'Free').sum()) if 'Type' in df.columns else 0,
        'paid_apps': int((df['Type'] == 'Paid').sum()) if 'Type' in df.columns else 0,
        'model_loaded': DATA['model'] is not None,
        'model_r2': '36.34%' if DATA['model'] is not None else 'N/A'
    }
    
    return jsonify(stats)

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', message="Server error"), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    load_data()
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
