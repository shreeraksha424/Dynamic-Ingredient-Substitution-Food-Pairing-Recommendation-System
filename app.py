from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

app = Flask(__name__)

# Load and preprocess dataset
def load_and_preprocess_data():
    try:
        # Load the dataset (replace with actual path to recipe_final (1).csv)
        recipe_df = pd.read_csv('recipe_final (1).csv')
        
        # TF-IDF for ingredients
        vectorizer = TfidfVectorizer()
        X_ingredients = vectorizer.fit_transform(recipe_df['ingredients_list'])
        
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']
        X_numerical = scaler.fit_transform(recipe_df[numerical_cols])
        
        # Combine features
        X_combined = np.hstack([X_numerical, X_ingredients.toarray()])
        
        # Train KNN model
        knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        knn.fit(X_combined)
        
        # Train Random Forest for nutritional classification (e.g., low/medium/high calories)
        # Create a simple classification target (e.g., calorie bins)
        calorie_bins = pd.cut(recipe_df['calories'], bins=[0, 200, 400, float('inf')], labels=['low', 'medium', 'high'])
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_numerical, calorie_bins)
        
        return recipe_df, vectorizer, scaler, knn, rf, numerical_cols
    except FileNotFoundError:
        raise Exception("Dataset 'recipe_final (1).csv' not found. Please ensure it is in the correct directory.")

# Load Indian food dataset for filtering (optional)
def load_indian_food_data():
    try:
        indian_df = pd.read_csv('filtered indian food.csv')
        return indian_df
    except FileNotFoundError:
        return None

# Global variables
recipe_df, vectorizer, scaler, knn, rf, numerical_cols = load_and_preprocess_data()
indian_df = load_indian_food_data()

# Function to find ingredient substitutions using Cosine Similarity
def find_ingredient_substitutions(input_ingredients, top_n=3):
    input_vector = vectorizer.transform([input_ingredients]).toarray()
    ingredient_vectors = vectorizer.transform(recipe_df['ingredients_list']).toarray()
    similarities = cosine_similarity(input_vector, ingredient_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    substitutions = []
    for idx in top_indices:
        substitutions.append({
            'recipe_name': recipe_df.iloc[idx]['recipe_name'],
            'ingredients': recipe_df.iloc[idx]['ingredients_list'],
            'similarity_score': similarities[idx]
        })
    return substitutions

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input provided'}), 400
        
        # Expected input: nutritional values and ingredients
        calories = data.get('calories', 0)
        fat = data.get('fat', 0)
        carbohydrates = data.get('carbohydrates', 0)
        protein = data.get('protein', 0)
        cholesterol = data.get('cholesterol', 0)
        sodium = data.get('sodium', 0)
        fiber = data.get('fiber', 0)
        ingredients = data.get('ingredients', '')
        diet_filter = data.get('diet', None)  # Optional: vegetarian, non-vegetarian
        course_filter = data.get('course', None)  # Optional: dessert, main course, etc.
        include_substitutions = data.get('include_substitutions', False)  # Optional: include ingredient substitutions
        
        # Prepare input features
        input_numerical = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber]
        input_features_scaled = scaler.transform([input_numerical])
        input_ingredients_transformed = vectorizer.transform([ingredients]).toarray()
        input_combined = np.hstack([input_features_scaled, input_ingredients_transformed])
        
        # Get KNN recommendations
        distances, indices = knn.kneighbors(input_combined)
        recommendations = recipe_df.iloc[indices[0]][['recipe_name', 'ingredients_list', 'image_url']].to_dict('records')
        
        # Filter by Indian food dataset (if available and requested)
        if indian_df is not None and (diet_filter or course_filter):
            filtered_recommendations = []
            for rec in recommendations:
                match = indian_df[indian_df['name'] == rec['recipe_name']]
                if not match.empty:
                    if (diet_filter and match['diet'].iloc[0] != diet_filter) or \
                       (course_filter and match['course'].iloc[0] != course_filter):
                        continue
                    filtered_recommendations.append(rec)
            recommendations = filtered_recommendations if filtered_recommendations else recommendations
        
        # Predict nutritional category with Random Forest
        nutritional_category = rf.predict([input_numerical])[0]
        
        # Get ingredient substitutions (if requested)
        substitutions = []
        if include_substitutions:
            substitutions = find_ingredient_substitutions(ingredients)
        
        # Prepare response
        response = {
            'recommendations': recommendations,
            'nutritional_category': nutritional_category,
            'substitutions': substitutions if include_substitutions else []
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)