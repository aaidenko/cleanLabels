# cleanLabels 🍏
Built for SunnyHacks Feb 2026

[![Streamlit App]](https://cleanlabels.streamlit.app/)

## The Problem
Navigating nutrition labels is confusing, and finding genuinely healthier alternatives that actually share the same ingredients and taste profile is incredibly difficult. 

## Our Solution
cleanLabels is a fully deployed machine learning application that analyzes the ingredients and macronutrients of over 130,000 US food products from the OpenFoodFacts database. 
It translates complex nutritional data into an easy-to-understand Nutri-Score grade (used commonly in Europe) and uses an advanced recommendation engine to suggest healthier, highly similar alternatives.
It can also predict a Nutri-Score grade for a novel food item that the user enters ingredients and nutritional information for.

## Key Features
- Product Lookup: Search for a specific product and/or its brand to find its nutritional profile.
- Healthier Recommendations: Select a minimum target health grade (e.g., "Show me alternatives that are a B or better") and an ML model finds the top 3 closest matches.
- Custom Recipe Predictor: If you want to know how healthy/unhealthy a new product may be, enter your raw ingredients and macros to predict how the algorithm will grade it.

## Under the Hood (ML Compononent)
We use a multimodal ML pipeline to train our models for this project:
1. NLP (TF-IDF Vectorization): We vectorize raw ingredient lists to understand the actual composition of the food, giving unique ingredients proper mathematical weight.
2. Feature Scaling (MinMaxScaler): We scale standard FDA macronutrients (fat, sugar, sodium, etc.) and unify them with the text vectors.
3. Recommendation Engine (KNN): We use K-Nearest Neighbors with cosine similarity in a 5,000 dimensional space to find foods that are mathematically aligned in both ingredients and macros.
4. Predictive Modeling (Random Forest): A Random Forest Regressor predicts the precise Nutri-Score of novel, unseen ingredient combinations. Our evaluation data reported a MAE of ±0.27, RMSE of 0.56, and R2 of 0.9961, demonstrating high accuracy and deployability in novel contexts.

## Tech Stack
- Frontend: Streamlit
- Machine Learning: Scikit-Learn (KNN, Random Forest, TF-IDF)
- Data Processing: Pandas
- Dataset: Open Food Facts (Kaggle) - Cleaned and filtered for US-only products.
