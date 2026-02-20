import streamlit as st
import pandas as pd
import joblib

# config
st.set_page_config(page_title="CleanLabels", page_icon="🍏", layout="wide")
st.title("CleanLabels")
st.markdown("Discover healthier food alternatives and predict nutritional scores using Machine Learning.")

@st.cache_data
def load_data():
    return pd.read_csv('data/clean_data.csv')

@st.cache_resource
def load_models():
    preprocessor = joblib.load('models/master_preprocessor.pkl')
    knn = joblib.load('models/swap_engine_knn.pkl')
    rf_pipeline = joblib.load('models/health_score_predictor_rf.pkl')
    unified_matrix = joblib.load('models/unified_matrix.pkl')
    return preprocessor, knn, rf_pipeline, unified_matrix

df = load_data()
preprocessor, knn, rf_pipeline, unified_matrix = load_models()

MACRO_FEATURES = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 
    'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g', 
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g'
]

tab1, tab2 = st.tabs(["Find Healthier Alternatives (KNN)", "Predict Health Score (RF)"])

with tab1:
    st.header("Search & Find Healthier Alternatives to Your Food")
    st.write("Find your current product to see healthier, AI-recommended alternatives.")

    col1, col2 = st.columns(2)
    
    with col1:
        search_product = st.text_input("Product Name (e.g., 'ketchup', 'cereal'):", "")
    with col2:
        search_brand = st.text_input("Brand (Optional, e.g., 'Heinz', 'Kellogg'):", "")
    
    if search_product or search_brand:
        results = df.copy()
        
        if search_product:
            results = results[results['product_name'].str.contains(search_product, case=False, na=False)]
        if search_brand:
            results = results[results['brands'].str.contains(search_brand, case=False, na=False)]
        
        results = results.head(50)
        
        if not results.empty:
            results['display_name'] = results['product_name'] + " (" + results['brands'].fillna('Unknown Brand') + ")"
            selected_display_name = st.selectbox("Select your exact product:", results['display_name'].unique())
            
            selected_row = results[results['display_name'] == selected_display_name].iloc[0]
            target_idx = selected_row.name 
            
            st.subheader(f"Current Stats: {selected_row['display_name']}")
            st.write(f"**Ingredients:** {selected_row['ingredients_text'].title()}") # Capitalized for cleaner UI
            st.write(f"**Nutri-Score:** {selected_row['health_score']} *(Lower is better)*")
            
            if st.button("Find Healthier Alternatives"):
                target_vector = unified_matrix[target_idx].reshape(1, -1)
                distances, indices = knn.kneighbors(target_vector)
                
                neighbor_indices = indices[0][1:] 
                neighbors_df = df.iloc[neighbor_indices].copy()
                
                better_options = neighbors_df[neighbors_df['health_score'] < selected_row['health_score']].head(4)
                
                if better_options.empty:
                    st.warning("This product is already quite healthy compared to its most similar neighbors!")
                else:
                    st.subheader("Side-by-Side Brand Comparison")
                    st.write("Here is your original product compared directly to the healthier recommendations:")

                    original_df = pd.DataFrame([selected_row])
                    original_df['display_name'] = "🛑 " + original_df['display_name'] + " [YOURS]"
                    
                    better_options['display_name'] = better_options['product_name'] + " (" + better_options['brands'].fillna('Unknown Brand') + ")"
                    better_options['display_name'] = "✅ " + better_options['display_name']

                    comparison_df = pd.concat([original_df, better_options])
                    cols_to_show = ['display_name', 'health_score'] + MACRO_FEATURES
                    final_comparison = comparison_df[cols_to_show].set_index('display_name').T

                    st.dataframe(final_comparison, use_container_width=True)
        else:
            st.info("No matching products found. Try adjusting your search terms.")

with tab2:
    st.header("Predict Score for a New Product")
    st.write("Enter the ingredients and macros per 100g to see how an AI model grades a brand new or custom product.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_ingredients = st.text_area("Ingredients (comma separated):", "water, sugar, natural flavors, citric acid")
        calories = st.number_input("Energy (kcal)", value=150.0)
        fat = st.number_input("Fat (g)", value=0.0)
        sat_fat = st.number_input("Saturated Fat (g)", value=0.0)
        trans_fat = st.number_input("Trans Fat (g)", value=0.0)
        cholesterol = st.number_input("Cholesterol (g)", value=0.0)
        
    with col2:
        carbs = st.number_input("Carbohydrates (g)", value=10.0)
        sugars = st.number_input("Sugars (g)", value=9.0)
        fiber = st.number_input("Fiber (g)", value=0.0)
        protein = st.number_input("Proteins (g)", value=0.0)
        sodium = st.number_input("Sodium (g)", value=0.05)
        
    if st.button("Grade My Product"):
        energy = calories * 4.184

        input_data = pd.DataFrame([{
            'ingredients_text': new_ingredients,
            'energy_100g': energy,
            'fat_100g': fat,
            'saturated-fat_100g': sat_fat,
            'trans-fat_100g': trans_fat,
            'cholesterol_100g': cholesterol,
            'carbohydrates_100g': carbs,
            'sugars_100g': sugars,
            'fiber_100g': fiber,
            'proteins_100g': protein,
            'sodium_100g': sodium
        }])
        predicted_score = rf_pipeline.predict(input_data)[0]
        
        st.success(f"AI Predicted Nutri-Score: **{predicted_score:.2f}**")