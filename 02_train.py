import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('data/clean_data.csv')

TEXT_FEATURE = 'ingredients_text'
MACRO_FEATURES = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 
    'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g', 
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g'
]

X = df[[TEXT_FEATURE] + MACRO_FEATURES]
y = df['health_score']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english', max_features=5000), TEXT_FEATURE), # TF-IDF for word vectorization
        ('macros', MinMaxScaler(), MACRO_FEATURES) # min max scale macros from 0-1 to fit text vectors
    ]
)

# building healthy alternative KNN
knn = NearestNeighbors(n_neighbors=30, metric='cosine', n_jobs=-1)
unified_matrix = preprocessor.fit_transform(X)
knn.fit(unified_matrix)

joblib.dump(preprocessor, 'models/master_preprocessor.pkl')
joblib.dump(knn, 'models/swap_engine_knn.pkl')
joblib.dump(unified_matrix, 'models/unified_matrix.pkl')

# building health score predictor RF
predictor_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictor_pipeline.fit(X_train, y_train)

predictions = predictor_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE: ±{mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

joblib.dump(predictor_pipeline, 'models/health_score_predictor_rf.pkl', compress=3)