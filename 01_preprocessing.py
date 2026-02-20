import pandas as pd

fda_standard_label = [
    'energy_100g',
    'fat_100g', 
    'saturated-fat_100g', 
    'trans-fat_100g', 
    'cholesterol_100g', 
    'carbohydrates_100g', 
    'sugars_100g', 
    'fiber_100g', 
    'proteins_100g', 
    'sodium_100g'
]

keep_cols = [
    'code', 'product_name', 'brands', 'countries', 'ingredients_text',
    'nutrition-score-fr_100g'
] + fda_standard_label

df = pd.read_csv('data/raw_data.tsv', sep='\t', usecols=lambda c: c in keep_cols, low_memory=False)

df = df.loc[df['countries'] == 'US'] # keeping US-only data
df = df.dropna(subset=['product_name', 'ingredients_text', 'nutrition-score-fr_100g','energy_100g', 'carbohydrates_100g', 'fat_100g', 'proteins_100g']) # keeping core information

safe_zeros = [
    'trans-fat_100g', 
    'cholesterol_100g', 
    'sugars_100g', 
    'fiber_100g', 
    'saturated-fat_100g', 
    'sodium_100g'
]
df[safe_zeros] = df[safe_zeros].fillna(0.0) # filling NaN with zeroes

df['health_score'] = df['nutrition-score-fr_100g']
df['ingredients_text'] = df['ingredients_text'].str.lower()

df.to_csv('data/clean_data.csv', index=False)