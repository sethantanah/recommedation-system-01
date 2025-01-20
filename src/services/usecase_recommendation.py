import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from load_data import load_data


df = load_data("data/model_inventory.csv")

# Convert float columns to strings
for col in df.columns:
    if pd.api.types.is_float_dtype(df[col]):
        df[col] = df[col].astype(str)

# Combine text-based features into a single column
df["content"] = (
    df["model_type"] + " " + df["framework/library"] + " " + 
    df["input_data"] + " " + df["institution"] + " " + 
    df["action"] + " " + df["license"] + " " + 
    df["github__stars1000"] + " " + df["citations"] + " " +
    df["model_sizemb"] + df["memory_requirementtraining"]
)

df = df.drop(columns=["inference__times","model_size_normalizedmb","memory_requirement_normalized__"])
df = df.dropna()

# Step 3: Feature extraction
# TF-IDF for text-based features
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])


# Step 4: Recommendation function based on user input
def recommend_models(user_input, top_n=5):
    # Transform user input into TF-IDF vector
    user_input_vector = tfidf.transform([user_input])

    # Compute cosine similarity between user input and all models
    similarity_scores = cosine_similarity(user_input_vector, tfidf_matrix)

    # Get top N similar models
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]

    # Return top N similar models
    return df.iloc[top_indices]


# Step 5: Get user input and recommend models
user_input = "A model for sentiment analysis using TensorFlow"
recommendations = recommend_models(user_input, top_n=5)

# Step 6: Display recommendations
print("\nRecommended Models:")
print(recommendations[["id", "model_type", "framework/library", "input_data", "action", "github__stars1000", "citations"]])