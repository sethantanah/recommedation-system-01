import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from load_data import load_data


df = load_data("data/model_inventory.csv")

# Combine text-based features into a single column
df["content"] = (
    df["model_type"] + " " + df["framework/library"] + " " + 
    df["input_data"] + " " + df["institution"] + " " + 
    df["action"] + " " + df["license"]
)

df = df.drop(columns=["inference__times","model_size_normalizedmb",  "memory_requirementtraining","memory_requirement_normalized__"])
df = df.dropna()

# Step 3: Feature extraction
# TF-IDF for text-based features
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features = df[["github__stars1000", "citations", "model_sizemb"]]
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Combine TF-IDF and numerical features into a single feature matrix
feature_matrix = hstack([tfidf_matrix, numerical_features_scaled])

# Step 4: Compute similarity
cosine_sim = cosine_similarity(feature_matrix)

# Step 5: Recommendation function
def recommend_items(item_id, top_n=5):
    # Get the index of the item
    idx = df.index[df["id"] == item_id].tolist()[0]

    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort items by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar items (excluding itself)
    sim_scores = sim_scores[1:top_n + 1]

    # Get item indices
    item_indices = [i[0] for i in sim_scores]

    # Return top N similar items
    return df.iloc[item_indices]

# Step 6: Test the recommendation function
# Example: Recommend items similar to item ID 2 (BERT for Named Entity Recognition)
# recommendations = recommend_items(item_id=5, top_n=5)
# # print("Recommendations for BERT (item ID 2):")
# print(recommendations[["id", "model_type", "framework/library", "input_data", "action", "github__stars1000", "citations"]])

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
user_input = input("Describe what you're looking for (e.g., 'A model for sentiment analysis using TensorFlow'): ")
recommendations = recommend_models(user_input, top_n=5)

# Step 6: Display recommendations
print("\nRecommended Models:")
print(recommendations[["id", "model_type", "framework/library", "input_data", "action", "github__stars1000", "citations"]])