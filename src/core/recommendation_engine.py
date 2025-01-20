import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import joblib
import numpy as np

class RecommendationEngine:
    def __init__(self):
        self.df = pd.DataFrame()
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["inference__times","model_size_normalizedmb","memory_requirement_normalized__"])
        df = df.dropna()
        # Convert float columns to strings
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        df["content"] = (
            df["model_type"] + " " + df["framework/library"] + " " + 
            df["input_data"] + " " + df["institution"] + " " + 
            df["action"] + " " + df["license"] + " " + 
            df["github__stars1000"] + " " + df["citations"] + " " +
            df["model_sizemb"] + df["memory_requirementtraining"]
        )
        return df
    
    def fit(self, df: pd.DataFrame):
        self.df = self.preprocess_data(df)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["content"])
        
    def add_model(self, model_data: dict):
        new_df = pd.DataFrame([model_data])
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.fit(self.df)  # Refit the model
        
    def get_recommendations(self, user_input: str, top_n: int = 5) -> List[dict]:
        user_input_vector = self.tfidf.transform([user_input])
        similarity_scores = cosine_similarity(user_input_vector, self.tfidf_matrix)
        top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        
        recommendations = self.df.iloc[top_indices]
        return recommendations.to_dict('records')
    
    def save_model(self, path: str):
        joblib.dump((self.df, self.tfidf, self.tfidf_matrix), path)
        
    def load_model(self, path: str):
        self.df, self.tfidf, self.tfidf_matrix = joblib.load(path)
