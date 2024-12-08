# Import necessary libraries
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from logger_config import logger

# Ensure NLTK VADER is downloaded
nltk.download('vader_lexicon')

# Perfume Recommender Class
class PerfumeRecommender:
    def __init__(self, perfume_data_path, review_data_path):
        self.df = pd.read_csv(perfume_data_path, encoding='latin1', delimiter=';')
        self.reviews_df = pd.read_csv(review_data_path)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.columns_to_clean = [
            'Perfume', 'Brand', 'Country', 'Gender', 'Top', 'Middle', 'Base',
            'mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5'
        ]

    def clean_text(self, text):
        if isinstance(text, float):
            text = ""
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower().strip()
        return text

    def preprocess(self):
        print("Columns in dataset:", self.df.columns.tolist())
        for col in self.columns_to_clean:
            if col in self.df.columns:
                print(f"Cleaning column: {col}")
                self.df[col] = self.df[col].apply(self.clean_text)
            else:
                print(f"Warning: Column '{col}' not found in the dataset. Skipping.")

    def preprocess_reviews(self):
        def calculate_sentiment_score(text):
            score = self.vader_analyzer.polarity_scores(text)
            return score['compound']

        self.reviews_df['Sentiment Score'] = self.reviews_df['Review Text'].apply(calculate_sentiment_score)
        perfume_sentiment_summary = self.reviews_df.groupby('Perfume Name')['Sentiment Score'].mean().reset_index()
        perfume_sentiment_summary.rename(columns={'Sentiment Score': 'Average Sentiment Score'}, inplace=True)
        self.df = pd.merge(self.df, perfume_sentiment_summary, left_on='Perfume', right_on='Perfume Name', how='left')
        self.df.drop(columns=['Perfume Name'], inplace=True, errors='ignore')

    def assign_random_sentiment_scores(self):
        """Assign random scores for perfumes without reviews."""
        self.df['Average Sentiment Score'] = self.df['Average Sentiment Score'].apply(
            lambda x: x if not pd.isna(x) else random.uniform(-1, 1)
        )

    def build_main_accord_trees(self):
        accord_columns = ['mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5']
        for i in range(len(accord_columns) - 1):
            input_col = accord_columns[:i+1]
            target_col = accord_columns[i+1]

            df_subset = self.df.dropna(subset=input_col + [target_col])
            X = pd.get_dummies(df_subset[input_col])
            y = df_subset[target_col]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            tree = DecisionTreeClassifier(random_state=42)
            tree.fit(X_train, y_train)

            print(f"Decision Tree for {target_col} based on {', '.join(input_col)}")
            print(export_text(tree, feature_names=X.columns.tolist()))

    def recommend_perfumes(self, selected_main_accords, sentiment_threshold=0.0):
        logger.info(f"Recommendation started with main accords: {selected_main_accords} and sentiment threshold: {sentiment_threshold}")
        filter_conditions = True
        for level, accord in selected_main_accords.items():
            filter_conditions &= (self.df[level] == accord)
        filtered_rows = self.df[filter_conditions]
        filtered_rows = filtered_rows[filtered_rows['Average Sentiment Score'] >= sentiment_threshold]
        if filtered_rows.empty:
            logger.warning("No perfumes found matching the specified criteria.")
            return None
        logger.info(f"Found {len(filtered_rows)} matching perfumes.")
        return filtered_rows[['Perfume', 'Brand', 'Average Sentiment Score']]


# Fine-tuned GPT Model Loader
def load_fine_tuned_gpt(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    logger.info("Fine-tuned GPT-2 model loaded successfully.")
    return tokenizer, model


# Generate personalized descriptions using GPT
def generate_custom_description(prompt, tokenizer, model, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.5,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Main Script
if __name__ == "__main__":
    # Initialize Recommender
    recommender = PerfumeRecommender('fra_cleaned.csv', 'extracted_reviews_with_perfume_names.csv')
    recommender.preprocess()
    recommender.preprocess_reviews()
    recommender.assign_random_sentiment_scores()
    recommender.build_main_accord_trees()

    # User-selected criteria
    selected_accords = {}
    while True:
        print("Available main accords: Citrus, Woody, Floral, etc.")
        user_choice = input("Choose a main accord or type 'complete' to finish: ").strip()
        if user_choice.lower() == 'complete':
            break
        selected_accords[f'mainaccord{len(selected_accords) + 1}'] = user_choice

    # sentiment_threshold = float(input("\nEnter a sentiment score threshold (e.g., 0.5 for positive perfumes only): "))
    sentiment_threshold = 0.1

    # Recommend perfumes
    recommendations = recommender.recommend_perfumes(selected_accords, sentiment_threshold)
    if recommendations is None:
        print("No perfumes found matching the specified criteria.")
    else:
        print("\nRecommended Perfumes:")
        print(recommendations)

        # Load fine-tuned GPT model
        tokenizer, model = load_fine_tuned_gpt("./fine_tuned_gpt2")

        # Generate personalized descriptions
        print("\nGenerated Descriptions:")
        for _, row in recommendations.iterrows():
            prompt = f"This perfume, {row['Perfume']} by {row['Brand']}, is known for its"
            description = generate_custom_description(prompt, tokenizer, model)
            print(f"Perfume: {row['Perfume']}, Brand: {row['Brand']}")
            print(f"Description: {description}\n")
