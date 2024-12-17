import pandas as pd
import re
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download("vader_lexicon", quiet=True)
vader_analyzer = SentimentIntensityAnalyzer()

class DataProcessor:
    def __init__(self, perfume_data_path, review_data_path):
        self.perfume_data_path = perfume_data_path
        self.review_data_path = review_data_path
        self.perfume_data = None
        self.review_data = None

    @staticmethod
    def clean_text(text):
        if isinstance(text, float):
            text = ""
        text = re.sub(r"\\W", " ", text)
        text = re.sub(r"\\d+", "", text)
        text = text.lower().strip()
        return text

    def calculate_sentiment(self, text):
        score = vader_analyzer.polarity_scores(text)["compound"]
        return score

    def preprocess_data(self):
        logger.info("Starting data preprocessing...")

        # Load Data
        try:
            logger.info(f"Loading perfume data from {self.perfume_data_path}")
            self.perfume_data = pd.read_csv(self.perfume_data_path, encoding="latin1", delimiter=";")
            logger.info("Perfume data loaded successfully.")

            logger.info(f"Loading review data from {self.review_data_path}")
            self.review_data = pd.read_csv(self.review_data_path)
            logger.info("Review data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

        # Clean text columns
        logger.info("Cleaning text columns in perfume data...")
        columns_to_clean = [
            "Perfume", "Brand", "Country", "Gender", "Top", "Middle", "Base",
            "mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5",
        ]
        for col in columns_to_clean:
            if col in self.perfume_data.columns:
                self.perfume_data[col] = self.perfume_data[col].apply(self.clean_text)

        logger.info("Text cleaning complete.")

        # Calculate Sentiment Scores
        logger.info("Calculating sentiment scores for reviews...")
        try:
            self.review_data["Sentiment Score"] = self.review_data["Review Text"].apply(self.calculate_sentiment)
            logger.info("Sentiment scores calculated successfully.")
        except Exception as e:
            logger.error(f"Error during sentiment score calculation: {e}")
            raise

        # Summarize Sentiment Scores
        logger.info("Summarizing sentiment scores by perfume name...")
        summary = self.review_data.groupby("Perfume Name")["Sentiment Score"].mean().reset_index()
        summary.rename(columns={"Sentiment Score": "Average Sentiment Score"}, inplace=True)

        # Merge sentiment scores into perfume data
        logger.info("Merging sentiment scores into perfume data...")
        try:
            self.perfume_data = pd.merge(
                self.perfume_data, summary, left_on="Perfume", right_on="Perfume Name", how="left"
            ).drop(columns="Perfume Name", errors="ignore")
            self.perfume_data["Average Sentiment Score"] = self.perfume_data["Average Sentiment Score"].fillna(0)
            logger.info("Data merging complete.")
        except Exception as e:
            logger.error(f"Error during data merging: {e}")
            raise

        logger.info("Data preprocessing completed successfully.")
        return self.perfume_data