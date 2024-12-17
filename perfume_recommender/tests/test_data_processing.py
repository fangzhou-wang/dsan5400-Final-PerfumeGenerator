import pandas as pd
from perfume_recommender.data_processing.data_processing import DataProcessor


def test_data_processing_preprocess_data():
    processor = DataProcessor(
        "tests/sample_perfume_data.csv", "tests/sample_review_data.csv"
    )
    df = processor.preprocess_data()

    assert not df.empty, "Processed data should not be empty"
    assert (
        "Average Sentiment Score" in df.columns
    ), "Processed data should contain sentiment scores"


def test_clean_text():
    cleaned_text = DataProcessor.clean_text("This is TEST text! 123")
    assert (
        cleaned_text == "this is test text"
    ), "Text cleaning should remove special characters and numbers"
