import streamlit as st
import logging
from data_processing import DataProcessor
from perfume_recommendation import PerfumeRecommender
from text_generation import TextGenerator

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Streamlit web application for perfume recommendation and description generation.

    - Users select main accords, sentiment thresholds, and optional inputs.
    - Perfume recommendations are displayed based on user preferences.
    - GPT-2 generates descriptions for the recommended perfumes.
    """
    logger.info("Starting Perfume Recommender and Description Generator app...")
    st.title("Perfume Recommender and Description Generator")

    # Data Processing
    try:
        logger.info("Initializing data processor...")
        processor = DataProcessor("fra_cleaned.csv", "extracted_reviews_with_perfume_names.csv")
        data = processor.preprocess_data()
        logger.info("Data successfully loaded and preprocessed.")
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        st.error("Failed to load and preprocess data.")
        return

    # Initialize Modules
    recommender = PerfumeRecommender(data)
    generator = TextGenerator()

    # User Input for Main Accords
    selected_accords = {}
    st.subheader("Select Main Accords")
    logger.info("Waiting for user input for main accords...")
    for i in range(1, 6):
        """
        Allow the user to input up to 5 main accords.
        
        Args:
            i (int): The main accord level (1-5).

        Returns:
            dict: User-selected main accords.
        """
        accord = st.text_input(f"Main Accord {i} (optional)", key=f"accord_{i}")
        if accord:
            selected_accords[f"mainaccord{i}"] = accord.lower()
            logger.info(f"User selected Main Accord {i}: {accord.lower()}")

    # User Input for Sentiment Threshold
    sentiment_threshold = st.slider("Sentiment Score Threshold", -1.0, 1.0, 0.0)
    logger.info(f"User selected sentiment threshold: {sentiment_threshold}")

    # Recommendation and Description Generation
    if st.button("Recommend Perfumes"):
        """
        Handle the perfume recommendation process and display generated descriptions.

        Steps:
            - Filter perfumes based on selected accords and sentiment threshold.
            - Generate and display descriptions for recommended perfumes.
        """
        logger.info("Recommendation process started...")
        try:
            # Perfume Recommendation
            perfumes = recommender.recommend_perfumes(selected_accords, sentiment_threshold=sentiment_threshold)
            if isinstance(perfumes, str) or perfumes.empty:
                logger.warning("No perfumes found for the selected criteria.")
                st.warning("No perfumes found matching the specified criteria.")
            else:
                st.success("Recommended Perfumes:")
                st.table(perfumes)
                logger.info(f"Recommended {len(perfumes)} perfumes.")

                # Text Generation
                st.subheader("Generated Descriptions")
                logger.info("Starting text generation for recommended perfumes...")
                for _, perfume in perfumes.iterrows():
                    prompt = f"This perfume, {perfume['Perfume']} by {perfume['Brand']}, is known for its"
                    description = generator.generate_description(prompt)
                    st.write(f"**{perfume['Perfume']} by {perfume['Brand']}**")
                    st.write(f"*Description:* {description}")
                    st.write("---")
                    logger.info(f"Generated description for {perfume['Perfume']} by {perfume['Brand']}.")
        except Exception as e:
            logger.error(f"Error during recommendation or text generation: {e}")
            st.error("An error occurred while generating recommendations or descriptions. Please try again.")

if __name__ == "__main__":
    main()