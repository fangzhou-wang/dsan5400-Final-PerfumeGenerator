import streamlit as st
import pandas as pd
import re
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import nltk
from logger_config import logger

# 确保 NLTK 数据下载
nltk.download('vader_lexicon')

# 初始化情感分析器
vader_analyzer = SentimentIntensityAnalyzer()

# 加载数据
@st.cache
def load_data():
    perfume_data = pd.read_csv('fra_cleaned.csv', encoding='latin1', delimiter=';')
    review_data = pd.read_csv('extracted_reviews_with_perfume_names.csv')
    return perfume_data, review_data

# 数据预处理
def preprocess_data(perfume_data, review_data):
    # 清理数据
    def clean_text(text):
        if isinstance(text, float):
            text = ""
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower().strip()
        return text

    columns_to_clean = [
        'Perfume', 'Brand', 'Country', 'Gender', 'Top', 'Middle', 'Base',
        'mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5'
    ]
    for col in columns_to_clean:
        if col in perfume_data.columns:
            perfume_data[col] = perfume_data[col].apply(clean_text)

    # 计算情感分数
    def calculate_sentiment_score(text):
        score = vader_analyzer.polarity_scores(text)
        return score['compound']

    review_data['Sentiment Score'] = review_data['Review Text'].apply(calculate_sentiment_score)
    perfume_sentiment_summary = review_data.groupby('Perfume Name')['Sentiment Score'].mean().reset_index()
    perfume_sentiment_summary.rename(columns={'Sentiment Score': 'Average Sentiment Score'}, inplace=True)

    # 合并数据
    perfume_data = pd.merge(perfume_data, perfume_sentiment_summary, left_on='Perfume', right_on='Perfume Name', how='left')
    perfume_data['Average Sentiment Score'] = perfume_data['Average Sentiment Score'].fillna(random.uniform(-1, 1))
    return perfume_data

# GPT 文案生成
def load_gpt_model():
    tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
    return tokenizer, model

def generate_description(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 主应用
def main():
    st.title("Perfume Recommender and Description Generator")
    logger.info("App started.")
    

    # 加载数据
    perfume_data, review_data = load_data()
    perfume_data = preprocess_data(perfume_data, review_data)
    logger.info("Data loaded and preprocessed.")

    # 用户选择主香调
    selected_accords = {}
    st.subheader("Select Main Accords")
    for i in range(1, 6):
        accord = st.text_input(f"Main Accord {i} (optional)", key=f"accord_{i}")
        if accord:
            selected_accords[f"mainaccord{i}"] = accord.lower()
            logger.info(f"User entered Main Accord {i}: {accord.lower()}")

    # 用户选择情感分数阈值
    sentiment_threshold = st.slider("Sentiment Score Threshold", -1.0, 1.0, 0.0)
    logger.info(f"User set sentiment threshold: {sentiment_threshold}")

    # 推荐香水
    if st.button("Recommend Perfumes"):
        logger.info("Recommendation process started.")
        try:
            filter_conditions = True
            for level, accord in selected_accords.items():
                filter_conditions &= (perfume_data[level] == accord)
            filtered_data = perfume_data[filter_conditions]
            filtered_data = filtered_data[filtered_data['Average Sentiment Score'] >= sentiment_threshold]

            if filtered_data.empty:
                st.warning("No perfumes found matching the specified criteria.")
                logger.warning("No perfumes found matching the criteria.")
            else:
                st.success("Recommended Perfumes:")
                st.table(filtered_data[['Perfume', 'Brand', 'Average Sentiment Score']])
                logger.info(f"Recommended {len(filtered_data)} perfumes.")

                # Generate Descriptions
                tokenizer, model = load_gpt_model()
                st.subheader("Generated Descriptions")
                logger.info("Started generating descriptions.")

                for _, row in filtered_data.iterrows():
                    prompt = f"This perfume, {row['Perfume']} by {row['Brand']}, is known for its"
                    description = generate_description(prompt, tokenizer, model)
                    st.write(f"**{row['Perfume']} by {row['Brand']}**")
                    st.write(f"*Description:* {description}")
                    st.write("---")
                    logger.info(f"Generated description for {row['Perfume']} by {row['Brand']}.")

        except Exception as e:
            logger.error(f"Error during recommendation process: {e}")
            st.error("An error occurred while processing your request. Please try again.")

if __name__ == "__main__":
    main()
