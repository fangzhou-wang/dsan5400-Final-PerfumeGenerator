# import pandas as pd
# import random
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.tree import DecisionTreeClassifier

# class PerfumeRecommenderWithCopywriting:
#     def __init__(self, perfume_data_path, review_data_path, description_data_path):
#         self.perfume_data_path = perfume_data_path
#         self.review_data_path = review_data_path
#         self.description_data_path = description_data_path
#         self.perfume_data = None
#         self.review_data = None
#         self.description_corpus = None
#         self.ngram_model = None
#         self.model = None

#     def preprocess(self):
#         """Load and preprocess perfume data."""
#         self.perfume_data = pd.read_csv(self.perfume_data_path)
#         self.perfume_data.fillna('', inplace=True)

#     def preprocess_reviews(self):
#         """Load and preprocess review data."""
#         self.review_data = pd.read_csv(self.review_data_path)
#         self.review_data.fillna('', inplace=True)

#     def assign_random_sentiment_scores(self):
#         """Assign random sentiment scores to perfumes."""
#         random.seed(42)
#         self.perfume_data['Average Sentiment Score'] = [random.uniform(0, 1) for _ in range(len(self.perfume_data))]

#     def train_decision_tree(self):
#         """Train a decision tree model for perfume classification based on main accords."""
#         main_accords = [f'mainaccord{i}' for i in range(1, 6)]
#         feature_cols = main_accords
#         self.perfume_data['label'] = self.perfume_data[main_accords].apply(lambda row: ' | '.join(row.values), axis=1)

#         # Prepare the data for training
#         X = self.perfume_data[feature_cols]
#         y = self.perfume_data['label']

#         # Train the decision tree
#         self.model = DecisionTreeClassifier(random_state=42)
#         self.model.fit(X, y)

#     def recommend_perfumes(self, selected_main_accords, sentiment_threshold):
#         """Recommend perfumes based on user preferences."""
#         filtered_perfumes = self.perfume_data.copy()

#         # Filter based on main accords
#         for accord, value in selected_main_accords.items():
#             filtered_perfumes = filtered_perfumes[filtered_perfumes[accord] == value]

#         # Filter based on sentiment score
#         filtered_perfumes = filtered_perfumes[filtered_perfumes['Average Sentiment Score'] >= sentiment_threshold]

#         # Check if there are recommendations
#         if filtered_perfumes.empty:
#             return "No perfumes found matching the criteria."

#         return filtered_perfumes

#     def train_ngram_model(self, n=2):
#         """Train an n-grams model on perfume descriptions."""
#         description_df = pd.read_csv(self.description_data_path)
#         self.description_corpus = description_df['description'].dropna().tolist()

#         vectorizer = CountVectorizer(ngram_range=(n, n))
#         vectorizer.fit_transform(self.description_corpus)
#         self.ngram_model = vectorizer

#     def generate_copywriting(self, perfume_row):
#         """Generate personalized copywriting for a perfume."""
#         if self.ngram_model is None:
#             raise ValueError("Please train the n-grams model first using train_ngram_model().")

#         perfume_name = perfume_row['Perfume']
#         brand = perfume_row['Brand']
#         main_accords = [perfume_row[f'mainaccord{i}'] for i in range(1, 6) if perfume_row[f'mainaccord{i}']]
#         sentiment_score = perfume_row['Average Sentiment Score']

#         random_description = random.choice(self.description_corpus)
#         copywriting = (
#             f"{perfume_name} by {brand} is a delightful creation with a unique combination of {', '.join(main_accords)} accords. "
#             f"With an average sentiment score of {sentiment_score:.2f}, this perfume evokes {random_description[:100]}..."
#         )
#         return copywriting

# # Main program
# if __name__ == "__main__":
#     recommender = PerfumeRecommenderWithCopywriting(
#         'fra_cleaned.csv', 'extracted_reviews_with_perfume_names.csv', 'final_perfume_data.csv'
#     )

#     # Data preprocessing
#     recommender.preprocess()
#     recommender.preprocess_reviews()
#     recommender.assign_random_sentiment_scores()

#     # Train decision tree model
#     recommender.train_decision_tree()

#     # Train n-grams model
#     recommender.train_ngram_model(n=2)

#     # User interaction for recommendations
#     selected_accords = {}
#     print("Available main accords: Citrus, Woody, Floral, etc.")
#     while True:
#         user_choice = input("Choose a main accord or type 'complete' to finish: ").strip()
#         if user_choice.lower() == 'complete':
#             break
#         selected_accords[f'mainaccord{len(selected_accords) + 1}'] = user_choice

#     # sentiment_threshold = float(input("Enter the minimum sentiment score (e.g., 0.6): "))
#     sentiment_threshold = 0.01

#     recommendations = recommender.recommend_perfumes(selected_accords, sentiment_threshold)
#     if isinstance(recommendations, str):
#         print(recommendations)
#     else:
#         print("\nRecommended Perfumes:")
#         print(recommendations)

#         # Generate copywriting for recommendations
#         for _, row in recommendations.iterrows():
#             print("\nGenerated Copywriting:")
#             print(recommender.generate_copywriting(row))






























# 未加入BERT

# import pandas as pd
# from gensim.models import KeyedVectors
# import gensim.downloader as api
# import random
# from collections import defaultdict

# # Step 1: Load CSV Data
# file_path = 'final_perfume_data.csv'
# data = pd.read_csv(file_path)

# # Preprocess the descriptions
# descriptions = data['Description'].dropna().tolist()  # Get non-empty descriptions

# # Step 2: Load GloVe Embedding Model
# print("Loading GloVe model...")
# embedding_model = api.load('glove-wiki-gigaword-300')  # Automatically download and load GloVe model
# print("GloVe model loaded!")

# # Step 3: Define N-grams Functions
# def generate_ngrams(text, n):
#     tokens = text.split()  # Tokenize the text
#     ngrams = [(tuple(tokens[i:i+n-1]), tokens[i+n-1]) for i in range(len(tokens)-n+1)]
#     return ngrams

# def build_ngram_model(corpus, n):
#     model = defaultdict(list)
#     for text in corpus:
#         ngrams = generate_ngrams(text, n)
#         for context, next_word in ngrams:
#             model[context].append(next_word)
#     return model

# # Build an N-gram model
# n = 3
# ngram_model = build_ngram_model(descriptions, n)

# # Step 4: Define Weighted Sampling with Word Embedding
# def weighted_choice_with_embeddings(word_counts, context, embedding_model):
#     """
#     使用词频加权和词向量相似性选择下一个单词。
#     :param word_counts: [(word, count), ...]
#     :param context: 当前上下文（tuple，前 n-1 个词）
#     :param embedding_model: 词嵌入模型
#     :return: 最优的下一个单词
#     """
#     total = sum(count for _, count in word_counts)  # 总权重
#     best_word = None
#     best_score = float('-inf')  # 初始化最优分数
#     context_word = context[-1]  # 上下文的最后一个词

#     for word, count in word_counts:
#         try:
#             # 计算上下文词和候选词的语义相似性
#             similarity = embedding_model.similarity(context_word, word)
#             # 综合词频权重和语义相似性
#             score = count / total + similarity  # 权重 + 相似性
#             if score > best_score:
#                 best_score = score
#                 best_word = word
#         except KeyError:
#             # 如果词不在嵌入模型中，则跳过
#             continue

#     # 如果没有找到合适的词，随机选择一个词
#     if not best_word:
#         return random.choice([word for word, _ in word_counts])
#     return best_word

# # Step 5: Generate Text with Embedding Assistance
# def generate_text_with_embeddings(model, n, embedding_model, max_words=50):
#     context = random.choice(list(model.keys()))  # 随机选择起始上下文
#     result = list(context)
    
#     for _ in range(max_words - (n-1)):
#         possible_next_words = model.get(context, None)
#         if not possible_next_words:  # 如果没有候选单词，停止生成
#             break
#         # 计算词频
#         word_counts = [(word, possible_next_words.count(word)) for word in set(possible_next_words)]
#         # 使用加权采样结合词向量选择下一个单词
#         next_word = weighted_choice_with_embeddings(word_counts, context, embedding_model)
#         result.append(next_word)
#         context = tuple(result[-(n-1):])  # 更新上下文
    
#     return ' '.join(result)

# # Generate a sample text
# generated_text_embeddings = generate_text_with_embeddings(ngram_model, n, embedding_model)
# print("Generated Text with Word Embeddings:")
# print(generated_text_embeddings)


























# 加入bert

# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# import torch
# import random
# from collections import defaultdict

# # Step 1: Load CSV Data
# file_path = 'final_perfume_data.csv'
# data = pd.read_csv(file_path)

# # Preprocess the descriptions
# descriptions = data['Description'].dropna().tolist()  # Get non-empty descriptions

# # Step 2: Load BERT Model and Tokenizer
# print("Loading BERT model...")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")
# print("BERT model loaded!")

# # Step 3: Define N-grams Functions
# def generate_ngrams(text, n):
#     tokens = text.split()  # Tokenize the text
#     if len(tokens) < n:  # Ensure the text is long enough
#         return []
#     ngrams = [(tuple(tokens[i:i+n-1]), tokens[i+n-1]) for i in range(len(tokens)-n+1)]
#     return ngrams

# def build_ngram_model(corpus, n):
#     model = defaultdict(list)
#     for text in corpus:
#         ngrams = generate_ngrams(text, n)
#         for context, next_word in ngrams:
#             model[context].append(next_word)
#     return model

# # Build an N-gram model
# n = 3
# ngram_model = build_ngram_model(descriptions, n)

# # Step 4: Define Weighted Sampling with BERT Embeddings
# def bert_similarity(context, candidate, tokenizer, model):
#     """
#     使用 BERT 计算上下文与候选词的相似性。
#     """
#     if not candidate.strip():
#         return 0.0  # Skip empty candidates
    
#     context_text = " ".join(context)
#     inputs = tokenizer(context_text, candidate, return_tensors="pt", padding=True, truncation=True)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Use [CLS] token for similarity calculation
#     embeddings = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embeddings
#     if embeddings.size(0) < 2:  # Ensure valid tensor shape
#         return 0.0

#     context_embedding = embeddings[0]
#     candidate_embedding = embeddings[1]
#     similarity = torch.nn.functional.cosine_similarity(context_embedding, candidate_embedding, dim=0)
#     return similarity.item()

# def weighted_choice_with_bert(word_counts, context, tokenizer, model):
#     """
#     使用词频加权和 BERT 语义相似性选择下一个单词。
#     """
#     total = sum(count for _, count in word_counts)  # 总权重
#     best_word = None
#     best_score = float('-inf')  # 初始化最优分数

#     for word, count in word_counts:
#         try:
#             similarity = bert_similarity(context, word, tokenizer, model)
#             score = 0.7 * (count / total) + 0.3 * similarity  # Adjust weights for frequency and similarity
#             if score > best_score:
#                 best_score = score
#                 best_word = word
#         except Exception as e:
#             print(f"Error processing word '{word}': {e}")
#             continue

#     if not best_word:
#         return random.choice([word for word, _ in word_counts])  # Random fallback
#     return best_word

# # Step 5: Generate Text with BERT Assistance
# def generate_text_with_bert(model, n, tokenizer, bert_model, max_words=50):
#     context = random.choice(list(model.keys()))  # Randomly choose a starting context
#     result = list(context)

#     for _ in range(max_words - (n-1)):
#         possible_next_words = model.get(context, None)
#         if not possible_next_words:
#             break
#         word_counts = [(word, possible_next_words.count(word)) for word in set(possible_next_words)]
#         next_word = weighted_choice_with_bert(word_counts, context, tokenizer, bert_model)
#         result.append(next_word)
#         context = tuple(result[-(n-1):])  # Update context

#     return ' '.join(result)

# # Generate a sample text
# generated_text_bert = generate_text_with_bert(ngram_model, n, tokenizer, model)
# print("Generated Text with BERT Embeddings:")
# print(generated_text_bert)




















# # gpt模型
# import pandas as pd

# # Step 1: Load CSV Data
# file_path = 'final_perfume_data.csv'  # Replace with your file path
# data = pd.read_csv(file_path)

# # Step 2: Extract descriptions and save to a text file
# descriptions = data['Description'].dropna().tolist()  # Get non-empty descriptions
# output_file = 'perfume_descriptions.txt'

# with open(output_file, 'w') as f:
#     for description in descriptions:
#         f.write(description.strip() + "\n")

# print(f"Descriptions saved to {output_file}")


# from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# # Step 1: Load pre-trained GPT-2 model and tokenizer
# print("Loading GPT-2 model and tokenizer...")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# print("GPT-2 model loaded!")

# # Step 2: Load dataset
# def load_dataset(file_path, block_size=128):
#     return TextDataset(
#         tokenizer=tokenizer,
#         file_path=file_path,
#         block_size=block_size
#     )

# train_dataset = load_dataset("perfume_descriptions.txt")
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Step 3: Define training arguments
# training_args = TrainingArguments(
#     output_dir="./fine_tuned_gpt2",
#     overwrite_output_dir=True,
#     num_train_epochs=5,  # You can increase this for better results
#     per_device_train_batch_size=4,
#     save_steps=500,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=100,
# )

# # Step 4: Train the model
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
# )

# print("Starting fine-tuning...")
# trainer.train()

# # Save the fine-tuned model and tokenizer
# trainer.save_model("./fine_tuned_gpt2")
# tokenizer.save_pretrained("./fine_tuned_gpt2")
# print("Fine-tuning complete. Model saved to './fine_tuned_gpt2'")



# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # Load fine-tuned model and tokenizer
# model_path = "./fine_tuned_gpt2"
# print("Loading fine-tuned GPT model...")
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# model = GPT2LMHeadModel.from_pretrained(model_path)
# print("Fine-tuned model loaded!")

# # Generate text function
# def generate_text(prompt, model, tokenizer, max_length=100, temperature=1.0):
#     """
#     Use the fine-tuned GPT model to generate text.
#     :param prompt: Starting text prompt
#     :param model: The fine-tuned GPT model
#     :param tokenizer: Tokenizer for the model
#     :param max_length: Maximum length of generated text
#     :param temperature: Sampling temperature for diversity
#     :return: Generated text
#     """
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(
#         inputs.input_ids,
#         max_length=max_length,
#         temperature=temperature,
#         top_p=0.7,
#         repetition_penalty=1.5,  # 抑制重复生成
#         do_sample=True
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Provide a custom prompt
# prompt = "This luxurious perfume evokes feelings of"
# generated_text = generate_text(prompt, model, tokenizer)
# print("Generated Text with Fine-Tuned GPT:")
# print(generated_text)
















# gpt模型
import pandas as pd

# Step 1: Load CSV Data
file_path = 'final_perfume_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 2: Extract descriptions and save to a text file
descriptions = data['Description'].dropna().tolist()  # Get non-empty descriptions
output_file = 'perfume_descriptions.txt'

with open(output_file, 'w') as f:
    for description in descriptions:
        f.write(description.strip() + "\n")

print(f"Descriptions saved to {output_file}")


from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Load pre-trained GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("GPT-2 model loaded!")

# Step 2: Load dataset
def load_dataset(file_path, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset("perfume_descriptions.txt")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=5,  # You can increase this for better results
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
)

# Step 4: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("Fine-tuning complete. Model saved to './fine_tuned_gpt2'")



from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_gpt2"
print("Loading fine-tuned GPT model...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
print("Fine-tuned model loaded!")

# Generate text function
def generate_text(prompt, model, tokenizer, max_length=100, temperature=1.0):
    """
    Use the fine-tuned GPT model to generate text.
    :param prompt: Starting text prompt
    :param model: The fine-tuned GPT model
    :param tokenizer: Tokenizer for the model
    :param max_length: Maximum length of generated text
    :param temperature: Sampling temperature for diversity
    :return: Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=0.7,
        repetition_penalty=1.5,  # 抑制重复生成
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Provide a custom prompt
prompt = "This luxurious perfume evokes feelings of"
generated_text = generate_text(prompt, model, tokenizer)
print("Generated Text with Fine-Tuned GPT:")
print(generated_text)
