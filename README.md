# **Perfume Recommender & Custom Description Generator**  

This project is an end-to-end **NLP-powered perfume recommendation system** that integrates **sentiment analysis**, **decision tree-based recommendation**, and **custom text generation** using **GPT-2**. Users can interact with the system through a web interface built using **Streamlit**.

---

## **Table of Contents**  

1. [Project Overview](#project-overview) 
2. [Team Members](#team-members)
3. [Features](#features)  
4. [Technologies Used](#technologies-used)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [How It Works](#how-it-works)  
8. [Project Structure](#project-structure)  
9. [Future Enhancements](#future-enhancements)  
10. [License](#license)  

---

## **Project Overview**  

This project recommends perfumes based on user-selected attributes such as **main accords** and **sentiment scores**. It also generates custom descriptions using a **fine-tuned GPT-2 language model**. The system combines **NLP techniques** like **text preprocessing**, **sentiment analysis**, **decision tree modeling**, and **text generation** into a unified pipeline.

---

## **Team Members**

* Kexin Lyu
* Fangzhou Wang
* Shuyu Meng

---

## **Features**  

✔️ **Interactive User Input:**  
Users can choose main accords, detailed accords, and apply sentiment score filters.  

✔️ **Sentiment Analysis:**  
User reviews are analyzed to calculate sentiment scores using **VADER**.  

✔️ **Decision Tree-Based Recommendations:**  
Predicts the next likely main accords and recommends relevant perfumes.  

✔️ **Custom Description Generation:**  
Generates unique perfume descriptions using a **GPT-2** model.  

✔️ **Web Interface:**  
A clean and interactive interface built using **Streamlit**.  

---

## **Technologies Used**  

- **Programming Language:** Python 3.9+  
- **Libraries:**  
  - **NLP:** Transformers (GPT-2), NLTK (VADER)  
  - **Machine Learning:** scikit-learn (DecisionTreeClassifier)  
  - **Web Framework:** Streamlit  
  - **Data Handling:** Pandas  
- **Environment Management:** Conda  
- **Code Quality:** Pylint, Black, Ruff  

---

## **Installation**  

1. **Clone the repository:**  

   git clone https://github.com/fangzhou-wang/dsan5400-Final-PerfumeGenerator
   cd perfume-recommender


2. **Create a Conda environment using the provided environment file:**  

   conda env create -f environment.yml
   conda activate perfume-recommender


3. **Download necessary resources:**  

   python -m nltk.downloader vader_lexicon

---

## **Usage**  

1. **Run the Streamlit web application:**  

   streamlit run website_generation.py


2. **Interact with the web app:**  

   - Select up to 5 main accords.
   - Choose a sentiment score threshold to filter recommendations.
   - View the list of recommended perfumes.
   - Read custom-generated descriptions for the perfumes.  

---

## **How It Works**  

### **1. Data Preprocessing:**  

- Cleans perfume descriptions and user reviews.  
- Sentiment scores are calculated using VADER.

### **2. Recommendation Engine:**  

- Decision Tree Models:
   1. Predicts next main accords.
   2. Filters perfumes based on selected main accords, detailed accords, and sentiment score.

### **3. Text Generation:** 

- A fine-tuned GPT-2 model generates personalized descriptions for the recommended perfumes.

### **4. User Interaction:**  

- The Streamlit interface collects user input and displays recommendations and generated descriptions.

---

## **Project Structure**  

```
.
├── README.md
├── environment.yml
├── perfume_recommender
│   ├── data
│   │   └── fra_cleaned.csv
│   ├── perfume_recommender
│   │   ├── __init__.py
│   │   ├── data_processing
│   │   │   ├── __init__.py
│   │   │   └── data_processing.py
│   │   ├── perfume_recommendation
│   │   │   ├── __init__.py
│   │   │   └── perfume_recommendation.py
│   │   ├── text_generation
│   │   │   ├── __init__.py
│   │   │   └── text_generation.py
│   │   └── website_generation.py
│   └── tests
│       ├── __init__.py
│       ├── sample_perfume_data.csv
│       ├── sample_review_data.csv
│       ├── test_data_processing.py
│       ├── test_perfume_recommendation.py
│       └── test_text_generation.py
└── pyproject.toml

```

---

## **Future Enhancements**  

🔮 **Enhanced Recommendation Model:** Incorporate advanced recommendation techniques like collaborative filtering or neural networks. 

🌍 **Multilingual Support:** Support multiple languages using multilingual models like mT5 or XLM-R.  

📊 **Data Visualization:** Include visualizations to display sentiment trends, recommendation statistics, and user input summaries.  

💬 **Voice Interaction:** Integrate voice input/output for a hands-free user experience.

---

## **License**  

This project is licensed under the MIT License. See `LICENSE` for more details.  

---

**Happy Exploring! 🚀**  
For contributions, feel free to fork the project and submit a pull request. 