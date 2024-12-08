### **README.md: Perfume Recommendation and Custom Description Generation Project**  

---

# **Perfume Recommender & Description Generator**  
This project is an end-to-end **NLP-powered perfume recommendation system** that integrates **sentiment analysis**, **decision tree-based recommendation**, and **custom text generation** using **GPT-2**. Users can interact with the system through a web interface built using **Streamlit**.

---

## **Table of Contents**  

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [How It Works](#how-it-works)  
7. [Project Structure](#project-structure)  
8. [Future Enhancements](#future-enhancements)  
9. [License](#license)  

---

## **Project Overview**  

This project recommends perfumes based on user-selected attributes such as **main accords** and **sentiment scores**. It also generates custom descriptions using a **fine-tuned GPT-2 language model**. The system combines **NLP techniques** like **text preprocessing**, **sentiment analysis**, **decision tree modeling**, and **text generation** into a unified pipeline.

---

## **Features**  

✔️ **Interactive User Input:**  
Users can choose perfume features (main accords, gender) and filter by sentiment score.  

✔️ **Sentiment Analysis:**  
The system analyzes user reviews and calculates sentiment scores using **VADER**.  

✔️ **Decision Tree-Based Recommendations:**  
Based on main accords, the system recommends relevant perfumes.  

✔️ **Custom Description Generation:**  
Generates unique perfume descriptions with a fine-tuned **GPT-2** model.  

✔️ **Web Interface:**  
Built using **Streamlit** for seamless user interaction.  

---

## **Technologies Used**  

- **Programming Language:** Python 3.9+  
- **Libraries:**  
  - **NLP:** Transformers (GPT-2), NLTK (VADER)  
  - **Machine Learning:** scikit-learn (DecisionTreeClassifier)  
  - **Web Framework:** Streamlit  
  - **Data Handling:** Pandas  
  - **Environment Management:** Anaconda  

---

## **Installation**  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-username/perfume-recommender.git
   cd perfume-recommender
   ```

2. **Create a virtual environment:**  
   ```bash
   conda create --dsan5400 
   conda activate dsan5400
   ```

3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download necessary models:**  
   ```bash
   python -m nltk.downloader vader_lexicon
   ```

---

## **Usage**  

1. **Run the application:**  
   ```bash
   streamlit run app.py
   ```

2. **Interact with the web app:**  
   - Choose main accords (e.g., Floral, Woody).  
   - Set the sentiment threshold.  
   - View recommendations.  
   - See custom-generated descriptions.  

---

## **How It Works**  

### **1. Data Preprocessing:**  
- Cleans perfume descriptions and user reviews.  
- Removes unwanted characters and formats text.

### **2. Sentiment Analysis:**  
- Analyzes reviews using **VADER** to calculate sentiment scores.

### **3. Recommendation Engine:**  
- **Decision Tree Classifier** predicts likely next main accords.  
- Recommends perfumes based on user preferences and sentiment scores.

### **4. Custom Description Generation:**  
- A **fine-tuned GPT-2 model** generates personalized perfume descriptions based on recommendations.

---

## **Project Structure**  

```
perfume-recommender/
│── app.py                    # Streamlit web app
│── recommend.py              # Main backend logic (NLP + ML)
│── generatetext.py          # Fine-tune GPT-2 model
│── extracted_final_perfume_data.csv    #comments dataset used to train generated model
├── fra_cleaned.csv       # Cleaned perfume dataset
├── final_perfume_data.csv           # User reviews dataset
│── environment.yml          # Dependencies
└── README.md
```

---

## **Future Enhancements**  

🔮 **Enhanced Recommendation Model:** Add more complex recommendation systems such as collaborative filtering.  

🌍 **Multilingual Support:** Expand to other languages using multi-lingual models like mT5 or BERT.  

📊 **Data Visualization:** Add charts to show sentiment trends and recommendation stats.  

💬 **Voice Interaction:** Add voice input and output functionality for a hands-free experience.  

---

## **License**  

This project is licensed under the MIT License. See `LICENSE` for more details.  

---

**Happy Exploring! 🚀**  
For contributions, feel free to fork the project and submit a pull request. 