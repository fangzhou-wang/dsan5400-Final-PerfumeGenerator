import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Load data
file_path = 'fra_cleaned.csv'
urls_df = pd.read_csv(file_path, usecols=[0, 1, 2], header=0)
urls_df.columns = ['URL', 'Perfume', 'Brand']
urls_df = urls_df[urls_df['URL'].str.startswith('http')]
urls_df = urls_df.iloc[26:50]
# User-Agent pool
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
]

# Scrape reviews
def scrape_reviews(url, perfume_name):
    reviews = []
    headers = {"User-Agent": random.choice(user_agents)}
    for attempt in range(3):  # Retry mechanism
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Debugging HTML
            with open(f"debug_{perfume_name}.html", "w", encoding="utf-8") as f:
                f.write(soup.prettify())

            # Find review elements
            for review_element in review_elements:
    try:
        # Safely attempt to find the element
        review_text_div = review_element.find('div', class_='flex-child-auto')
        if review_text_div:  # Check if the element exists
            review_text = review_text_div.text.strip()
            reviews.append({
                'Perfume Name': perfume_name,
                'Review Text': review_text,
                'URL': url
            })
        else:
            print(f"Review text not found for {perfume_name} in element.")
    except Exception as e:
        print(f"Error extracting review for {perfume_name}: {e}")

            time.sleep(random.uniform(30, 60))  # Random delay
            break
        except requests.exceptions.RequestException as e:
            print(f"Retrying ({attempt+1}/3) for {url}: {e}")
            time.sleep(30)
    return reviews

# Iterate through URLs
all_reviews = []
batch_size = 5
for i in range(0, len(urls_df), batch_size):
    batch = urls_df.iloc[i:i + batch_size]
    for _, row in batch.iterrows():
        url = row['URL']
        perfume_name = row['Perfume']
        print(f"Scraping reviews for '{perfume_name}' from {url}")
        reviews = scrape_reviews(url, perfume_name)
        all_reviews.extend(reviews)
    print("Waiting before next batch...")
    time.sleep(120)  # Cooldown period

# Save to CSV
reviews_df = pd.DataFrame(all_reviews)
if not reviews_df.empty:
    reviews_df.to_csv('extracted_reviews_with_perfume_names.csv', index=False)
    print("Reviews saved.")
else:
    print("No reviews were collected.")
