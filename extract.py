import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Load the CSV file with URLs and perfume names
file_path = 'fra_cleaned.csv'  # Replace with your actual file path
urls_df = pd.read_csv(file_path, usecols=[0, 1, 2], header=None)  # Load first three columns
urls_df.columns = ['URL', 'Perfume', 'Brand']  # Assign column names
urls_df = urls_df.head(100)  # Limit to first 10 rows for testing

# Function to scrape reviews from a URL
def scrape_reviews(url, perfume_name):
    reviews = []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Save the HTML for debugging
        with open(f"debug_{perfume_name}.html", "w", encoding="utf-8") as f:
            f.write(soup.prettify())

        # Find review elements
        review_elements = soup.find_all('div', class_='cell fragrance-review-box')  # Adjust as needed
        print(f"Found {len(review_elements)} review elements for {perfume_name} at {url}.")
        for review_element in review_elements:
            review_div = review_element.find('div', class_='flex-child-auto')
            if review_div:
                review_text = review_div.text.strip()
                reviews.append({
                    'Perfume Name': perfume_name,
                    'Review Text': review_text,
                    'URL': url
                })
            else:
                print(f"No review text found for {perfume_name} at {url}.")
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
    return reviews

# Iterate over URLs and perfume names in the CSV and scrape reviews
all_reviews = []
for index, row in urls_df.iterrows():
    url = row['URL']  # Use the URL directly
    perfume_name = row['Perfume']
    print(f"Scraping reviews for '{perfume_name}' from {url}")
    reviews = scrape_reviews(url, perfume_name)
    all_reviews.extend(reviews)

# Save the reviews to a CSV file
reviews_df = pd.DataFrame(all_reviews)
if not reviews_df.empty:
    output_file = 'extracted_reviews_with_perfume_names.csv'
    reviews_df.to_csv(output_file, index=False)
    print(f"Reviews saved to {output_file}")
else:
    print("No reviews were collected.")
