import pandas as pd
import json
import os

def preprocess_data(input_csv, output_jsonl):
    """
    Clean and transform data sets to convert CSV format to JSONL format.

    Args:
        input_csv (str): Enter the CSV file path.
        output_jsonl (str): The path to the output JSONL file.
    """
    # Ensure that the output directory exists
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    df = pd.read_csv(input_csv)

    # Clean text
    def clean_text(text):
        if pd.isna(text):  # Processing missing value
            return ""
        text = text.replace("\n", " ").strip()
        return text

    df['prompt'] = df['Perfume Name'].apply(lambda x: f"Describe the perfume: {x}")
    df['completion'] = df['Review Text'].apply(clean_text)

    # Save in JSONL format
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json_line = {
                "prompt": row["prompt"],
                "completion": row["completion"]
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Data preprocessed and saved to {output_jsonl}")

# Running data preprocessing
if __name__ == "__main__":
    preprocess_data("../../data/extracted_reviews_with_perfume_names.csv", "../../data/perfume_reviews.jsonl")
