import gdown

# File ID and download URL
file_id = "1er2Xp9qU0JQyEydZ1PXRkiLEw7vSK1O0"
url = f"https://drive.google.com/uc?id=1er2Xp9qU0JQyEydZ1PXRkiLEw7vSK1O0"
output = "fra_cleaned.csv"  # Name of the file to save locally

# Download the file
print("Downloading the file...")
gdown.download(url, output, quiet=False)
print(f"File downloaded and saved as {output}")



# File ID and download URL
file_id = "1R1rPcjVKfReAAUpBceHb1uw733KdFzbZ"
url = f"https://drive.google.com/uc?id={file_id}"
output = "extracted_reviews_with_perfume_names.csv"  # Name the file appropriately

# Download the file
print("Downloading the file...")
gdown.download(url, output, quiet=False)
print(f"File downloaded and saved as {output}")

import gdown

# File ID and download URL
file_id = "1kv3DkGu_NhOfFeHCi-gEFdKcjcgudtnP"
url = f"https://drive.google.com/uc?id={file_id}"
output = "perfume_reviews.jsonl"  # Name the file appropriately

# Download the file
print("Downloading the JSON file...")
gdown.download(url, output, quiet=False)
print(f"File downloaded and saved as {output}")
