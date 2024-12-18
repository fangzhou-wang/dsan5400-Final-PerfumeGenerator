import gdown

# Public link to your data file (Google Drive link example)
url = 'https://drive.google.com/drive/folders/1k0QI4B8QA_0HTdN41Y2CbdfX2GYLIKrN'  # Replace FILE_ID with the actual ID
output = 'data.csv'


# Download the file
print("Downloading data...")
gdown.download(url, output, quiet=False)

print(f"Data downloaded and saved as {output}")