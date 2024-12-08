import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,  # Change to DEBUG or ERROR as needed
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Create a logger instance
logger = logging.getLogger("PerfumeRecommender")
