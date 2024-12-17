import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    """
    A class for generating perfume description text using a fine-tuned GPT-2 model.
    """
    def __init__(self, model_path="./fine_tuned_gpt2"):
        """
        Initialize the TextGenerator class and load the GPT-2 model and tokenizer.

        Args:
            model_path (str): Path to the pre-trained or fine-tuned GPT-2 model.
        """
        logger.info("Initializing TextGenerator...")
        try:
            logger.info(f"Loading GPT-2 tokenizer from {model_path}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

            logger.info(f"Loading GPT-2 model from {model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading GPT-2 model or tokenizer: {e}")
            raise

    def generate_description(self, prompt):
        """
        Generate a descriptive text based on the input prompt using GPT-2.

        Args:
            prompt (str): Input prompt to guide the text generation.

        Returns:
            str: Generated descriptive text.
        """
        logger.info(f"Generating text for prompt: '{prompt}'")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids, max_length=100, temperature=0.7, top_p=0.9, do_sample=True
            )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated text: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return "Error occurred during text generation."