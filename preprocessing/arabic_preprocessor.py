import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

class arabicPreprocessor:
    def __init__(self, model_name='aubmindlab/bert-base-arabertv2', 
                 normalize_alef=True, normalize_teh=False, 
                 normalize_yeh=True, normalize_hamza=True):
        """
        Args:
            model_name: Hugging Face model name for tokenizer
            normalize_alef: Normalize different forms of Alef
            normalize_teh: Normalize taa marbuta to haa (careful!)
            normalize_yeh: Normalize alef maksura to yeh
            normalize_hamza: Normalize hamza forms
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalize_alef = normalize_alef
        self.normalize_teh = normalize_teh
        self.normalize_yeh = normalize_yeh
        self.normalize_hamza = normalize_hamza

    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_diacritics(self, text):
        arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        return re.sub(arabic_diacritics, '', text)

    def normalize_arabic(self, text):
        if self.normalize_alef:
            text = re.sub("[إأآا]", "ا", text)
        if self.normalize_yeh:
            text = re.sub("ى", "ي", text)
        if self.normalize_hamza:
            text = re.sub("ؤ", "ء", text)
            text = re.sub("ئ", "ء", text)
        if self.normalize_teh:
            text = re.sub("ة", "ه", text)
        return text

    def preprocess(self, text):
        """Clean text before tokenization"""
        # Remove HTML
        text = BeautifulSoup(text, features="html.parser").get_text()
        
        # Remove emojis (optional - consider keeping for sentiment tasks)
        text = self.remove_emojis(text)
        
        # Remove diacritics
        text = self.remove_diacritics(text)
        
        # Normalize letters
        text = self.normalize_arabic(text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # def tokenize(self, text, max_length=128, **kwargs):
    #     """
    #     Tokenizes text for Transformer input.
        
    #     Args:
    #         text: Input text (can be string or list of strings)
    #         max_length: Maximum sequence length
    #         **kwargs: Additional arguments for tokenizer
            
    #     Returns:
    #         Dictionary with input_ids, attention_mask, etc.
    #     """
    #     return self.tokenizer(
    #         text,
    #         padding='max_length',
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors='pt',
    #         **kwargs
    #     )
    
    # def preprocess_and_tokenize(self, text, max_length=128):
    #     """Convenience method to preprocess and tokenize in one step"""
    #     cleaned = self.preprocess(text)
    #     return self.tokenize(cleaned, max_length=max_length)