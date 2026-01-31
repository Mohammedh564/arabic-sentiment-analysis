from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

class SentimentPredictor:
    def __init__(self):
        # Build relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        self.model_path = os.path.join(parent_dir, "models", "transformer", "araBERT_merged")
        
        # Label mapping
        self.label_map = {
            0: "positive",
            1: "neutral", 
            2: "negative"
        }
        
        print(f"Loading model from: {self.model_path}")
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            print(f"✓ Model loaded successfully on {self.device}!")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, text):
        """
        Predict sentiment for given text.
        Returns: (predicted_class_id, label_name, confidence_score)
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = logits.argmax().item()
        confidence = probabilities[0][predicted_class_id].item()
        label_name = self.label_map[predicted_class_id]
        
        return predicted_class_id, label_name, confidence
    
    def predict_batch(self, texts, batch_size=32):
        """
        Predict sentiment for multiple texts efficiently.
        Returns: list of (predicted_class_id, label_name, confidence_score)
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_classes = logits.argmax(dim=-1).cpu().numpy()
            confidences = probabilities.max(dim=-1).values.cpu().numpy()
            
            for class_id, conf in zip(predicted_classes, confidences):
                label_name = self.label_map[int(class_id)]
                results.append((int(class_id), label_name, float(conf)))
        
        return results

# Test code
if __name__ == "__main__":
    try:
        import time
        
        predictor = SentimentPredictor()
        
        # Test cases
        test_texts = [
            "هذا المنتج رائع جداً",      # Very good product (positive)
            "المنتج جيد والخدمة سيئة",                # Normal product (neutral)
            "هذا سيء للغاية"             # Very bad (negative)
        ]
        
        print("\nTesting Single Predictions:")
        print("-" * 60)
        
        start = time.time()
        for text in test_texts:
            class_id, label, confidence = predictor.predict(text)
            print(f"Text: {text}")
            print(f"Prediction: {label} (class {class_id})")
            print(f"Confidence: {confidence:.4f}")
            print("-" * 60)
        
        print(f"Time taken: {time.time() - start:.3f}s")
        
        # Batch test
        print("\nTesting Batch Predictions:")
        print("-" * 60)
        many_texts = test_texts * 100
        start = time.time()
        results = predictor.predict_batch(many_texts)
        elapsed = time.time() - start
        print(f"Processed {len(many_texts)} texts in {elapsed:.3f}s")
        print(f"Average: {elapsed / len(many_texts) * 1000:.2f}ms per text")
        
    except Exception as e:
        print(f"Error: {e}")