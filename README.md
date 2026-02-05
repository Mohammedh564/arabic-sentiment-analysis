# Arabic Sentiment Analysis 


End-to-end Arabic sentiment analysis using AraBERT fine-tuned with LoRA. Includes preprocessing, baseline ML model, transformer training, evaluation, error analysis, and deployment via FastAPI for real-time sentiment prediction.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the FastAPI server with:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

You can test endpoints using:

* Browser (Swagger UI): `/docs`
* Postman

---

## Model

* Base model: `aubmindlab/bert-base-arabertv2`
* Fine-tuning: LoRA (PEFT)
* Task: Arabic sentiment classification

---

## Project Structure

```
├── app/
│   ├── main.py
│   └── ...
├── models/
├── data/
├── requirements.txt
├── README.md
```

---

## Notes

* Text preprocessing is applied before tokenization
* Designed for easy local testing and deployment
