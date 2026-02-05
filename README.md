# Arabic Sentiment Analysis 

A simple FastAPI application for Arabic sentiment analysis using a fine-tuned AraBERT model with LoRA (PEFT).

The API takes Arabic text as input and returns the predicted sentiment class.

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
