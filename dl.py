from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "BellaAndBria/distilbert-base-uncased-finetuned-emotion"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("model")
tokenizer.save_pretrained("model")
