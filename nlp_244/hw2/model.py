from transformers import AutoModelForMaskedLM

current_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
