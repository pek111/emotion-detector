import asyncio

import torch
import torch.nn.functional as F
from googletrans import Translator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

model.config.id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

print("Enter a sentence to predict the emotion.")

text = input("\nEnter a sentence: ")

async def translate(txt):
    async with Translator() as translator:
        out = await translator.translate(txt, dest="en")
    return out.text

if input("Translate the sentence to English? (y/n): ").lower() == "y":
    text = asyncio.run(translate(text))

inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits
probs = F.softmax(logits, dim=-1) * 100

labels = model.config.id2label

for idx, percent in enumerate(probs[0]):
    print(f"{labels[idx]}: {percent.item():.2f}%")

print("\nPredicted emotion:", labels[probs.argmax().item()])
