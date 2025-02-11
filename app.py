import asyncio

import gradio as gr
import torch.nn.functional as F
from googletrans import Translator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")
model.config.id2label = {0: "เศร้า", 1: "มีความสุข", 2: "รัก", 3: "โกรธ", 4: "กลัว", 5: "ตกใจ"}


async def translate_text(txt):
    async with Translator() as translator:
        out = await translator.translate(txt, dest="en")
    return out.text


def predict_emotion(sentence: str, translate: bool) -> str:
    if translate:
        sentence = asyncio.run(translate_text(sentence))
    inputs = tokenizer(sentence, return_tensors="pt")
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1) * 100
    labels = model.config.id2label

    pred_index = probs.argmax().item()
    predicted_emotion = labels[pred_index]
    predicted_prob = probs[0, pred_index].item()

    html_output = (
        f"<h1 style='text-align:center; font-size:48px;'>{predicted_emotion} ({predicted_prob:.2f}%)</h1>"
    )

    other_emotions = []
    for idx, prob in enumerate(probs[0]):
        if idx != pred_index:
            other_emotions.append((labels[idx], prob.item()))

    other_emotions_sorted = sorted(other_emotions, key=lambda x: x[1], reverse=True)

    html_output += "<ul style='list-style-type:none; font-size:24px; text-align:center;'>"
    for emotion, prob in other_emotions_sorted:
        html_output += f"<li>{emotion}: {prob:.2f}%</li>"
    html_output += "</ul>"
    return html_output


with gr.Blocks() as iface:
    gr.Markdown("## โปรแกรมคาดการณ์อารมณ์จากข้อความ")
    sentence = gr.Textbox(lines=2, placeholder="ใส่ข้อความที่นี่", label="ใส่ข้อความ")
    translate = gr.Checkbox(label="ไม่ใช่ภาษาอังกฤษ", value=True)
    output = gr.HTML(label="ผลการคาดการณ์")
    predict_button = gr.Button("เริ่มต้น")
    predict_button.click(fn=predict_emotion, inputs=[sentence, translate], outputs=output)

iface.launch()
