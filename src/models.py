from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import numpy


class SentimentModel():
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def predict_sentiment(self, texts: str) -> List[str]:
        cl_texts = self.clean_text(texts)
        inputs = self.tokenizer(cl_texts, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs)

        scores = F.softmax(logits[0], dim=1)

        confidence_prob, predicated_class = torch.max(scores, 1)

        labels = {'label': self.model.config.id2label[predicated_class.item(
        )], 'confidence': confidence_prob.item(), "text": texts}

        return labels

    def replace_numbers(self, text: str) -> str:
        return text.replace("0", " null").replace("1", " eins").replace("2", " zwei").replace("3", " drei").replace("4", " vier").replace("5", " fünf").replace("6", " sechs").replace("7", " sieben").replace("8", " acht").replace("9", " neun")

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub('', text)
        text = self.clean_at_mentions.sub('', text)
        text = self.replace_numbers(text)
        text = self.clean_chars.sub('', text)  # use only text chars
        # substitute multiple whitespace with single whitespace
        text = ' '.join(text.split())
        text = text.strip().lower()
        return text
