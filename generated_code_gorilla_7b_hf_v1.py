
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def process_data(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    response = {
        'availability': scores[0],
        'environmental impact': scores[1],
        'cost': scores[2],
        'reliability': scores[3],
        'flexibility': scores[4],
        'sustainability': scores[5]
    }
    return response

input_text = "What are the key differences between renewable and non-renewable energy sources?"
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Load the model and token