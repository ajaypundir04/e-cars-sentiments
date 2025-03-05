import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Fine-Tuned Model
model_path = "ajay-pundir/e_car_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample Test Data (Manually Created)
test_words = ["efficient", "expensive", "charging station", "battery", "eco-friendly"]
true_labels = [2, 0, 1, 0, 2]  # Ground truth labels (Positive=2, Neutral=1, Negative=0)

# Tokenization
encoded_inputs = tokenizer(
    test_words, padding=True, truncation=True, max_length=32, return_tensors="pt"
).to(device)

# Model Prediction
model.eval()
with torch.no_grad():
    outputs = model(**encoded_inputs)
    predictions = torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=-1), dim=-1).cpu().numpy()

# Evaluation Metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

# Print Results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
