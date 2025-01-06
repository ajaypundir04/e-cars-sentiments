import json

# Define the dataset for training and validation (the train data can be larger).
train_data = [
    {"context": "Electric vehicles are becoming more accessible to a wide range of consumers.", "answer": "Likely", "question": "What is the likelihood that electric vehicles are becoming more accessible to consumers?"},
    {"context": "The adoption of electric vehicles is growing rapidly in many countries.", "answer": "Not Likely", "question": "What is the likelihood that the adoption of electric vehicles is growing rapidly?"},
    {"context": "Battery technology for electric cars has significantly improved in recent years.", "answer": "Most Likely", "question": "What is the likelihood that battery technology for electric cars has improved significantly?"},
    {"context": "Electric cars have become more affordable for the average consumer.", "answer": "Disagree", "question": "Do you agree that electric cars have become more affordable for consumers?"},
    {"context": "The environmental benefits of electric cars are well-documented and widely accepted.", "answer": "Strongly Agree", "question": "Do you strongly agree that electric cars have environmental benefits?"}
]

valid_data = [
    {"context": "Electric vehicles have lower maintenance costs compared to traditional gasoline-powered cars.", "answer": "Likely", "question": "What is the likelihood that electric vehicles have lower maintenance costs compared to gasoline cars?"},
    {"context": "The range of electric vehicles has improved, allowing them to travel longer distances on a single charge.", "answer": "Not Likely", "question": "What is the likelihood that the range of electric vehicles has improved for longer travel distances?"},
    {"context": "The availability of charging stations for electric vehicles has increased in recent years.", "answer": "Most Likely", "question": "What is the likelihood that the availability of EV charging stations has increased recently?"},
    {"context": "Electric vehicles contribute to reducing overall air pollution in urban areas.", "answer": "Disagree", "question": "Do you agree that electric vehicles help reduce air pollution in urban areas?"},
    {"context": "The cost of electric vehicle batteries is expected to continue decreasing in the future.", "answer": "Strongly Agree", "question": "Do you agree that the cost of electric vehicle batteries will continue to decrease?"}
]

# Write the training data to a JSONL file
with open("train.jsonl", "w") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

# Write the validation data to a JSONL file
with open("valid.jsonl", "w") as f:
    for entry in valid_data:
        f.write(json.dumps(entry) + "\n")
