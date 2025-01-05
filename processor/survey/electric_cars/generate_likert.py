import json

# Training data
train_data = [
    {"prompt": "Electric vehicles are becoming more accessible to a wide range of consumers.", 
     "completion": "How strongly do you agree with the statement: 'Electric vehicles are becoming more accessible to a wide range of consumers.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "The adoption of electric vehicles is growing rapidly in many countries.", 
     "completion": "To what extent do you agree with the statement: 'The adoption of electric vehicles is growing rapidly in many countries.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "Battery technology for electric cars has significantly improved in recent years.", 
     "completion": "How much do you agree with the statement: 'Battery technology for electric cars has significantly improved in recent years.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "Electric cars have become more affordable for the average consumer.", 
     "completion": "How strongly do you agree with the statement: 'Electric cars have become more affordable for the average consumer.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "The environmental benefits of electric cars are well-documented and widely accepted.", 
     "completion": "How much do you agree with the statement: 'The environmental benefits of electric cars are well-documented and widely accepted.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."}
]

# Validation data
valid_data = [
    {"prompt": "Electric vehicles have lower maintenance costs compared to traditional gasoline-powered cars.", 
     "completion": "How strongly do you agree with the statement: 'Electric vehicles have lower maintenance costs compared to traditional gasoline-powered cars.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "The range of electric vehicles has improved, allowing them to travel longer distances on a single charge.", 
     "completion": "How much do you agree with the statement: 'The range of electric vehicles has improved, allowing them to travel longer distances on a single charge.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "The availability of charging stations for electric vehicles has increased in recent years.", 
     "completion": "How much do you agree with the statement: 'The availability of charging stations for electric vehicles has increased in recent years.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "Electric vehicles contribute to reducing overall air pollution in urban areas.", 
     "completion": "How much do you agree with the statement: 'Electric vehicles contribute to reducing overall air pollution in urban areas.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."},
    {"prompt": "The cost of electric vehicle batteries is expected to continue decreasing in the future.", 
     "completion": "How much do you agree with the statement: 'The cost of electric vehicle batteries is expected to continue decreasing in the future.'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."}
]

# Save training data to train.jsonl
with open('train.jsonl', 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + '\n')

# Save validation data to valid.jsonl
with open('valid.jsonl', 'w') as f:
    for entry in valid_data:
        f.write(json.dumps(entry) + '\n')
