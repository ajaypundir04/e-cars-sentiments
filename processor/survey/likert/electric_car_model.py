from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

class LikertScaleQuestionGenerator:
    def __init__(self, model_name="t5-small", fine_tuned_model_path="./fine_tuned_likert_model"):
        self.model_name = model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        
        # Load the tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        
        # Load the pretrained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def prepare_data(self, data):
        """ Prepare the dataset and tokenize """
        # Convert data into HuggingFace Dataset object
        dataset = Dataset.from_dict(data)

        print('data')
        print(dataset)
        
        # Ensure the labels are integers (classification labels should be integers)
        dataset = dataset.map(lambda x: {"label": int(x["label"])})
        
        # Tokenize data
        def tokenize_function(examples):
            input_text = ["Generate a Likert scale question: " + str(text) for text in examples["text"]]
            return self.tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # Apply tokenizer to dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        # Make sure the dataset returns proper labels
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return tokenized_datasets

    def split_data(self, tokenized_datasets, test_size=0.1):
        """ Split dataset into training and validation sets """
        return tokenized_datasets.train_test_split(test_size=test_size)

    def train_model(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3):
        """ Fine-tune the T5 model """
        training_args = TrainingArguments(
            output_dir=output_dir,             # output directory
            evaluation_strategy="epoch",       # evaluation strategy to use
            learning_rate=2e-5,                # learning rate
            per_device_train_batch_size=8,     # batch size for training
            per_device_eval_batch_size=8,      # batch size for evaluation
            num_train_epochs=num_epochs,       # number of training epochs
            weight_decay=0.01,                 # strength of weight decay
        )
        
        # Define Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(self.fine_tuned_model_path)

    def infer(self, context):
        """ Generate a Likert scale question from context """
        input_text = f"Generate a Likert scale question: {context}"
        
        # Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Generate the question
        output = self.model.generate(**inputs)
        
        # Decode the output
        generated_question = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_question


# Sample data for electric car-related Likert scale question generation
data = {
    "text": [
        "Electric vehicles and battery technology",
        "Electric vehicles and their environmental impact",
        "Cost of electric vehicles and affordability"
    ],
    "label": [
        4,  # Numeric labels instead of strings
        5,  # Numeric labels instead of strings
        2   # Numeric labels instead of strings
    ]
}

# Initialize the LikertScaleQuestionGenerator class
likert_generator = LikertScaleQuestionGenerator()

# 1. Dataset Processing
tokenized_datasets = likert_generator.prepare_data(data)
train_test_split = likert_generator.split_data(tokenized_datasets)

train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

print(train_dataset)

print(val_dataset)

# 2. Model Training
likert_generator.train_model(train_dataset, val_dataset)

# 3. Inference
context = "How important is it to reduce emissions with electric vehicles?"
generated_question = likert_generator.infer(context)

print(f"Generated Likert scale question: {generated_question}")
