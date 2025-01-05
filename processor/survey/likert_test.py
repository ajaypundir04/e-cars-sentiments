from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import os
import evaluate

from utils.utils import Utils

class LikertQuestionModel:
    def __init__(self, model_name="MaRiOrOsSi/t5-base-finetuned-question-answering", train_file="train.jsonl", valid_file="valid.jsonl", 
                 output_dir="./t5_likert_finetuned_ev"):
        self.model_name = model_name
        self.train_file = train_file
        self.valid_file = valid_file
        self.output_dir = output_dir
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        
        # Load dataset
        self.dataset = load_dataset("json", data_files={"train": self.train_file, "validation": self.valid_file})
        self.tokenized_datasets = None
    
    def preprocess_data(self, batch):
        # Preparing inputs in the form: "Generate a Likert scale question based on the following statement: {passage}"
        inputs = [f"Generate a Likert scale question based on the following statement: {context}" for context in batch["context"]]
        targets = batch["question"]
        
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def tokenize_dataset(self):
        # Tokenize the dataset
        self.tokenized_datasets = self.dataset.map(self.preprocess_data, batched=True)
    
    def train_model(self, learning_rate=3e-5, batch_size=8, num_epochs=3, save_steps=1000):
        if self.tokenized_datasets is None:
            raise ValueError("Tokenized datasets are not prepared. Call `tokenize_dataset()` first.")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=save_steps,
            eval_steps=save_steps,
            logging_steps=500,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            save_total_limit=3,
            weight_decay=0.01,
            warmup_steps=500,
            report_to=["tensorboard"]
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        self.save_model()
    
    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def generate_question(self, passage):
        # Format the input as a Likert question generation prompt
        context = f"Generate a Likert scale question based on the following statement: '{passage}'"
        inputs = self.tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate the Likert-scale question
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=5,  # Beam search to explore multiple options
            do_sample=False,  # Ensure deterministic output
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def evaluate_model(self, eval_file="valid.jsonl"):
        eval_dataset = load_dataset("json", data_files={"validation": eval_file})["validation"]
        rouge = evaluate.load("rouge")

        for example in eval_dataset:
            passage = example["context"]
            true_question = example["question"]
            generated_question = self.generate_question(passage)

            print(f"Passage: {passage}")
            print(f"True Question: {true_question}")
            print(f"Generated Question: {generated_question}")
            print("-" * 50)

            rouge.add(prediction=generated_question, reference=true_question)

        # Compute evaluation metrics
        results = rouge.compute()
        print(f"Evaluation Results: {results}")

    def summarize_passages(self, file_paths):
        all_passages = []
        
        passages = Utils.scrape_data_from_file(file_path)
        all_passages.extend(passages)
        #print(f"passages::${passages}")
        # Join all passages into a single string
        combined_passage = " ".join(all_passages)
        
        # Use the summarization model to create a summary
        summarized_text = self.summarizer(combined_passage, max_length=150, min_length=50, do_sample=False)
        summary = summarized_text[0]['summary_text']
        return summary
    
if __name__ == "__main__":
    model = LikertQuestionModel(train_file="train.jsonl", 
                                valid_file="valid.jsonl", 
                                output_dir="./t5_likert_finetuned_ev_qa")
    
    # Tokenize the dataset
    model.tokenize_dataset()

    # Train the model
    model.train_model(learning_rate=3e-5, batch_size=8, num_epochs=3, save_steps=1000)

    # Test the model by generating a Likert question
    file_paths = ['stats/ev_china.md', 'stats/ev_germany.md', 'stats/ev_norway.md', 
                          'stats/hybrid_germany.md', 'stats/stats.md', 'stats/reviews.csv']
    for file_path in file_paths:
        passage = model.summarize_passages(file_path)
        question = model.generate_question(passage)
        print(f"Generated Likert Question: {question}")

    # Evaluate the model if you have a validation set
    #model.evaluate_model(eval_file="valid.jsonl")
