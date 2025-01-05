from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

class T5LikertFineTuner:
    def __init__(self, model_name="t5-small", train_file="train.jsonl", valid_file="valid.jsonl", output_dir="./t5_likert_finetuned"):
        self.model_name = model_name
        self.train_file = train_file
        self.valid_file = valid_file
        self.output_dir = output_dir
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.dataset = load_dataset("json", data_files={"train": self.train_file, "validation": self.valid_file})
        self.tokenized_datasets = None
    
    def preprocess_data(self, batch):
        inputs = [f"Generate a Likert scale question based on the following text: {prompt}" for prompt in batch["prompt"]]
        targets = batch["completion"]
        
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
        model_inputs["labels"] = labels
        
        # Log a sample of tokenized inputs and labels for debugging
        print(f"Sample input: {inputs[0]}")
        print(f"Sample label: {targets[0]}")
        print(f"Tokenized input: {model_inputs}")
        return model_inputs

    
    def tokenize_dataset(self):
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
            report_to=["tensorboard"]  # Logs metrics to TensorBoard
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

if __name__ == "__main__":
    finetuner = T5LikertFineTuner(train_file="train.jsonl", 
                                  valid_file="valid.jsonl", 
                                  output_dir="./t5_likert_finetuned_ev")
    finetuner.tokenize_dataset()
    finetuner.train_model(learning_rate=3e-5, batch_size=8, num_epochs=3, save_steps=1000)
