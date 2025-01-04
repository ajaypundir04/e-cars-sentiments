from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os


class GPT2Trainer:
    def __init__(self, pretrain_output_dir="./pretrained_gpt2", finetune_output_dir="./fine_tuned_gpt2"):
        self.pretrain_output_dir = pretrain_output_dir
        self.finetune_output_dir = finetune_output_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure padding uses EOS token
        self.model = None

    def pretrain(self, dataset_name="wikitext", dataset_config="wikitext-103-raw-v1", num_epochs=3, batch_size=8):
        # Load and tokenize pretraining dataset
        pretrain_dataset = load_dataset(dataset_name, dataset_config)

        def tokenize_function(example):
            return self.tokenizer(
                example["text"], truncation=True, padding="max_length", max_length=128
            )

        pretrain_dataset = pretrain_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Initialize GPT-2 model for pre-training
        config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size, n_positions=1024, n_embd=768, n_layer=12, n_head=12
        )
        self.model = GPT2LMHeadModel(config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.pretrain_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
            learning_rate=5e-4,
            evaluation_strategy="steps",
            logging_dir="./logs_pretrain",
            logging_steps=500,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=pretrain_dataset["train"],
            eval_dataset=pretrain_dataset["validation"],
            tokenizer=self.tokenizer,
        )

        # Train the model
        print("Starting Pre-training...")
        trainer.train()

        # Save the pre-trained model
        self.model.save_pretrained(self.pretrain_output_dir)
        self.tokenizer.save_pretrained(self.pretrain_output_dir)
        print(f"Pre-trained model saved to {self.pretrain_output_dir}")

    def finetune(self, train_data_path, test_data_path, num_epochs=3, batch_size=4):
        # Load pre-trained model
        if self.model is None:
            self.model = GPT2LMHeadModel.from_pretrained(self.pretrain_output_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrain_output_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and tokenize fine-tuning dataset
        finetune_dataset = load_dataset(
            "json", data_files={"train": train_data_path, "test": test_data_path}
        )

        def tokenize_finetune(example):
            return self.tokenizer(
                example["prompt"],
                text_pair=example["response"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        finetune_dataset = finetune_dataset.map(tokenize_finetune, batched=True)

        # Training arguments for fine-tuning
        training_args = TrainingArguments(
            output_dir=self.finetune_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=5_000,
            save_total_limit=2,
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            logging_dir="./logs_finetune",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=finetune_dataset["train"],
            eval_dataset=finetune_dataset["test"],
            tokenizer=self.tokenizer,
        )

        # Fine-tune the model
        print("Starting Fine-tuning...")
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(self.finetune_output_dir)
        self.tokenizer.save_pretrained(self.finetune_output_dir)
        print(f"Fine-tuned model saved to {self.finetune_output_dir}")

# Run the Class to Generate a Fine-tuned Model
if __name__ == "__main__":
    trainer = GPT2Trainer()

    # Pre-train the model
    trainer.pretrain(num_epochs=1, batch_size=4)  # Set lower values for testing purposes

    # Fine-tune the model
    trainer.finetune(train_data_path="train_data.json", test_data_path="test_data.json", num_epochs=1, batch_size=2)
