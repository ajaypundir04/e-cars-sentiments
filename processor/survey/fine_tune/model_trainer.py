import logging
from transformers import TrainingArguments, Trainer
from utils.log_utils import LoggerManager


class ModelTrainer:
    def __init__(self, model, tokenizer, output_dir="./t5_likert_finetuned_ev", log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def train_model(self, tokenized_dataset, learning_rate=3e-5, batch_size=8, num_epochs=3, save_steps=1000):
        self.logger.info("Initializing training arguments...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
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
        self.logger.info("Training arguments initialized.")

        self.logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"]
        )
        self.logger.info("Starting model training...")
        trainer.train()
        self.logger.info("Model training complete.")
        self.save_model()

    def save_model(self):
        self.logger.info("Saving model and tokenizer...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.logger.info("Model and tokenizer saved.")
