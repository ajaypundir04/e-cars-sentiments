from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

class ModelTrainer:
    def __init__(self, model_name="julian-schelb/rup-answer-option-likert-scale"):
        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3):
        """
        Fine-tune the model with the provided train and validation datasets.
        """
        # Define training arguments
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
        return trainer
