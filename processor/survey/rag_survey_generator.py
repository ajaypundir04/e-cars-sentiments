from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments
from datasets import Dataset
import faiss
import torch

class RAGSurveyGenerator:
    def __init__(self, model_name="facebook/rag-sequence-nq"):
        # Initialize RAG model, retriever, and tokenizer
        self.tokenizer = RagTokenizer.from_pretrained(model_name)
        self.retriever = RagRetriever.from_pretrained(
            model_name,
            index_name="custom",
            passages_path="data/knowledge_base.json",  # Preprocessed knowledge base
        )
        self.model = RagSequenceForGeneration.from_pretrained(model_name, retriever=self.retriever)
    
    def fine_tune(self, dataset_path, output_dir="rag-finetuned"):
        # Load dataset
        dataset = Dataset.from_json(dataset_path)

        # Tokenize data
        def preprocess_data(example):
            question = example["question"]
            context = example["context"]
            inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True)
            return {"input_ids": inputs["input_ids"], "labels": inputs["input_ids"]}
        
        processed_dataset = dataset.map(preprocess_data)

        # Fine-tune RAG model
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=8,
            save_steps=10_000,
            num_train_epochs=3,
            logging_steps=500,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def generate_survey_questions(self, query, num_questions=5):
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**inputs, num_return_sequences=num_questions, num_beams=5)
        questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return questions
