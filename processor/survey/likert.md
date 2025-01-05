# Electric car Likert Question Generation Model

This repository contains the code for a **Likert Question Generation Model** built using the T5 model, fine-tuned for generating Likert-scale questions from a given context. The workflow involves preprocessing data, fine-tuning the model, generating questions, and evaluating the model.

## Project Overview

The main goal of this project is to generate Likert-scale questions based on context passages, particularly focused on the electric vehicles (EV) industry. The process follows a series of steps:

1. **Data Preprocessing**: Prepare input-output pairs for the model.
2. **Model Fine-Tuning**: Fine-tune a pre-trained T5 model on your dataset.
3. **Question Generation**: Use the fine-tuned model to generate Likert-scale questions from passages.
4. **Evaluation**: Evaluate the model performance using Rouge metrics.
5. **Passage Summarization**: Use a BART model to summarize long passages.

## Workflow

The workflow consists of several key steps, which are visualized in the flowchart below:

![Workflow](https://example.com/path-to-your-image.png)  <!-- Replace with actual path to your image -->

### Key Steps in the Workflow

1. **Load Datasets**: The training and validation datasets are loaded from the `train.jsonl` and `valid.jsonl` files.
2. **Load Tokenizer and Model**: The T5 tokenizer and model are loaded using the `AutoTokenizer` and `AutoModelForSeq2SeqLM` from the Hugging Face `transformers` library.
3. **Preprocess Data**: The input-output pairs are formatted, and the data is tokenized.
4. **Fine-Tune T5 Model**: The model is fine-tuned using the `Trainer` API with specified training arguments.
5. **Generate Likert Question**: The trained model generates Likert-scale questions based on the passage context.
6. **Evaluate Model**: Rouge metrics are computed to evaluate the model's performance.
7. **Summarize Passages**: Long passages are summarized using the Facebook BART-large model.

## Installation

To set up this project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/likert-question-model.git
cd likert-question-model
