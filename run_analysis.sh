#!/bin/bash

echo "Starting Sentiment Analysis and Data Processing..."

# URL-Based Analysis
echo "Executing English Data Analysis..."
python -m executor.executor --mode=url --language=EN

echo "Executing German Data Analysis..."
python -m executor.executor --mode=url --language=DE

echo "Executing Chinese Data Analysis..."
python -m executor.executor --mode=url --language=CN

echo "Executing Norwegian Data Analysis..."
python -m executor.executor --mode=url --language=NG

# Sentiment Analysis using Transformer
echo "Performing Sentiment Analysis using Transformer..."
python -m executor.executor --mode=transformer --language=EN

# Cosine Similarity Calculation
echo "Computing Cosine Similarity between different regions..."
python -m executor.executor --mode=similarity --language=NG

# Review Sentiment Analysis
echo "Analyzing sentiment from review data..."
python -m executor.executor --mode=file --language=EN

# Survey Generation
echo "Generating a survey with 10 features..."
python -m executor.survey_creator --mode=file --language=EN --num_features=10

# Survey Response Analysis
echo "Analyzing survey responses..."
python -m executor.executor --mode=survey --language=EN

# Prediction for Electric Car Adoption
echo "Predicting electric car adoption trends..."
python -m executor.executor --mode=prediction --language=EN

echo "All tasks completed successfully!"
