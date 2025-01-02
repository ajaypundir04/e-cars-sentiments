import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Download NLTK resources
nltk.download("stopwords")
from nltk.corpus import stopwords

def analyze_text_file(file_path):
    """
    Analyzes the text file to identify key features, trends, or patterns.
    Returns a list of extracted topics or frequent terms.
    """
    try:
        # Read the file
        with open(file_path, "r") as file:
            text = file.read().lower()
        
        # Preprocess the text
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        words = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in stop_words]
        
        # Use CountVectorizer to identify frequent terms
        vectorizer = CountVectorizer(max_features=10)  # Adjust max_features as needed
        vectorizer.fit(filtered_words)
        features = vectorizer.get_feature_names_out()
        
        return list(features)
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return []

def generate_questions_based_on_features(features, num_questions=5):
    """
    Generates survey questions based on identified features.
    """
    try:
        # Load T5 model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        questions = []
        for feature in features:
            # Create a prompt for each feature
            prompt = f"Generate {num_questions} survey questions about {feature} in electric cars."
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            
            # Generate questions
            outputs = model.generate(
                input_ids, max_length=150, num_beams=5, temperature=0.7
            )
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            questions.extend(decoded_output.split("\n"))
        
        return questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

# Example Usage
if __name__ == "__main__":
    data_file = "processor/survey/fine_tune/stats.md"  # Replace with your dataset file
    
    # Analyze the text file
    features = analyze_text_file(data_file)
    print(f"Identified Features/Trends: {features}")
    
    # Generate survey questions
    if features:
        questions = generate_questions_based_on_features(features, num_questions=3)
        print("\nGenerated Survey Questions:")
        for i, question in enumerate(questions, start=1):
            print(f"{i}. {question}")
