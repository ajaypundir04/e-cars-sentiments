import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def fetch_electric_car_data():
    """Fetch articles about electric cars from the internet."""
    url = "https://en.wikipedia.org/wiki/Electric_car"  # Example source
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract paragraphs from the article
        paragraphs = soup.find_all('p')
        text = "\n".join([para.text for para in paragraphs[:5]])  # Limit to 5 paragraphs
        return text
    else:
        raise Exception("Failed to fetch data from the web.")

def extract_features(text):
    """Extract key features from the text."""
    # Simplify to return sentences containing keywords for now
    features = []
    keywords = ["battery", "range", "charging", "cost", "sustainability"]
    for line in text.split("."):
        if any(keyword in line.lower() for keyword in keywords):
            features.append(line.strip())
    return features

def generate_survey_questions(features):
    """Generate survey questions using a pre-trained transformer pipeline."""
    question_generator = pipeline("text2text-generation", model="google/t5-small-ssm-nq")
    questions = []
    for feature in features:
        # Generate a "How likely" question based on the feature
        result = question_generator(f"Generate a 'How likely' survey question about: {feature}", max_length=64)
        questions.append(result[0]['generated_text'])
    return questions

# Main script
if __name__ == "__main__":
    try:
        print("Fetching data about electric cars...")
        data = fetch_electric_car_data()
        
        print("Extracting key features...")
        features = extract_features(data)
        print(f"Features extracted: {features}")
        
        print("Generating survey questions...")
        questions = generate_survey_questions(features)
        
        print("\nSurvey Questions:")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
    except Exception as e:
        print(f"An error occurred: {e}")
