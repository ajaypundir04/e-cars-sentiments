from transformers import pipeline

class InferenceModel:
    def __init__(self, model_path="./fine_tuned_ev_model"):
        # Load fine-tuned model
        self.model_pipeline = pipeline("text-classification", model=model_path)

    def predict(self, text):
        """
        Predict the Likert scale score for a given input text.
        """
        return self.model_pipeline(text)
