import matplotlib.pyplot as plt

class SurveyPrinter:
    @staticmethod
    def plot_feature_frequencies(features_with_occurrences):
        """
        Plot the occurrences of top features with their frequencies.

        Args:
            features_with_occurrences (list): List of tuples containing (feature, occurrence).
        """
        # Check if features_with_occurrences is wrapped in an extra list and flatten if necessary
        if len(features_with_occurrences) == 1 and isinstance(features_with_occurrences[0], list):
            features_with_occurrences = features_with_occurrences[0]

        # Ensure the input is a list of tuples with two elements: (feature, occurrence)
        if not all(isinstance(item, tuple) and len(item) == 2 for item in features_with_occurrences):
            raise ValueError("features_with_occurrences must be a list of tuples containing (feature, occurrence)")

        # Extract the features and their frequencies from the input
        features = [feature for feature, occurrence in features_with_occurrences]
        frequencies = [occurrence for feature, occurrence in features_with_occurrences]

        # Create a figure and axes for the frequency plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot a bar chart for the feature occurrences
        ax.barh(features, frequencies, color='skyblue')
        ax.set_xlabel('Frequency')
        ax.set_title('Top Features and Their Occurrences')

        # Adjust the layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    @staticmethod
    def plot_survey_questions(survey_questions):
        """
        Display the survey questions with only the question text and numbering.

        Args:
            survey_questions (list): List of survey questions for the features.
        """
        # Calculate the vertical spacing based on the number of questions
        num_questions = len(survey_questions)
        spacing = 1 / (num_questions + 1)  # Dynamic spacing to prevent overlap

        # Create a new figure for the survey questions
        fig, ax = plt.subplots(figsize=(12, 8))

        # Add the survey questions with appropriate spacing
        for i, question in enumerate(survey_questions, start=1):
            # Format the question with numbering
            question_text = f"{i}. {question}"

            # Add each question as text with dynamic spacing
            ax.text(0, 1 - i * spacing, question_text, va='top', ha='left', fontsize=10, transform=ax.transAxes)

        # Hide axes as we are just displaying text
        ax.axis('off')

        # Set the title
        plt.title("Survey Questions", fontsize=14)

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()
