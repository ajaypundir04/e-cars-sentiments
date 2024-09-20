import argparse
from handler.feature_handler import FeatureHandler  # Import the class from the other file

class FeatureExecutor:
    def __init__(self):
        """
        Initialize the FeatureExecutor class with the argument parser and FeatureHandler instance.
        """
        self.app = FeatureHandler()  # Initialize FeatureHandler

    def parse_arguments(self):
        """
        Parse command-line arguments using argparse and return them.
        """
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Run Feature Extraction from URLs or Files.")
        
        # Define arguments
        parser.add_argument('--mode', type=str, choices=['url', 'file'], default='file',
                            help="Mode to run the analysis: 'url' or 'file'. Default is 'file'.")
        parser.add_argument('--language', type=str, required=True,
                            help="Language code to use for analysis, e.g., 'EN', 'DE'.")
        parser.add_argument('--keyword', type=str, default='cars',
                            help="Keyword to search for in the analysis. Default is 'cars'.")
        parser.add_argument('--num_features', type=int, default=10,
                            help="Number of top features to extract and log. Default is 10.")

        # Parse the arguments
        return parser.parse_args()

    def execute(self):
        """
        Execute the feature extraction process based on the parsed arguments.
        """
        # Parse arguments
        args = self.parse_arguments()

        # Run the app based on mode
        if args.mode == 'url':
            self.app.process_from_url(args.language, args.keyword, args.num_features)
        elif args.mode == 'file':
            file_paths = ['stats/ev_china.md']
            self.app.process_from_file( args.keyword, args.language, args.num_features, file_paths)
        else:
            self.app.logger.error(f"Invalid mode: {args.mode}")


# Example usage when running the script directly
if __name__ == '__main__':
    executor = FeatureExecutor()  # Initialize the FeatureExecutor
    executor.execute()  # Execute the feature extraction process
