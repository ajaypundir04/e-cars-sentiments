class CategoryHandler:
    def __init__(self):
        """
        Initialize the feature categories with corresponding keywords.
        """
        self.categories = {
            'performance': ['speed', 'performance', 'efficiency', 'battery', 'power', 'range', 'acceleration'],
            'design': ['design', 'look', 'appearance', 'style', 'aesthetic', 'color', 'body', 'shape'],
            'usability': ['usability', 'ease of use', 'comfort', 'interface', 'features', 'controls', 'accessibility'],
            'affordability': ['price', 'cost', 'value', 'affordability', 'economy'],
            'safety': ['safety', 'airbags', 'brakes', 'security', 'emergency', 'crash', 'stability'],
            'charging': ['charging', 'charger', 'charge time', 'charging station', 'fast charge', 'plug'],
            'technology': ['technology', 'tech', 'innovation', 'software', 'updates', 'electronics', 'display'],
            'interior': ['interior', 'cabin', 'seats', 'dashboard', 'upholstery', 'spacious'],
            'maintenance': ['maintenance', 'service', 'repair', 'durability', 'longevity'],
            'environment': ['environment', 'emissions', 'eco-friendly', 'sustainability', 'green'],
            'warranty': ['warranty', 'guarantee', 'coverage'],
            'reliability': ['reliability', 'dependability', 'trust', 'confidence'],
            'noise': ['noise', 'quiet', 'sound', 'silent'],
            'brand': ['brand', 'manufacturer', 'reputation', 'name', 'legacy'],
            'luxury': ['luxury', 'premium', 'exclusive', 'high-end'],
            'resale': ['resale', 'depreciation', 'residual value'],
            'insurance': ['insurance', 'coverage', 'policy'],
            'connectivity': ['connectivity', 'Bluetooth', 'Wi-Fi', 'smartphone', 'apps', 'remote'],
            'autonomous': ['autonomous', 'self-driving', 'automation', 'driver assistance', 'AI'],
            'availability': ['availability', 'stock', 'supply', 'waiting time', 'delivery']
        }

    def categorize_feature(self, feature):
        """
        Categorize the feature based on predefined categories and keywords.
        
        Args:
            feature (str): A feature extracted from the data.
        
        Returns:
            str: The category of the feature (e.g., 'performance', 'usability', etc.)
        """
        feature_lower = feature.lower()

        for category, keywords in self.categories.items():
            if any(keyword in feature_lower for keyword in keywords):
                return category

        return 'general'
