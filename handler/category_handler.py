import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class CategoryHandler:
    def __init__(self, model_name="bert-base-uncased"):
        """
        Initialize the transformer-based model and tokenizer.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=21)  # 21 categories in total
        self.model.eval()  # Set the model to evaluation mode
        
        # Define categories and their labels
        self.categories = [
            'performance',          # Performance-related features (e.g., acceleration, speed)
            'design',               # Design-related features (e.g., aesthetics, body)
            'usability',            # Usability-related features (e.g., ease of use, interface)
            'affordability',        # Price and affordability-related features (e.g., cost, value)
            'safety',               # Safety-related features (e.g., airbags, stability)
            'charging',             # Charging-related features (e.g., fast charging, charging stations)
            'technology',           # Technology-related features (e.g., software, electronics)
            'interior',             # Interior features (e.g., cabin, dashboard, seats)
            'maintenance',          # Maintenance-related features (e.g., service, repair)
            'environment',          # Environmental features (e.g., eco-friendliness, emissions)
            'warranty',             # Warranty and guarantee features (e.g., coverage, service)
            'reliability',          # Reliability features (e.g., durability, dependability)
            'noise',                # Noise-related features (e.g., cabin quietness, sound)
            'brand',                # Brand and reputation-related features (e.g., manufacturer, legacy)
            'luxury',               # Luxury-related features (e.g., high-end features, premium)
            'resale',               # Resale value and depreciation
            'insurance',            # Insurance-related features (e.g., coverage, policy)
            'connectivity',         # Connectivity-related features (e.g., Bluetooth, Wi-Fi, apps)
            'autonomous',           # Autonomous driving features (e.g., self-driving, AI)
            'availability',         # Availability and delivery (e.g., stock, waiting time)
            
            # Additional categories for electric cars
            'battery',              # Battery-related features (e.g., battery life, charging time)
            'range',                # Range-related features (e.g., how far the car can travel on a single charge)
            'regenerative_braking', # Regenerative braking features (e.g., energy recovery)
            'charging_infrastructure',  # Charging infrastructure (e.g., availability of charging stations)
            'range_optimization',   # Range optimization (e.g., energy-saving modes, efficiency)
            'fleet_management',     # Fleet management (e.g., for commercial electric vehicles)
            'smart_navigation',     # Smart navigation (e.g., route planning for charging stops)
            'vehicle_to_grid',      # V2G (Vehicle-to-Grid) capabilities (e.g., power supply to grid)
            'energy_efficiency',    # Energy efficiency features (e.g., consumption per mile, regenerative systems)
            'solar_integration',    # Solar integration (e.g., solar roof panels for charging)
            'driving_modes',        # Driving modes (e.g., sport mode, eco mode)
            'app_integration',      # App integration (e.g., control and monitor car from a smartphone)
            'sustainability',       # Sustainability (e.g., eco-friendly materials, green technology)
            'charging_speed',       # Charging speed (e.g., fast charge vs standard charge)
            'integrated_assistants', # Virtual assistants (e.g., voice control, AI assistants)
            'smart_climate',        # Smart climate control (e.g., pre-conditioning, energy-saving HVAC)
            'vehicle_customization', # Vehicle customization options (e.g., interior, exterior)
            'autopilot_features',   # Autopilot and driver-assist features (e.g., lane assist, self-parking)
            'driver_feedback',      # Feedback for driver (e.g., range estimation, energy consumption data)
            'after_sales_service',  # After-sales services (e.g., service centers, customer support)
            'consumer_reports',     # Consumer reviews and reports on EVs (e.g., ratings, feedback)
            'charging_networks',    # Charging networks (e.g., Tesla Supercharger, public networks)
            'vehicle_security',     # Security features (e.g., anti-theft, remote locking)
            'insurance_rates',      # Insurance rates and premiums (e.g., EV-specific rates, discounts)
            'government_incentives', # Government incentives (e.g., tax credits, rebates for electric vehicles)
        ]

        
        # Create a mapping from category to index
        self.category_to_idx = {category: idx for idx, category in enumerate(self.categories)}

    def categorize_feature(self, feature):
        """
        Categorize the feature using a pre-trained transformer model.
        
        Args:
            feature (str): A feature extracted from the data.
        
        Returns:
            str: The category of the feature (e.g., 'performance', 'usability', etc.)
        """
        inputs = self.tokenizer(feature, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        
        # Convert predicted index to category label
        predicted_category = self.categories[predicted_class_idx]
        
        return predicted_category