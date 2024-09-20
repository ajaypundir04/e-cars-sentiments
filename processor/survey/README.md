# Electric Vehicle Feature Selection & Survey Generation

## Overview

This project focuses on analyzing the key features of electric vehicles (EVs) by processing textual data from multiple sources, selecting the most relevant features, and generating surveys based on those features. The workflow includes feature extraction, domain-specific categorization, applying weights to EV-related terms, and producing survey questions to gather user feedback on critical EV characteristics.

---

## Key Components

### 1. **Feature Selection**
The process of feature selection extracts the most significant features from text data, which can be derived from various sources, such as articles, reviews, or surveys related to electric vehicles. Feature selection focuses on identifying important EV-specific terms, including:

- **Battery Performance**
- **Range**
- **Charging Infrastructure**
- **Acceleration**
- **Emissions Reduction**

#### Workflow:

- **Textual Input Sources**: Text can be extracted from URLs, CSV files, or other text documents.
- **Text Processing**: Tokenization, filtering, and stemming are performed to clean the text for feature extraction.
- **TF-IDF**: We use Term Frequency-Inverse Document Frequency (TF-IDF) to identify terms that are important in the EV domain but less common in the general language context.

### 2. **Domain-Specific Categorization**

After feature extraction, the terms are categorized into different domains based on their relevance to electric vehicles. This categorization helps in grouping similar terms and focusing on key aspects such as performance, design, usability, and sustainability. 

#### Categories:

- **Performance**: Features like battery capacity, acceleration, torque, and power.
- **Sustainability**: Features related to emissions, eco-friendliness, and carbon footprint.
- **Usability**: Focused on user comfort, ease of use, and charging infrastructure.
- **Affordability**: Pricing-related features, including cost, value, and long-term savings.
- **Design**: Aesthetic features like look, style, and overall design.

#### Synonym Mapping:
To avoid redundancy, the system also maps similar terms to a unified feature name:
- "Range" → ["Mileage", "Distance"]
- "Battery" → ["Battery Life", "Battery Capacity"]
- "Charging" → ["Charging Time", "Charging Speed"]

### 3. **Domain-Specific Weights**

In addition to categorization, we apply **domain-specific weights** to certain keywords to prioritize terms that are especially important in the electric vehicle context. For example:

- Keywords like **battery**, **range**, **charging**, and **sustainability** are given higher weights since they are central to the EV experience.
- The system boosts occurrences of these terms to ensure they are prioritized during feature selection.

#### Boosting Weights:
- Terms found in the **electric vehicle keyword list** (e.g., "battery", "range", "charging") are given a weight boost during the feature counting process. This ensures that domain-relevant terms get higher scores and are included in the top feature list.

### 4. **Survey Generation**

Once the top features are identified, the system automatically generates survey questions that are designed to gather user feedback on the most relevant aspects of electric vehicles.

#### Survey Question Generation:
Survey questions are dynamically generated based on the selected features. For example:
- Feature: "Battery"
  - Question: "How satisfied are you with the battery performance of your electric car?"

#### Likert Scale:
Each survey question is presented with Likert scale response options, allowing users to rate their level of agreement or satisfaction:
- Likely
- Not Likely
- Neutral
- Very Likely
- Unlikely

### Example Workflow:

1. **Input Data**:
   - Input data is gathered from multiple sources, such as review files, websites, or survey data, to be processed for feature extraction.

2. **Feature Extraction**:
   - The system runs TF-IDF analysis and filters out irrelevant or generic terms while prioritizing domain-specific EV-related terms.

3. **Feature Categorization**:
   - Extracted features are categorized into predefined domains, such as performance, sustainability, or usability.

4. **Survey Generation**:
   - Based on the top features, a set of survey questions is generated, asking users to rate their satisfaction or the importance of these features.

---

## How to Use the System

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/electric-vehicle-feature-analysis.git
   cd electric-vehicle-feature-analysis
