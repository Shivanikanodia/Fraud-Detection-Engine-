
### FRAUD DETECTION ENGINE (TRAIN, TEST, DEPLOY AND CONTAINERIZE):

#### OBJECTIVE:

The objective of this project is to Build and Productionize a ML Engine that classifies whether a given transaction is Fraudulent or Legitimate. Detecting fraudulent transactions is critical in the financial and banking sectors to minimize monetary losses, protect customers, and maintain trust.

#### GOAL:

Maximize fraud detection accuracy (high recall) while maintaining balanced precision-recall trade-off)
To Provide an interface to Risk Team and Monitoring Teams to input details and receive fraud risk score, decisions and AI driven explanaitions using SHAP.  

---- 
#### STEPS:

1. **Data Collection and Preparation:**
2. **Exploratory Data Analysis:**
3. **Data Preprocessing and Feature Engineering:**
4. **Model Selection, Model Building, Model Training and Model Evaluation:**
5. **Model Deployement and Model Hosting on Docker:**
____

**THE FOLLOWING ASSUMPTIONS HAVE BEEN MADE ON  DATA:**

- I've assumed that transaction timestamps, Account IDs, and amounts are accurate.
- Labels for “fraud” and “non-fraud” are correctly assigned.
- Data from different Channels **POS Modes** are aligned by same AccountID.
- Model errors are random, not systematically biased toward certain user groups.
- Real-time transactions reflect similar dynamics as training data.
- No major policy or product changes occurred mid-dataset that affect fraud patterns.

---- 

## Data Collection and Preparation:

- The dataset contained 700,000 rows and 28 features with mixed data types. It was received in JSON format and processed using the JSONLines library.
- The dataset was checked for missing values using the isnull().sum() function and for empty strings and whitespace characters using regular expressions (Regex).  
- The nunique() function in Python was used to detect unique value distribution and category diversity.
- The pd.to_datetime() function was applied to convert datetime columns stored as strings into proper date objects (TransactionDateTime, AccountOpenDate, ExpiryDate).
- For numerical data distribution, skewness was calculated and visualized using KDE plots and histograms. For right-skewed transaction data, the log1p transformation was applied to normalize the distribution.

---

## Exploratory Data Analysis: 

Built bar charts to visualize merchant categories by fraud rate and fraud amount, analyze channels (Pos_Entry_Mode),  and high-velocity patterns, and study temporal trends like transaction hour, night-time activity, and time since last transaction.


**INSIGHTS:** 
  

- IN AND OUT showed a high fraud rate even with moderate transaction volume.

- Uber, Lyft, Walmart, Target, Sears, and Amazon had losses of $10K–$35K with 2–5% fraud rates, indicating fraud focus on major brands.

- Higher fraud occurred between 12 AM–6 AM for Uber and Lyft, and Walmart, Target  and Sears showed $5-10K of losses for each hour, requiring strong monitoring and verification systems.

-  Some Account Numbers appeared consistently among these hours for similar merchants, Requires deliberate monitoring and strong verification.  

<img width="786" height="600" alt="Screenshot 2025-11-11 at 12 19 22" src="https://github.com/user-attachments/assets/8ca7df68-7ec1-4326-aed4-6ddadb2efeba" />

---

## FEATURE ENGINEERING:

- The dataset contained over 2,000 unique merchants, leading to high cardinality. Merchant names with a frequency of fewer than 250 transactions were grouped under “Others”. This reduced dimensionality, improved model efficiency, and prevented overfitting, allowing the model to focus on merchants with sufficient data to learn meaningful fraud patterns.

- Since fraud detection datasets are highly imbalanced, a data preprocessing pipeline was created to handle class imbalance effectively. To ensure modularity and reproducibility, ColumnTransformers were used within the pipeline, enabling separate preprocessing for numerical, categorical, and binary features while avoiding data leakage.

-----

## Model Selection, Model Traning and Model Evaluation:

- To reduce data leakage, transactions were split using a time-based strategy (train on earlier dates, test on later dates), ensuring the model does not
learn from future information.
-Stratified train-test split and scale_pos_weight = (# non-fraud / # fraud) in XGBoost for imbalance data.

### **Model Training:**

Two datasets were explored during development.

- Initial exploratory modeling was conducted on a transaction-only dataset. (01_eda_and_prototype_model.ipynb)
  to understand merchant, time, account level activity and channel-level fraud patterns.

- The final production model was trained on an enriched dataset that includes
  both transaction-level and user-level features (demographics, location,
  distance-based features), which significantly improved recall and stability. (02_production_training_pipeline.ipynb)

Only the enriched dataset and corresponding pipeline are used in production.

Multiple Models were evaluated: Logisitc Regression, Random Forest, Gradient Boosting Decision trees and Xgboost. 

Xgboost was selected as the base model due to its interpretability and ability to deal with complex features.  Built a Scikit-learn pipeline that encapsulated, ColumnTransformer for scaling and preprocessing.

- Threshold Strategy:
The fraud classification threshold is set to 0.7 instead of the default 0.5
to prioritize high recall while maintaining acceptable precision for manual review capacity.


<img width="427" height="180" alt="Screenshot 2025-12-24 at 16 00 08" src="https://github.com/user-attachments/assets/a8f7b0b8-f088-4fbe-8af4-0dc730037c37" />


##  Model Deployment and Model Hosting:

- Saved the XGBoost model as a .pkl file since it provided the best balance between precision and recall. 
- Developed a lightweight API service to accept new inputs and return predictions instantly and packaged using dockerfile to allow users to access host server from anywhere.
-  It preps the input data, trains a model, and  return predictions.
- The FastAPI layer strictly validates schema and enforces feature ordering to prevent training–serving skew using pydantic, and Containerized the app.
  

<img width="1282" height="430" alt="image" src="https://github.com/user-attachments/assets/5e837d58-88b0-4a6a-8375-fbf5083d4a6a" />

---

#### Fraud Detection UI:

<img width="1161" height="590" alt="Screenshot 2025-12-25 at 23 35 23" src="https://github.com/user-attachments/assets/f9e69fae-a357-455c-a0ae-460d26a10519" />

The UI Takes User inputs on transaction_amount, Merchant Category, Transaction_time, Zip Code and State, which sends the request to predict @post at endpoint, which then return backs to predictions using Xgboost model (using preprocessing pipeline). 

#### Results:

<img width="1038" height="302" alt="Screenshot 2025-12-25 at 23 34 14" src="https://github.com/user-attachments/assets/07a30532-4e44-4288-87e5-563949ba3968" />

The Model works on engineering features and returns predictions based on logic provided.(Depends on business trade-off, Cost, Nature of Problem and Priorities) 

### FUTURE IMPORTANT:

- Working on creating AI agent Interface which help risk team and customers to list important features like Transaction_id, Transaction_Amount, Merchant_name, Transaction_Hour, Merchant_Location, Transation Channel.  
- The AI agent will not generate explanations directly. Instead, SHAP feature attributions will be computed by the model, and an open-source LLM will convert these structured outputs into human-readable explanations while preserving compliance and data privacy.

**Monitoring & Observability:**
- Data drift detection using Population Stability Index (PSI)
- Performance monitoring using Recall@K and Precision over rolling windows
- API monitoring: latency, error rate, model version, and prediction logs

---

### Prerequisites

**1. Make sure you have the following installed:**
Python 3.9+

pip

Git

(Optional) Docker

**2.Check versions:**

python --version

pip --version

### How to Run the Project (Local):

🔹 1. Clone the Repository:

git clone https://github.com/shivanikanodia/Fraud-Detection-Engine-.git

cd Fraud-Detection-Engine-

🔹 2. Create & Activate Virtual Environment:

python -m venv venv

source venv/bin/activate    # Mac/Linux

venv\Scripts\activate       # Windows

🔹 3. Install Dependencies:

pip install -r requirements.txt

🔹 4. Run the Application:

python apps.py

this is a Flask API app, open your browser at: http://127.0.0.1:3000

5. Run with Docker:
   
 - Build Docker Image
  docker build -t fraud-detection-engine .

- Run Container
  docker run -p 3000:3000 fraud-detection-engine

### Model Details: 
Algorithm: XGBoost
Output: Binary fraud classification
Model file: fraud_xgb_pipeline.joblib
Feature importance: xgb_feature_importance_gain.csv

### Notes:
Virtual environments (venv/) are excluded from Git using .gitignore
