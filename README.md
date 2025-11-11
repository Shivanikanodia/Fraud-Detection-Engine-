
## Identifying Fraudulent Transactions:

### OBJECTIVE:

The objective of this project is to build and productionize a predictive model that classifies whether a given transaction is fraudulent or legitimate. Detecting fraudulent transactions is critical in the financial and banking sectors to minimize monetary losses, protect customers, and maintain trust.

### GOAL:

Maximize fraud detection accuracy (high recall) while maintaining a low false positive rate (balanced precision-recall trade-off). 
To Provide an interface to Risk Team, banking system, or customer support team — whoever is using the model.

---- 
### STEPS:

1. Data Collection and Preperation:
2. Exploratory Data Analysis:
3. Data Preprocessing and Feature Engineering:
4. Model Selection, Model Traning and Model Evaluation:
5. Model Deployement and Model Hosting:
6. Model Perfomance Monitoring and Data drift tracking: 

## Data Collection and Preperation :

- The dataset had 7 lakh rows and 28 features, with mixed data type. We received it in JSON format and processed it using JSONlines library. 
- The dataset was checked for missing values using the isnull().sum() function and empty strings and whitespace characters using  regular expressions (Regex) for  EcoBuffer, 'posOnPremises'.  
- Used nunique function in python to detect unique value distribution and category diversity.
- Used pd.to_datetime to convert datetime coulmn stored as string into date object (Transactiondatetime, AcountOpenDate, ExpiryDate). 
- For numerical data distribution, calculated skewness, visualized using KDE and histrograms, for rightly skewed trasanctions used log1p to make data normalised. 

---

## Data Visualisation - EDA: 

1. Build Bar charts to Visualise Merchants Categories based on (Fraud Rate and Fraud Amountt), Channe - Pos_Entry_Model , (Spending patterns and High Velocity in transactions), Temporal trends (Transaction_hour, Is transaction happening at night and Time Since last transaction)  


**INSIGHTS:** 
  
1. IN AND OUT showed a high fraud rate even at moderate transaction volumes, indicating vulnerabilities at specific outlets.

2.Uber, Lyft, Walmart, Target, Sears, and Amazon consistently showed loss amounts ranging from $10K to $35K with fraud rates between 2–5%, suggesting that fraudsters may target major brands with high transaction volumes and heavy traffic.
3. Fraud patterns varied by time and merchant — higher fraud amounts occurred between 12 AM–6 AM for Uber and Lyft, while Walmart, Target, and Sears experienced losses of $20K–$35K, highlighting off-hour vulnerabilities in e-commerce and transportation platforms

<img width="786" height="600" alt="Screenshot 2025-11-11 at 12 19 22" src="https://github.com/user-attachments/assets/8ca7df68-7ec1-4326-aed4-6ddadb2efeba" />

---

## FEATURE ENGINEERING:

1. Merchants data was significantly higher then usual, 2000 uniques merchants. Grouped mechant name as "Others" for merchant frequency less than 250. This reduced high cardinality, improved model efficiency, and prevented overfitting. It also helped the model focus on merchants with enough data to learn meaningful fraud patterns. 

<img width="833" height="453" alt="Screenshot 2025-10-16 at 10 08 05" src="https://github.com/user-attachments/assets/511e2a0f-c948-427c-85b4-f28eb06e407f" />

<img width="778" height="448" alt="Screenshot 2025-10-16 at 10 08 10" src="https://github.com/user-attachments/assets/d63cb34a-8498-44eb-be4e-19dbddde1dfc" />

<img width="766" height="460" alt="Screenshot 2025-10-16 at 10 08 16" src="https://github.com/user-attachments/assets/9ff0b5a3-d19e-4bff-bb76-222d30cd3278" />

2. Fraud Detection dataset is usually highly imbalanced, to solve this created data preoprocessing pipeline and handlled class imbalance effectively. To Make Preprocessing Modular and Reproducible, used coulmn transformers inside pipeline. This allows seperate preprocessing for numerical, categorical, log_transformed and binary values, avoiding data leakage.

-----

### Model Selection, Model Traning and Model Evaluation:

Logistic Regression was selected as base model as its more interpretable and provide clear coefficents for feature importance. 

Created sklearn logisitc pipeline has encapsulated coulmn transformer (Scaling and preprocessing), SMOTE to handle class imbalance, by creating synthetic samples of minority class and Logistic Regression serves as the final classifier with max_iter=1000 for convergence. 

Pipeline allow training on raw data X_train, y_train and testing on X_test and y_test. 

Stored performance metrics from log model like classfication report (F1, Precision and Recall) confusion matrix . With recall of 70% , model did a good job capturing 70% of the fraudulent transaction i.e minimizing false negative and mazimizing Recall. While high number of false positives (>8000) were detected at thresold of 0.5. We also checked by improving thresold little which improved our precision. 

Because catching even a small fraction of actual frauds among millions of legitimate transactions is already a major win.

Next, Moved to Xgboost for more complex model, implemented classifier using similar preoprocesor using scale_pos_weight, n_estimators, Max_depth, col_sample, sub_sample. 

##  Model Deployement and Model Hosting:

XGBoost performed notably better than the Logistic Regression model due to its ability to capture non-linear relationships and complex feature interactions that linear models somestime fail to capture . It achieved a recall of 0.69, indicating a balanced trade-off between identifying true positives and minimizing false negatives.

Saved Xgboost Model as pkl as best model as its gives a balance between precison and recall,  created a lightweight API or service that can accept new input and return predictions immediately using FlaskAPI and deployed on AWS Elastic beanstalk for monitoring logs, thresolds and drifts in performance. 

The pipeline ensures automation, consistency, and high interpretability — making it suitable for catching fraud in real time. 

 <img width="957" height="630" alt="Screenshot 2025-10-07 at 17 05 53" src="https://github.com/user-attachments/assets/9846a1c8-f6b1-4eb7-ad24-e4551d7285bb" />


 <img width="1320" height="324" alt="image" src="https://github.com/user-attachments/assets/4a72641e-b71f-4749-ac45-528fc6bde1b3" />
 

<img width="610" height="120" alt="Screenshot 2025-10-07 at 17 06 54" src="https://github.com/user-attachments/assets/ab5053e9-4cdb-446d-9417-00a87fcf3a16" />


<img width="1234" height="906" alt="image" src="https://github.com/user-attachments/assets/d3687817-566d-4d29-94bf-b4a1e58a0c6e" />

-----

**Data Sanity and quality Checks and Data Validation**:

Checked If data was transformed correctly using name_steps/Features_name after preprocessing.
Checked  actual vs predicted values, ensured doing SMOTE before splitting into train/test.
Applying StandardScaler or LabelEncodermanually before the split.
Ensure scale_pos_weight and Evaluated with metrics focused on PR-AUC , Recall at top k (top 500 predictions), Precision at different threshold and tuned threshold to maximize F1. 

------

### SHAP for feature contribution towards prediction:

This plot ranks features by their average absolute SHAP value, which means:

<img width="783" height="860" alt="image" src="https://github.com/user-attachments/assets/e28f773c-36b0-4c38-9856-9969e41d803f" />

Model is most driven by merchant category, transaction amount and spending patterns, and whether the card is present during the transaction — likely important indicators of fraud or transaction legitimacy.

**transactionAmount:** : Red points (high amounts) → mostly on the right → large amounts increase predicted risk (e.g., higher chance of fraud).

Blue points (low amounts) → on the left → lower amounts reduce the probability of fraud.

**cardPresent:** : Blue (card not present) on the right → non-present cards increase risk (typical of online fraud).

Red (present) on the left → card present reduces risk.

**currentBalance:** : Red (high balance) → left → higher balances reduce risk.

Blue (low balance) → right → low balances increase risk.

**merchantCategoryCode_LE and merchantName_FE:**: Both show wide spread → certain merchants or merchant categories strongly influence whether a transaction is classified as risky.

---

### TOP PREDICTORS OF FRAUD:

**Merchants and Peak Hours to Watch out:**

Uber, Lyft, Ebay.com, Walmart, discount, Gap and Sears consistently appeared in list where Fraud transaction volume and  fraud counts were high. This evidented from the temporal analysis where hours like 12:00 AM, 01:00 AM and 03:00 AM were targeted mostly and these specific merchants showed fraudulent activity indicating low monitoring hours or bot testing.

### FUTURE WORK:

- Working on creating AI agent Interface which help risk team and customers to input Case details like Transaction_id,  Amount, Merchant_name, Transaction_Hour, Merchant_Location, Channel to detect fraud. AI would sends those inputs to your API Endpoint and will respond based on model behaviour, ensuring sensitivty, data privacy and compliance.
- I’d set up model monitoring with tools like EvidentlyAI to track drift, precision, and recall over time.

