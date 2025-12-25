# Autisim_prediction_using_ML_project_
This project uses Machine Learning to predict Autism Spectrum Disorder (ASD) based on behavioral and demographic data. It involves an end-to-end pipeline covering data cleaning, exploratory analysis, handling class imbalances, and hyperparameter tuning of tree-based models.

Project Workflow
1. Data Cleaning & Engineering
Feature Selection: Removed non-predictive columns such as ID and age_desc.

Data Standardization: Fixed country name inconsistencies and handled missing values in ethnicity and relation by categorizing them as "Others".

Label Encoding: Categorical variables were converted to numerical format using LabelEncoder, and the encoders were saved as encoders.pkl for future inference.

2. Exploratory Data Analysis (EDA)
Outlier Detection: Identified outliers in age and result columns using IQR-based box plots.

Distribution Analysis: Visualized feature distributions and target class balance.

Correlation Mapping: Generated a heatmap to ensure no high multicollinearity existed between features.

3. Preprocessing
Outlier Treatment: Implemented a custom function to replace extreme values in numerical columns with the median.

Handling Class Imbalance: Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the minority class in the training dataset.

4. Machine Learning Models
The project evaluates and compares three powerful tree-based algorithms:

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

5. Hyperparameter Tuning
Optimization: Used RandomizedSearchCV to fine-tune model parameters like max_depth, n_estimators, and learning_rate.

Cross-Validation: Performed 5-fold cross-validation to ensure the best model generalized well to unseen data.

Persistence: The highest-performing model is saved as best_model.pkl.

--> Future Scope
Predictive System: Development of a user interface (like Streamlit) to take input features and provide real-time ASD predictions.

Performance Optimization: Further refinement of the model through feature engineering or deep learning approaches.
