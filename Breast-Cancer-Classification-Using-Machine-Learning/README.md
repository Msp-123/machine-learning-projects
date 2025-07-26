# Breast-Cancer-Classification-Using-Machine-Learning
- Created a machine learning model that classifies whether breast cancer is Malignant or Benign.
- Analyzed dataset with 569 observations and 33 columns
- Processed the data for making it suitable of better machine learning
- Optimized Logistic, and Random Forest Regressors using RandomSearchCV and cross validation to reach the best model.
- Made a sample prediction for a given observation

## Dataset Information
Dataset contains 32 features that are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

![image](https://user-images.githubusercontent.com/83719212/230721730-5ada8776-0fd0-4e33-93ee-43ad2c6f13f6.png)


## Data Cleaning
After collecting the data, I needed to clean it up so that it was usable for our model. I made the following changes to the variables:

- Removed unnecessary columns
- Encoded categorical variables
- Removed features with multicollinearity
- Extracted features and target
- Standardized the dataset

![image](https://user-images.githubusercontent.com/83719212/230721937-86c25ae6-e89d-4638-b246-4c2cf1edd441.png)



## Model Building
First I split the data into train and tests sets with a test size of 20%.

I tried two different models and evaluated them using Accuracy score. Fine tuned model with best accuracy for better machine learning process.

Machine Learning models:

- Logistic Regression – chose this model because our problem is binary classification
- Random Forest – with default parameters


## Model performance
The Logistic regression model outperformed the Random forest model on the test and validation sets.

Random Forest : Accuracy = 98.24 %
Logistic Regression: Accuracy = 95.61 %
Chose Logistic regression as final model this problem and fine tuned its hyperparameters.

Tuned Logistic regression : Accuracy = 98.24 %, Precision = 100 %, Recall = 94.47 %, F1 Score = 97.29 %

![image](https://user-images.githubusercontent.com/83719212/230721981-5f8c3975-b55c-474b-93f8-958cb6717549.png)

- The tuned model has same accuracy as default logistic regression model. 
- Logistic regression does not really have any critical hyperparameters to tune.
