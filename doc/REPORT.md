# Applicant Loan Prediction Using Random Forests and Decision Trees

## Saif Awan, Param Patel

**Abstract**

Every bank aims to restrict loans to applicants that they deem capable of repayment. This key function can be automated and performed by a machine learning model that is trained on the right dataset. This type of machine learning problem is a classification problem that can be handled by methods like logistic regression, decision trees, and random forests. As machine learning models start addressing more mission critical tasks, data scientists must address and mitigate the bias in their datasets. To overcome this bias we have studied IBM’s AI Fairness toolkit and created our own method for addressing bias in our model. In this project, we have trained our model using a Kaggle Dataset through a random forest classifier and decision tree. We then show how accurate the models are by running a test dataset through the models and documenting their accuracy. 

**Introduction**

Loaning money is the principal function of banks and all banks want to avoid loaning money to applicants who can not repay the loan. To ensure that loans are given to people who will not default, banks collect important data about applicants, such as Marital Status, Education, Income, CoApplicant Income, and Dependents. They also gather data about the loan that the applicant is requesting, such as Loan Amount and Loan Term. Then the bank takes all of these factors into account and decides whether to give an applicant a loan. We believe that we can create a machine learning model that automates this process for the bank and can predict loan eligibility automatically. This can allow applicants to apply for loans online rather than in person and verify their eligibility without coming into a bank. We devised two machine learning models, one using a Random Forest Classifier and one using a Decision Tree Classifier. Our results show that the Random Forest Classifier performs much better than the Decision Tree Classifier. 

**Related Work**

In our research we first looked for approaches to solving loan eligibility and loan prediction problems and found a paper titled Monetary Loan Eligibility Prediction using Machine Learning which use many machine learning algorithms to determine which one was the best fit, such as K-Nearest Neighbor, Naive Bayes, Decision Trees, and Logistic Regression. Their research concluded that all those algorithms were capable of achieving 76% to 80% accuracy rates [1]. We also came across a Towards Data Science article that performed loan eligibility prediction using Logistic Regression, Random Forests, XGBoost, and Decision Trees [2]. Based on these two articles and their methods we decided to create two models, first the basic Decision Tree model and then the Random Forest model. We found this second article to be particularly helpful in outlining the steps that we would need to take to do our entire project and borrowed from their methods for understanding the data. However, we differed from the article in our approach to our classifier models because they used Stratified K-Folds for all their models but we created our models without K-Folds and still achieved higher accuracy. 

**Data**

The data we worked with is a Loan Prediction dataset that we found on Kaggle. The data consisted of 614 rows of data with 13 columns. The 13 columns were the following variables: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoApplicantIncome, Loan Amount, and Loan_Amount_Term. However this data was not ready for processing and so we had to perform some preprocessing before putting it into the models. The first preprocessing step we did was identify all the null values and replace them. For categorical variables we replaced the nulls with the mode value. For ordinal and numerical variables we replaced nulls with the mean value. We found this technique in the Towards Data Science article mentioned in the previous section. For outlier treatment we used log normalization during preprocessing. For values that were non-numerical we replaced them with 1 or 0 if there were just two classifications (Gender). For variables with more than two classifications like dependents we just made the largest value “3” instead of “3+”.  

**Methods**

When we picked the project we knew that it was a classification problem and so we knew that we would have to work with models like Support Vector Machines, Logistic Regression, Decision Trees, Random Forests, and Boosting Methods. Our first step was to study the data and understand what our data told us about the target variable. We analyzed all the variables separately through box plots and distribution curves. Then we analyzed each variable with loan status and observed how loan status was different for each variable. We knew this previous step was very important because we wanted to use this to identify the bias in our data and see how we can address it. Our noteworthy observation here was that Male and Female applicants had a small but noticeable difference in loan acceptance. Male applicants were receiving loans 70% of the time and female applicants were receiving loans 65% of the time in our training data. Through our reading of the AI Fairness 360 tutorials and guides we found that the approach for bias mitigation was in three parts of the machine learning process: preprocessing, inprocessing, and postprocessing. We decided to perform our bias mitigation in the preprocessing so that we could use the default classifier algorithms provided by Python’s sklearn library. We also did this because we concluded that we did not want our model to be trained to include Gender as a delimiting factor for loan eligibility. 

**Experiment**

We did some experimenting with features and heat maps to see if we wanted to do some feature engineering. One feature we were particularly interested in was Total Income(ApplicantIncome+CoApplicantIncome). We hypothesized that applicants who had a high Total Income would be more likely to receive loans. To test this we added Total Income as a feature and then ran a heatmap(Fig 1.)  over all the features. Unfortunately, Total Income did not have a high correlation with Loan Status. However, we did see a trend that we were interested in which is that Total Income had very high correlation with Loan Amount. This clearly indicates that applicants who have a high total income are more likely to apply for larger loans which they might not be qualified for due to other variables. The other experiment that we did with our model was to remove the Gender feature to remove any bias that may arise from it. We are a little less certain about our results in this case because both of our models gave prediction accuracy values of 100%(Fig 2.). This could mean that removing Gender as a variable makes it easier for the classifiers to predict the test data. When deciding between machine learning models to implement we ultimately decided to use decision trees and random forests because we believe using both is a good way to demonstrate how random forests are more powerful than decision trees. Our results reflect this observation. 

<img width="345" alt="Screen Shot 2021-11-28 at 10 30 07 PM" src="https://user-images.githubusercontent.com/60564460/143804394-aa0e2898-5ae6-4762-a5a6-daad04fd7057.png">

Figure 1. Heatmap with Total_Income Feature

<img width="404" alt="Screen Shot 2021-11-28 at 10 31 33 PM" src="https://user-images.githubusercontent.com/60564460/143804505-44fe8f84-4640-4159-ba47-8acf0a26a531.png">

Figure 2. Random Tree Accuracy with Gender Feature Removed

**Conclusion**

In this project we proposed two machine learning models to solve the loan eligibility problem. Our models performed very well with the Random Forest Classifier yielding 82.83% accuracy and the Decision Tree Classifier yielding 75.76% accuracy. With more time and resources this model could be improved with more features and a much larger dataset for training. We also believe that there may be better methods for bias mitigation that could be applied but the project team was unable to implement. Commercial banks have implemented their own proprietary algorithms for solving this problem beyond the generic  Python sklearn machine learning models which would have been useful in our study if we had access to them.  

**Works Cited**

[1] 	S, Ramya, et al. Monetary Loan Eligibility Prediction Using Machine Learning, 11 , no. No.07, 2021. 
https://ijesc.org/upload/0e4caa4fba55382053b74cf62fe7e8aa.Monetary%20Loan%20Eligibility%20Prediction%20using%20Machine%20Learning%20(1).pdf​​

[2] 	Bhandari, Mridul. “Predict Loan Eligibility Using Machine Learning Models.” Medium, Towards Data Science, 19 Oct. 2020, https://towardsdatascience.com/predict-loan-eligibility-using-machine-learning-models-7a14ef904057. 


[3]	Janeway, Nicole. Handle Outliers with Log-Based Normalization, https://www.nicolejaneway.com/python/outliers-with-log/. 

