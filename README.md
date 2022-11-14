# **PROPOSAL**
### Group 11 - Param Patel, Saif Awan
### CS301-103
### Topic - Loan Approval Prediction
### Project Proposal
**What is the problem that you will be investigating? Why is it interesting?**

We're working on a supervised binary classification challenge. The goal is to train the best machine learning model in order to evaluate a loan applicant's eligibility for a loan and lower the risk of future loan defaults for a financial institution. . We will be trying to create an algorithm that will help determine whether a person is eligible for a loan or not. This is an interesting problem to solve because everyone needs loans and banks already implement some method of loan exclusion. If we could develop a better algorithm or just learn how existing algorithms likely work that would be a great asset to our understanding of the banking industry. 

**What reading will you examine to provide context and background?**

We have compiled a list of academic papers on loan prediction and risk management for banks. We want to get familiar with all the features that banks take into consideration when approving applicants and see if we can find a better set of features, preferably a smaller set of features.

**What data will you use? If you are collecting new data, how will you do it?**

We found a loan prediction data set on Kaggle that has data on 11 features relating to loan eligibility, including: income, marital status, credit history, employment type, etc. We also found a loan prediction dataset from a paper which had around 20 features which we might use if time allows.

**What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations?** 

Machine learning models will be able to detect the patterns of a good/bad candidate after all of the exploratory data analysis, purification, and dealing with any oddities we may discover along the way. We can utilize Random Forest because it is a tree-based ensemble model that improves the model's accuracy. It builds a powerful forecasting model by combining a huge number of Decision trees. We will have a set of decision trees, each one randomly testing one of the eleven possible features. The median of all forecasters or the mean of all predictors is the final prediction class.

**How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?**

We will evaluate our results against historical data on loan defaults. This will be a qualitative assessment of whether our algorithm and feature choices are accurate at predicting loan eligibility.
