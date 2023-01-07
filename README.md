# CC-Fraud-Detection
Comparing the usage of over-sampling on three approaches to tackle imbalanced data in less than two hours.

## Context & Content
The data can be obtained from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains only numerical input variables which are the result of a PCA transformation. 
This work will show that over-sampling is not needed to tackle highly imbalanced dataset. This work is divided into three approaches:
1.  Kaggle approach, which use Area Under the Precision-Recall Curve (AUPRC)
2.  Precision@Recall=0.90 approach, which try to maintaining a certain value of recall while obtaining the highest precision possible
3.  example-dependant cost-sensitive learning approach, which try to minimize the transaction amount lost due to fraud

## Basic Exploratory Data Analysis
1.  No missing data
2.  Obviously no correlation between the predictors since they are the result of a PCA transformation
![image](https://user-images.githubusercontent.com/48485276/201525477-c4351b4a-e8c4-40dd-952f-62c9ec40d6ec.png)
3.  There are clear differences in density on several variables, especially in v10, v11, v12, v14, v16, v17, and v18
4.  Based on point bi-serial correlation, all predictors are statistically correlate with fraud, but this might be due to the large number of observations. So, this correlation is actually useless

## Data term
•	Training		      : 80% of all observations

•	Cross-validation	: 10-folds CV constructed from training data

•	Testing			      : the remaining 20% observations

## Modelling
At first, I decided to compare a lot of supervised methods with different data pre-processing. But I think it takes a lot of time (maybe around 4 days with 8-core CPU) and consumes more than 30 GBs of hard drive. My alternative is using an AutoML since one can decide how many seconds the models to train themselves. I prepare two data, first data with no over-sampling and second data with borderline smote over-sampling. 

### 1st approach, the Kaggle approach
Since AutoML engine don’t have recall/precision metric, the datasets are trained to obtained the highest pr_auc (Precision-Recall Area Under Curve) possible. The first data is trained for 30 mins while the second data is trained for an hour. In 30 mins, AutoML fitted 71 models for the first data

![image](https://user-images.githubusercontent.com/48485276/201525535-ebc68a68-3eb8-45b6-85a8-b20026c6a116.png)

and 49 modes for the second.

![image](https://user-images.githubusercontent.com/48485276/201525540-ab54be32-3455-49cb-9139-f08b7b9d5b51.png)

Top 3 models are stacked ensemble models. StackedEnsemble_AllModels_4_AutoML_1 are constructed by 7 deep learnings, 2 distributed random forests, 51 gradient boosting machines, and 1 generalized linear model while StackedEnsemble_BestOfFamily_5_AutoML_2 only consist of 1 deep learning, 2 distributed random forests, 1 gradient boosting machines, and 1 generalized linear model.
Applying over-sampling on the second data clearly improved the cross-validation data as it achieved almost perfect aucpr. But when facing a new data, applying over-sampling didn’t help that much as the performance between those two datasets are quite the same. Implementing over-sampling was pretty much not helpful, the models were also need more time to train.

![image](https://user-images.githubusercontent.com/48485276/201525579-1f03ac87-57a9-4e4f-9c2a-45466bf584db.png)

### 2nd approach, the Precision@Recall=0.90
Suppose we've discussed with the business, and they said they really really want to detect at least 90% of fraudulent transactions. This means that they want a recall of 90%. That is non-negotiable.
So, what we want is a model that has a high precision when the recall is 90%, that is, we want to minimize how often the model tells us that a transaction was fraudulent when it wasn't. Otherwise, the bank will end up sending text messages to too many clients to verify their transactions.
To achieve this, we can change the threshold value. Generally, the threshold value is 0.5. I haven’t implemented this with cross-validation in this work. So, we’ll change the threshold based on the training data only, then apply the value to calculate recall on the testing data.

![image](https://user-images.githubusercontent.com/48485276/201525587-2d762797-fc1f-42de-8e6b-a204e5f8431a.png)

Based on the graph above, we can get recall at least 0.9 when the precision is as low as 0.117. Meanwhile, if we want at least 0.9 precision, we get 0.8 recall. The dashed line refers to a threshold that have the same value of recall and precision. The threshold values for these three lines are 0.002002002 for recall, 0.3293293 for precision, and 0.1121121 for balancing recall-precision. This means that out of all fraud transactions, our model classifies 90% as fraud and we have to verify 88.3% of our clients about their transactions that we suspect are fraudulent.

![image](https://user-images.githubusercontent.com/48485276/201525608-b7c9ab2e-7c0e-4d9b-a398-3349f08d3ea4.png)

Meanwhile, for the second data, we get a lower precision, only 0.041 by keeping recall at around 0.9. Maintaining precision at 0.9 gave us 0.856 recall and we were able to maximizing precision and recall at 0.875. We can see that applying over-sampling made the model underperformed as this gave us a lower precision.

### 3rd approach, the example-dependant cost-sensitive learning
Since we have the variable ‘amount’, we can change the threshold to minimize the total money lost due to fraud transactions. The threshold value is chosen using testing data. Here goes the first data:

![image](https://user-images.githubusercontent.com/48485276/201525622-5e42b656-d944-4e5f-827c-3d11195daf9a.png)

The best thing the company do is to loss its customer’s money worth $1985.52 by changing the threshold to 0.074. On the hand, on the second data, we got this graph below.

![image](https://user-images.githubusercontent.com/48485276/201525629-4a6e51e4-bba7-40e1-9c34-7eee05a8a65a.png)

We managed to only lost $1866.287 on the testing data. This result seemed slightly better than using the first data, perhaps over-sampling played a role in this approach?

## Conclusions
The values below were calculated using testing data and rounded to the first 4 digits.
![image](https://user-images.githubusercontent.com/48485276/201525652-f7038cf9-6214-41e2-a2bd-48a7ab3e0712.png)

Over-sampling didn't improve the model performance (aucpr and precision@recall), so implementing over-sampling to deal with imbalanced dataset is such a waste of time as it consumes more time to train the model. On the third approach, we need to validate the result with a Cost-Sensitive Resampling to verify it.

### Pros and Cons
Aurpr metric can’t be translated directly to businessmen, but it can be implemented with any probabilistic models

Precision@Recall is easy to understand but it’s difficult to tune the threshold with cross-validation, because the recall value may not reach the expected value in each fold. That’s why we tuned the threshold on holdout/testing data.

example-dependant cost-sensitive is another easy-to-understand metric. In addition to changing the value of the threshold, we can also force the model to classify transactions as fraud when the number of transactions is large. However, this will result in a decrease in the precision value. But there are perhaps three main groups of cost-sensitive methods that are most relevant for imbalanced learning; they are:
1.  Cost-Sensitive Resampling
2.  Cost-Sensitive Algorithms
3.  Cost-Sensitive Ensembles

which use case weight or class weight argument

