# Project - Prediction of Starbucks Customers' Loyalty Status

## Contribution
| Student Name | Student ID | Contribution |
| --------| ---------|-------- |
| Ooi Li Yoong  | 1211306826 | 25% |
| Heng Peng Yong  | 1211306514 | 25% |
| Pang Shi An  | 1221303733 | 25% |
| Teoh Chee Hong  | 1221303824 | 25% |

## Data Source
https://www.kaggle.com/datasets/mahirahmzh/starbucks-customer-retention-malaysia-survey

## Introduction

In today's competitive Malaysian market, understanding customer behavior and maintaining customer loyalty are key to business success. Starbucks, a global leader in the coffee industry, is always looking to improve its customer retention strategies. The ability to accurately identify loyal customers and understand the factors that influence loyalty can provide valuable insights to enhance customer experience and improve customer retention.

This study aims to develop predictive models to categorize Starbucks customers into loyal and non-loyal customers based on various characteristics such as income, time spent at Starbucks, and other relevant characteristics. By utilizing machine learning techniques, we will create models that accurately identify loyal customers, thus enabling Starbucks to effectively adapt its marketing strategies.

This process consists of several key steps, starting with data cleaning to ensure the quality and completeness of the dataset. Then, we will construct and evaluate different classification models under various experimental settings. Our evaluation criteria include accuracy, precision, recall, and F1 score to ensure a comprehensive performance evaluation.

By identifying the best performing models, we can provide actionable insights to Starbucks. We can reward loyal customers to keep them loyal, while developing targeted programs to increase interest from non-loyal customers. This approach will not only increase customer satisfaction, but also strengthen Starbucks' market position by fostering long-term customer relationships.

## Dataset

For the purpose of this project, the area of study is customer loyalty in the coffee industry, specifically Starbucks Malaysia. Understanding the factors that contribute to customer loyalty is crucial for Starbucks to maintain its competitive advantage and increase customer satisfaction. This study aims to gain insights into customer behavior, which will help Starbucks to adjust its marketing strategy to improve customer loyalty.

The Starbucks Customer Retention Malaysia survey dataset publicly available on Kaggle contains various variables related to customer experience and behavior at Starbucks. These variables help predict customer loyalty and understand the key drivers of customer retention. The target variable in this dataset is customer loyalty, which is categorized as both loyal and non-loyal.

### Key Attributes of the Dataset

| Demographic information  | Customer Experience | Customer Behavior |
| ------------- | ------------- | ------------- |
| Gender  | Friendly service  | Spend per visit |
| Age  | Convenient location  | Frequency of visits |
| Occupation| Quality of products | Preferred products |
| Income | Ambiance | Time spent in store |
| | | Price Range |

The Starbucks customer satisfaction survey dataset comprises a total of 122 responses. The data collected includes both numerical and categorical information, which can be summarized as follows:

Total Responses: The dataset contains 122 individual responses from participants.

Data Types:

Integer Columns: There are 7 columns that hold numerical data, representing various ratings and counts.

Object Columns: There are 14 columns that contain categorical data, including information such as gender, current status, visit frequency and membership details.

This combination of numerical and categorical data allows for a comprehensive analysis of customer satisfaction, providing insights into both quantitative ratings and qualitative preferences.

## Understanding Data and Data Cleaning

The data is up to the year 2019 and later.

Data cleaning is critical to improving data quality in multiple dimensions. Below are some common steps we have to do for data cleaning:

1. Check for data duplication:
   
   There are no duplicated rows for this dataset.

2. Drop unrelated column:
   
   Since duplication has been checked, the (Timestamp) index key is no longer useful and can be removed.
  
3. Check Missing Value:

   In this dataset, there are two columns containing empty values which are OrderType (0.82) and PromotionTools (0.82). The percentage of missing data is low. Hence, we simply remove missing data as it will not cause significant information loss.
   

4. Rename all feature names:
 
   The column names and column values are quite long, thus, it is a good idea to rename them or convert them into a more convenient and feasible format.
   
   Original:

   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/ca78a85d-dabc-4639-b750-bd443fd1c5b4)

   After modify:
   
   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/6736cabd-dc26-4912-8092-6f65f072007b)

6. Check Feature's value:

   There are some columns that contain duplicate values. For example:
   
   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/24972039-afed-43e0-a1e6-7984f2472511)
   
   Replaced 'never', 'Never buy', 'I dont like coffee', 'Never', 'Never ' with 'Never buy'.
   
   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/155c4df2-441e-4b0e-a43a-6ea9a24b122c)

   Process the value of columns that are not suitable to be neatly arranged:

   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/77520768-ec84-4aa6-a095-787af046831e)

   Separate the values and make new columns:

   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/3cf4d874-f1b1-4de1-b077-2aac235a9c9d)

   After the data cleaning process, the dataset contains 121 entries and 36 columns.

7. Exploratory Data Analysis (EDA):

   Univariate Analysis (Categorical)
   
   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/abf90ee8-0d23-4f73-824f-fbf128e67c0a)

   Bivariate (Numerical-Numerical) [Correlation Heatmap]

   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/ac1b2d18-8f93-410e-bba0-023c4a2d7395)

   Class Distribution of Target Variable

   ![image](https://github.com/Heng12312/Machine-Learning-Project/assets/174890151/3c1f6519-f1b5-4489-8c1d-cd458e60b106)




[//]: <> (HENG PENG YONG)

## Model Construction
The table below summarizes the experiments conducted in this project using default parameters and a fixed random state of 42. The experiments compare the performance of Logistic Regression (LR) and Random Forest (RF) models under various conditions, including baseline setups, Principal Component Analysis (PCA) preprocessing, and PCA combined with Synthetic Minority Over-sampling Technique (SMOTE).

### Experimental Setup

Each experiment in the table below explores a specific setup:
- **Baseline**: Represents the model's performance without any additional preprocessing or sampling techniques.
- **PCA**: Utilizes Principal Component Analysis (PCA) as a preprocessing step to reduce the dimensionality of the feature space.
- **PCA + SMOTE**: Extends PCA with Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance by generating synthetic samples of the minority class.


| Model Code | Preprocessing | Algorithm | 
| --------| ---------|-------- |
| M<sub>1</sub>  | Feature encoding + Train Test Split (Baseline)              | LR   | 
| M<sub>2</sub>  | Feature encoding + PCA + Train Test Split         | LR   |
| M<sub>3</sub>  | Feature encoding + PCA + Train Test Split + SMOTE | LR   | 
| M<sub>4</sub>  | Feature encoding + Train Test Split (Baseline)             | RF |
| M<sub>5</sub>  | Feature encoding + PCA + Train Test Split         | RF |
| M<sub>6</sub>  | Feature encoding + PCA + Train Test Split + SMOTE | RF |

Each experiment explores different configurations to evaluate the performance of the models in handling the dataset, with LR and Gaussian NB chosen for their contrasting approaches and assumptions about the underlying data distribution.

## Model Evaluation

For model evaluation, we employ various metrics, such as accuracy, precision, recall, and F<sub>1</sub>-score score to ensure a more thorough assessment. 

| Metric     | Description                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------- |
| Accuracy   | Accuracy is a fundamental metric that measures the overall correctness of predictions made by a model. It calculates the ratio of correctly predicted instances (both positive and negative) to the total number of instances evaluated. While accuracy provides a quick snapshot of model performance, it may not be sufficient when dealing with imbalanced datasets where one class dominates the other.  |
| Precision  | Precision focuses on the correctness of positive predictions made by the model. It measures the proportion of true positive predictions (correctly predicted positive instances) among all instances predicted as positive. Precision is particularly useful in applications where the cost of false positives is high, such as in medical diagnostics or fraud detection.|
| Recall     |Recall, also known as sensitivity or true positive rate, assesses the model's ability to correctly identify positive instances from all actual positive instances. It measures the proportion of true positive predictions among all instances that are actually positive. Recall is crucial when the cost of false negatives (missed positive instances) is high, such as in disease detection or anomaly detection.|
| F<sub>1</sub>-score   |F<sub>1</sub>-score score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, making it useful when you need to take into account both false positives and false negatives. The F<sub>1</sub>-score score reaches its best value at 1 (perfect precision and recall) and worst at 0. |

## Findings
The testing performances of models constructed are summarized as the following:

| Model Code | Accuracy | Precision | Recall | F<sub>1</sub>-score |
| --------| ---------|-------- | ------| ------|
| M<sub>1</sub>  | 0.760 | 0.735 | 0.760 | 0.742 |
| M<sub>2</sub>  | 0.760 | 0.735 | 0.760 | 0.742 |
| M<sub>3</sub>  | 0.720 | 0.772 | 0.720 | 0.736 |
| M<sub>4</sub>  | 0.720 | 0.667 | 0.620 | 0.684 |
| M<sub>5</sub>  | 0.720 | 0.570 | 0.720 | 0.636 |
| M<sub>6</sub>  | 0.720 | 0.667 | 0.720 | 0.684 |

From the results, we can see that both M<sub>1</sub> and M<sub>2</sub> were yielding the best performances. In this case, we should claim that the optimal algorithm is logistic regression. The importances of applying PCA and SMOTE are not highlighted because models augmented with these techniques pre-classification do not outperform the baseline models. PCA is not yielding supreme results, most probably due to the nature of our dataset where it lacks of highly correlated features, which is necessary for a PCA to work well. SMOTE do not significantly improves the performance of classifiers because the class imbalance problem is not too extreme.

## Conclusion
Our work has revealed a specialized logistic regression to be the best-performing classifier in distinguishing between loyal and non-loyal Starbucks customer. By having such a capability, Starbucks staffs can understand more about their customer base and apply suitable business strategies to boost the profit. For a loyal customer, it is recommended to provide loyalty rewards for maintaining his or her loyalty to the brand. On the other hand, marketing campaigns should be held for non-loyal customers to boost their interest towards the brand.

## Reference
https://www.kaggle.com/datasets/mahirahmzh/starbucks-customer-retention-malaysia-survey

https://www.kaggle.com/code/aditiani/starbucks-customers-accuracy#Data-Cleaning

https://medium.com/towards-data-science/machine-learning-step-by-step-6fbde95c455a

[//]: <> (HENG PENG YONG)
  


