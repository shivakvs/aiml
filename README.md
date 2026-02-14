a. Problem statement

    In Healthcare, finding out a person is diagnosed with breast tissues or not.
    Finding out the best model that produces accuracy of given dataset which contains Breast Cancer Wisconsin (Diagnostic)
    
b. Dataset description : 

        Dataset      : Breast Cancer Wisconsin (Diagnostic) Data Set
        Data set URL : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
        Features : 33
        Instance size : 569
        Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
        They describe characteristics of the cell nuclei present in the image.       
        The mean, standard error and "worst" or largest (mean of the three largest values) of these
        features were computed for each image, resulting in 30 features. For instance, field 
        3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

c. Models used: 

    Logistics Regression, Decision Tree, KNN, Naive Bayes, Random Forest ( Ensemble ) and XGBoost ( Ensemble )


## Evaluation Metrics of each model

| ML Model Name         |  Accuracy |      AUC |    Precision  |   Recall |        F1 |       MCC | 
| --------------------  | ----------  | -------- |  ----------- |  -------- |  -------- |  -------- |
| Logistics Regression  |  0.973684 | 0.969246 |   0.97561| 0.952381 | 0.963855 |  0.94334 |
| Decision Tree         |  0.938596 | 0.922454 |   0.948718 | 0.880952 |  0.91358 |   0.867493 |
| KNN                   |  0.938596 | 0.921627 |   0.972973 |  0.857143 | 0.911392 | 0.868766 |
| Naive Bayes           |  0.921053 | 0.907738 |    0.923077 |  0.857143 | 0.888889 | 0.829162 |
| Random Forest         |  0.973684 | 0.997354 |    1       |  0.928571 | 0.962963 | 0.944155 |
| XGBoost               |  0.964912 | 0.952381 |    1       |  0.904762 |  0.95    |  0.92582 |

## Observation of each model performance

| ML Model Name   | Observation about model performance |
| --------------------  | ----------  | 
| Logistics Regression |  Performed well overall except AUC. Random Forest better |
| Decision Tree   |  Low bias and high variance. More FP and FN comparing Random Forest|
| KNN             |  Performed better than Naive bayes and equivalent to Decision tree. Missing positive |
| Naive Bayes   | Lowest accuracy and other metrics parameter among all the models.  |
| Random Forest | Performed well overall.Best among other model. Accuracy is same as Logistic regression but other factors are well. |
| XGBoost       | Dataset is < 600 so not performed well comparing RF. FN is 4, slightly higher than RF   |
