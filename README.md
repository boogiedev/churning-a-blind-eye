![Churning a Blind Eye Header](https://raw.githubusercontent.com/boogiedev/churning-a-blind-eye/master/media/churnheader.png)

<p align="center">
  <img src="https://img.shields.io/badge/Maintained%3F-IN PROG-blue?style=flat-square"></img>
  <img src="https://img.shields.io/github/commit-activity/m/boogiedev/churning-a-blind-eye?style=flat-square">
  <img src="https://img.shields.io/github/license/boogiedev/churning-a-blind-eye?style=flat-square">
</p>

> *Why can't ride-sharing companies retain their customers?
  ... cause they're always driving their customers away!* 

<p align="center">
  <img src="https://img.shields.io/badge/JOKE-10/10-blue?style=flat-square"></img>
  <img src="https://img.shields.io/badge/LAUGHTER-KNEE%20SLAPPER-brightgreen?style=flat-square"></img>
  <img src="https://img.shields.io/badge/100%25-GLUTEN%20FREE-red?style=flat-square"></img>
</p>

## Team

[Feli Gentle](https://github.com/oro13)  | [Tu Pham](https://github.com/phamc4) | [Wesley Nguyen](https://github.com/boogiedev)
---|---|---|


Check [TEAM DEBRIEF](team_debrief.md) to get started
 
 
## Table of Contents

- [Basic Overview](#basic-overview)
- [Exploring Data](#exploring-data)
  - [Initial Intake](#initial-intake)
  - [Feature Engineering](#feature-engineering)
  - [Visualizations](#visualizations)
- [Predictive Modeling](#predictive-modeling)
  - [Baseline](#baseline)
  - [Evaluation](#evaluation)
  - [Tuning](#tuning)
- [Performance](#performance)
- [Future Considerations](#future-considerations)
- [License](#license)
- [Credits](#credits)
- [Thanks](#thanks)

## Basic Overview

A churn prediction case study focused on cleaning, analyzing, and modeling ride-sharing data aimed at seeking the best predictors for retention

## Exploring Data

Intially going into this case study, we decided to tackle the task of cleaning the data and get a better understanding of the data together as a group.

![alt text](https://i.gyazo.com/d374ece0e6454f46cf15fe91d499b586.png)



### Initial Intake

<<<<<<< HEAD
Our immediate task was to identify the column we were trying to predict and transform it into a numerical column. To use our columns in our algorithims we used the get dummies function to transform our categorical data into numerical.


![alt text](https://i.gyazo.com/1e5b834f86ed5c69799aa60f9e9adb1c.png)

![alt text](https://i.gyazo.com/f6eb2ac21718057b4c6c3f126977d19a.png)

![alt text](https://i.gyazo.com/36577607d61dadc29141180f4efd1581.png)




Examining the columns further, we looked to see if any features we correlated with any other features. Upon investigation, we saw that surge_pct was highly correlated with avg_surge so we decided to drop avg_surge to attempt to decorrelate the data.



=======
Our immediate task was to identify the column we were trying to predict and transform it into a numerical column. We then One-Hot-Encoded the "phone" and "city" values by using the pd.get_dummies method.

![alt text](https://i.gyazo.com/36577607d61dadc29141180f4efd1581.png)

>>>>>>> 98bb37b244406e09f180d9235213117d40250f9d
![alt text](https://i.gyazo.com/26e5056af25e24766d00a9a68eb65ca6.png)


Visualizing the NaN values in the columns, avg_rating_of_driver contianed the most NaN values with 16%. As a group, we decided to use sklearn's SimpleInputer to fill the NaN values with the mean of that column.

![alt text](https://i.gyazo.com/6f35dbfc2614c846ba1fcdaf1931e3b9.png)


### Feature Engineering



### Visualzations

- Fill!

---
## Predictive Modeling


### Baseline

Settling on a baseline model of a Random Forest Classifier, we uncovered metrics such as feature importance, out-of-bag error, and the model's initial accuracy predicitons.

```python
# Create X, y arrays from dataframe
X = churn
y = churn.pop("target")

# Train Test Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create Random Forest Model
model_rf = RandomForestClassifier(
                            oob_score=True,
                            max_features=3)
# Fit Data
model_rf.fit(X_train, y_train)

# Use Helper function to get score
get_score(model_rf, X_train, y_train)
```
OOB | MSE | R2 | ACC
---|---|---|---|
0.7469 | 0.2538 | -0.088 | 0.744

<p align="center">
  Feature Importances
</p>

<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/feature_importance.png"> </img>

### Evaluation

Honestly, the OOB score and the model's initial accuracy was not bad, although it could be improved with possibly a different model! We went ahead and trained a host of other models in order to get a comparison of what the differences were. A simple, succinct and handy function was used to compare all of these models against each other. Click the tab below if you would like to see.


<details>
  <summary>
    <b> Model Comparison Code </b>  
  </summary>
  
```python
def get_model_scores(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)    
    #Fit the logistic Regression Model
    logmodel = LogisticRegression(random_state = 50)
    logmodel.fit(X_train,y_train)

    #Predict the value for new, unseen data
    pred = logmodel.predict(X_test)

    # Find Accuracy using accuracy_score method
    logmodel_accuracy = round(accuracy_score(y_test, pred) * 100, 2)

    # Scaler
    scaler = MinMaxScaler()

    #Fit the K-Nearest Neighbor Model
    knnmodel = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2) #p=2 represents Euclidean distance, p=1 represents Manhattan Distance
    knnmodel.fit(scaler.fit_transform(X_train), y_train) 

    #Predict the value for new, unseen data
    knn_pred = knnmodel.predict(X_test)

    # Find Accuracy using accuracy_score method
    knn_accuracy = round(accuracy_score(y_test, knn_pred) * 100, 2)

    #Fit the Decision Tree Classification Model
    dtmodel = DecisionTreeClassifier(criterion = "gini", random_state = 50)
    dtmodel.fit(X_train, y_train) 

    #Predict the value for new, unseen data
    dt_pred = dtmodel.predict(X_test)

    # Find Accuracy using accuracy_score method
    dt_accuracy = round(accuracy_score(y_test, dt_pred) * 100, 2)

    #Fit the Random Forest Classification Model
    rfmodel = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    rfmodel.fit(X_train, y_train) 

    #Predict the value for new, unseen data
    rf_pred = rfmodel.predict(X_test)

    # Find Accuracy using accuracy_score method
    rf_accuracy = round(accuracy_score(y_test, rf_pred) * 100, 2)

    #Fit the Gradient Boosted Classification Model
    gbmodel = GradientBoostingClassifier(random_state=50)
    gbmodel.fit(X_train,y_train)

    #Predict the value for new, unseen data
    pred = gbmodel.predict(X_test)

    # Find Accuracy using accuracy_score method
    gbmodel_accuracy = round(accuracy_score(y_test, pred) * 100, 2)

    #Fit the Gradient Boosted Classification Model
    gbmodel_grid = GradientBoostingClassifier(learning_rate=0.1,
                                         max_depth=6,
                                         max_features=0.3,
                                         min_samples_leaf=10,
                                         n_estimators=100,
                                         random_state=50)
    gbmodel_grid.fit(X_train,y_train)

    #Predict the value for new, unseen data
    pred = gbmodel_grid.predict(X_test)

    # Find Accuracy using accuracy_score method
    gbmodel_grid_accuracy = round(accuracy_score(y_test, pred) * 100, 2)
    
    #Fit the Gradient Boosted Classification Model
    gbmodel_grid_cv = GradientBoostingClassifier(learning_rate=0.2,
                                         max_depth=4,
                                         max_features=9,
                                         min_samples_leaf=2,
                                         n_estimators=150,
                                         random_state=50)
    gbmodel_grid_cv.fit(X_train,y_train)

    #Predict the value for new, unseen data
    pred = gbmodel_grid_cv.predict(X_test)

    # Find Accuracy using accuracy_score method
    gbmodel_grid_cv_accuracy = round(accuracy_score(y_test, pred) * 100, 2)
    
    return [logmodel_accuracy, knn_accuracy, dt_accuracy, rf_accuracy, gbmodel_accuracy, gbmodel_grid_accuracy, gbmodel_grid_cv_accuracy]

```
  
  
  
</details>



<p align="center">
  Using Other Models
</p>
<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/pre_model_scores.png"> </img>



### Tuning

- Fill!

---
## Performance

#### GBC GRID CV MODEL
Confusion Matrix
| -        |       Predicted Negative      |  Predicted Positive |
| ------------- |:-------------:| -----:|
| Actual Negative | 2314 (TN)  | 1169 (FP)
| Actual Positive | 867 (FN) | 5030 (TP)

PRECISION / RECALL
| -        |       Precision      |  Recall |
| ------------- |:-------------:| -----:|
| Best | 0.816  | 0.852


## Future Considerations



## License
[MIT ©](https://choosealicense.com/licenses/mit/)

## Credits

Pandas Profiling

## Thanks

Thanks to instructors at Galvanize Seattle (the puns were for you Andrew)

Thanks to the generous work of the [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) team


