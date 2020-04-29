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

A churn prediction case study focused on cleaning, analyzing, and modeling ride-sharing data aimed at seeking the best predictors for retention. A sample dataset of a cohort of users who signed up for an account in Januaray 2014 was used. We considered a user retained if they were "active" (i.e. took a trip) in the preceding 30 days from the day the data was pulled. For this dataset, a user is "active" if they have taken a trip since June 1, 2014 as the data was pulled on July 1, 2014.

## Exploring Data

Intially going into this case study, we decided to tackle the task of cleaning the data and get a better understanding of the data together as a group. 

Here is a detailed description of the data:

- `city`: city this user signed up in phone: primary device for this user
- `signup_date`: date of account registration; in the form `YYYYMMDD`
- `last_trip_date`: the last time this user completed a trip; in the form `YYYYMMDD`
- `avg_dist`: the average distance (in miles) per trip taken in the first 30 days after signup
- `avg_rating_by_driver`: the rider’s average rating over all of their trips 
- `avg_rating_of_driver`: the rider’s average rating of their drivers over all of their trips 
- `surge_pct`: the percent of trips taken with surge multiplier > 1 
- `avg_surge`: The average surge multiplier over all of this user’s trips 
- `trips_in_first_30_days`: the number of trips this user took in the first 30 days after signing up 
- `luxury_car_user`: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise 
- `weekday_pct`: the percent of the user’s trips occurring during a weekday


### Initial Intake

Our immediate task was to identify the column we were trying to predict and transform it into a numerical column. Our target column is last_trip_date which was converted to a datetime type and then transformed into a numerical column where churned useres were assigned a 1 and active users a 0. 

Visualizing the NaN values in the columns, avg_rating_of_driver contianed the most NaN values with 16%. Since we decided to intially predict with a Random Forest Classifier, we dropped these NaN rows instead of filling them with a mean value. 

<img src="https://i.gyazo.com/b5e55239362ee42f2090c68c7d9c61e0.png" width="600"> </img>

In an attempt to decorrelate features we looked at the correlation matrix and found surge_pct and average_surge to be highly correlated so we decided to drop average_surge. Then we One Hot Encoded the city column as a final step to prepare our data to be used in a predictive model.

![alt text](https://i.gyazo.com/26e5056af25e24766d00a9a68eb65ca6.png)

Here is the data after One Hot Encoding and Cleaning. The converted numerical last_trip_date column was renamed to target:

![alt text](https://i.gyazo.com/5bbe790a7c8c3522faffc918ecf2817c.png)

### Feature Engineering


### Visualzations

<img src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/churn_pie_chart.png"> </img>

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

```python
# Create X, y arrays from dataframe
X = churn
y = churn.pop("target")

# Train Test Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create Random Forest Model
model_gb = GradientBoostingClassifier()

# Fit Data
model_gb.fit(X_train, y_train)

# Use Helper function to get score
get_score(model_rf, X_train, y_train)
```

<p align="center">
  Using Other Models
</p>
<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/pre_model_scores.png"> </img>

Creating a baseline Gradient Boosting Classifier model, we were also able to plot its ROC curve. Out of the box, it was able to obtain an AUC of 0.86, which seems to be pretty good. Let's see if we can do better!


<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/pre_roc.png"> </img>

### Tuning

Outside of the features, we decided to use Grid Searching in order to tune the model in order to get better predictions.

<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/grid_search_cv_process.png"> </img>

After a whopping 67 minutes of the Cross Validated Grid Search, we finally obtained the what was percieved to be the best hyperparameters for this model. 

```
 'learning_rate': 0.2,
 'loss': 'exponential',
 'max_depth': 4,
 'max_features': 9,
 'min_samples_leaf': 2,
 'n_estimators': 150
```

We then checked the feature importances of the new model with the painstakinly tuned hyperparameters and there were differences! Keep in mind as well, we also chopped off outliers from "surge_pct".

<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/gdb_feature_imp.png"> </img>

A new ROC curve was plotted and there was a slight improvement to the model as seen with a higher AUC.

<img align="center" src="https://github.com/boogiedev/churning-a-blind-eye/blob/master/media/post_roc.png"> </img>



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

 MSE | R2 | ACC
---|---|---|
 0.2184 | 0.071 | 0.792

## Future Considerations

Ideally, the predictive model would be trained on a online system. As the data gets updated and repulled this would improve our predictive model overtime and be updating active users to churned users and visa versa. 

## License
[MIT ©](https://choosealicense.com/licenses/mit/)

## Credits

Pandas Profiling

## Thanks

Thanks to instructors at Galvanize Seattle (the puns were for you Andrew)

Thanks to the generous work of the [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) team


