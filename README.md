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
  - [Visualzations](#visualizations)
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

Our immediate task was to identify the column we were trying to predict and transform it into a numerical column. We then One-Hot-Encoded the "phone" and "city" values by using the pd.get_dummies method.

![alt text](https://i.gyazo.com/36577607d61dadc29141180f4efd1581.png)

![alt text](https://i.gyazo.com/26e5056af25e24766d00a9a68eb65ca6.png)

Visualizing the NaN values in the columns, avg_rating_of_driver contianed the most NaN values with 16%. As a group, we decided to use sklearn's SimpleInputer to fill the NaN values with the mean of that column.

![alt text](https://i.gyazo.com/b5e55239362ee42f2090c68c7d9c61e0.png)


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

- Fill!

### Tuning

- Fill!

---
## Performance

- Fill!

## Future Considerations

- Fill

## License
[MIT ©](https://choosealicense.com/licenses/mit/)

## Credits

- Fill if other packages used

## Thanks

Thanks to instructors at Galvanize Seattle (the puns were for you Andrew)

Thanks to the generous work of the [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) team


