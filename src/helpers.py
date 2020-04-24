from sklearn.metrics import mean_squared_error, r2_score, make_scorer, confusion_matrix, accuracy_score, plot_roc_curve
import os
from itertools import islice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score



# Helper Functions
def get_score(model, X, y) -> tuple:
    mse = np.mean(cross_val_score(model, X, y, scoring=make_scorer(mean_squared_error)))
    r2 = np.mean(cross_val_score(model, X, y, scoring=make_scorer(r2_score)))
    acc = np.mean(cross_val_score(model, X, y, scoring="accuracy"))
    print(f"""{model.__class__.__name__}     Train CV | MSE: {mse} | R2: {r2} | Acc: {acc}""")
    return mse, r2, acc
def display_score_metrics(model, y_pred, y_test) -> tuple:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"""{model.__class__.__name__}     Train CV | MSE: {mse} | R2: {r2} | Acc: {acc}""")
    return mse, r2, acc

# Plotting

'''
example code:

from src.plotting import *

fig, ax = plt.subplots(figsize=(10,10))

col1 = X['weekday_pct']
col2 = X['avg_dist']

compare_feat_plot(col1, col2, y, ax)

'''
def compare_feat_plot(col1, col2, target_vals, ax, alpha=.5):
    
    f = plt.scatter(col1, col2, c=target_vals, alpha=alpha)
    plt.xlabel(f'{col1.name}')
    plt.ylabel(f'{col2.name}')
    ax.legend(*f.legend_elements())
    
    return ax


