import matplotlib.pyplot as plt


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

