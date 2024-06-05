# %% Libraries and functions
import sys
from pathlib import Path
path_root=Path.cwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flim_module as flim
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold

def Get_XY_FromDataFrame(df, target_column):
    '''
    splits the dataset into X (features) and Y (target)
    '''
    target=df[target_column]
    df=df.drop(labels=target_column, axis=1)
    return df, target
    
#%% config variables
reduce_dataset=True
palette=['#c00000', '#00b050']

# %% Main

#### load data
filename=path_root/'data'/'HI_features.dat'
df=pd.read_csv(filename, index_col=None)

# # clean dataset by hand
if reduce_dataset:
    # load rows to exclude from csv/xlsx file
    filename_todrop=path_root/'data'/'ErrorAnalysis_rowstodrop.csv'
    df_todrop=pd.read_csv(filename_todrop)
    df=pd.concat([df, df_todrop]).drop_duplicates(subset=list(df_todrop.columns), keep=False)

le=LabelEncoder()
df['glucose']=le.fit_transform(df['glucose'])
df['cell_type']=le.fit_transform(df['cell_type'])

x,y=Get_XY_FromDataFrame(df, 'cell_type')
features_todrop=['date', 'islet', 'cell_number', 'sex']
x=x.drop(labels=features_todrop, axis=1)

#### preprocessing
x.fillna(0, inplace=True)
# y=le.fit_transform(y)

#### compute roc_auc
roc_auc=[]
scaler=MinMaxScaler()

kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

for i in list(x.columns):
    auc_scores = []
    
    for train, test in kf.split(x, y):
        # indexes=np.concatenate((train_index, test_index))
        X_split = x[x.index.isin(test)]
        X_split = X_split[i]
        Y_split = y[y.index.isin(test)]
        
        # Predict probabilities for the positive class
        
        # Compute ROC AUC score
        # auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(roc_auc_score(Y_split, X_split))
        # roc_auc.append(roc_auc_score(y, x[i]))
        
    roc_auc.append(np.mean(auc_scores))
    
    
#%% plot data
score=pd.DataFrame(
    {
     'feature': list(x.columns),
     'roc_auc': roc_auc
     }
    )

# roc_auc histplot
plt.figure(dpi=300)
sns.histplot(x='roc_auc', data=score, binwidth=0.05, stat='percent')
plt.ylabel('Percent (%)')
plt.show()

#%% best feature analysis
#### plot distribution of best feature
best_feature=list(score[score['roc_auc']==score['roc_auc'].max()].feature)[0]
sns.histplot(x=best_feature, data=df, hue='cell_type')

precision=[]
recall=[]
accuracy=[]
threshold=np.arange(x[best_feature].min(), x[best_feature].max())

for i in threshold:
    y_pred=x[best_feature]>i
    precision.append(precision_score(y, y_pred, average=None))
    recall.append(recall_score(y, y_pred, average=None))
    accuracy.append(accuracy_score(y, y_pred))

precision=pd.DataFrame(precision, columns=['alpha', 'beta'], index=np.round(threshold, 0))
recall=pd.DataFrame(recall, columns=['alpha', 'beta'], index=np.round(threshold, 0))
accuracy=pd.DataFrame(accuracy)

plt.figure(dpi=300, figsize=(12,3))
sns.lineplot(data=precision, dashes=False, palette=palette)
plt.xlabel(best_feature+' (a.u.)')
plt.ylabel('Precision (%)')
plt.show()
