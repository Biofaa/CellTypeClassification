# %% Libraries and functions
import sys
from pathlib import Path
path_root=Path.cwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flim_module as flim
import seaborn as sns
    
#%% config variables
compact_score=False #if false, you'll obtain a separate score for alpha and beta cells
reduce_dataset=False
xgb_feature_selection=True
save_model_bool=True
model_name='xgb_optuna_FeatureSelection_div0.pkl'
feature_threshold=0.001

# %% Load data

# load dataset
filename=path_root/'data'/'HI_features.dat'
df=pd.read_csv(filename, index_col=None)

# # clean dataset by hand
if reduce_dataset:
    # load rows to exclude from csv/xlsx file
    filename_todrop=path_root/'data'/'ErrorAnalysis_rowstodrop.csv'
    df_todrop=pd.read_csv(filename_todrop)
    df=pd.concat([df, df_todrop]).drop_duplicates(subset=list(df_todrop.columns), keep=False)

#%% count cells and cell ratio
df=pd.DataFrame(data=[df['date'].astype(str)+'_'+df['islet'], df['cell_type']]).T
df.columns=['date', 'cell type']
df['beta']=df['cell type'].str.contains('beta')
df['alpha']=df['cell type'].str.contains('alpha')
# df.index=df['date']
df.drop(labels=['cell type'], axis=1, inplace=True)


a=df.value_counts()
a.to_excel(path_root/'data'/'celltypecounts_raw.xlsx')

# # %% pre-processing

# #### get x and y (extract only features)
# x,y=Get_XY_FromDataFrame(df, 'cell_type')
# x=x.iloc[:,3:]
# x=x.drop(labels='sex', axis=1)

# #### dataset splitting
# X_train, X_test, Y_train, Y_test = train_test_split(x, y)

# #### transform data: scaling, categorical features encoding etc
# X_train, Y_train = DataTransform(X_train, Y_train)
# X_test, Y_test = DataTransform(X_test, Y_test)

# #### drop almost all features
# # cols=['g_barycenter_std', 'g_CV', 'g_IQR', 'g_std']
# # cols=['g_barycenter_std', 'g_CV', 'g_IQR', 'g_std', 'cell_circularity', 'g_99', 'g_CI_95_max', 'g_max', 'g_mean', 'g_mode', 'g_whisker_high', 'intensity_cytoplasm_rel_CV', 'lipofuscin_area_rel', 's_mode']
# # X_train=X_train[cols]
# # X_test=X_test[cols]

# #### scaling
# # Zscore
# # x=scaling.Zscore(x)
# # x=x.dropna()
# # y=y.loc[x.index]

# # MinMaxScaler
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# X_train=pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_test=pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# #### cross validation
# from sklearn.model_selection import RepeatedStratifiedKFold
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# #### Feature Selection
# # Multicollinearity
# # to_drop=MultiCollinearity.fit(X_train, 0.90)
# # X_train=MultiCollinearity.transform(X_train, to_drop)
# # X_test=MultiCollinearity.transform(X_test, to_drop)

# # Elastic Net
# # coefs=MyElasticNet.fit(X_train, Y_train, cv)
# # X_train=MyElasticNet.transform(X_train, coefs)
# # X_test=MyElasticNet.transform(X_test, coefs)

# # LASSO
# # coefs=MyLasso.fit(X_train, Y_train, cv)
# # X_train=MyLasso.transform(X_train, coefs)
# # X_test=MyLasso.transform(X_test, coefs)

# #### Balance classes
# # SMOTE
# oversample = imblearn.over_sampling.SMOTE()
# X_train, Y_train = oversample.fit_resample(X_train, Y_train)

# # %% XGBoost optimized
# from xgboost import XGBClassifier
# from sklearn.model_selection import cross_val_score
# import optuna

# if xgb_feature_selection:
#     # load original xgb model
#     model_path=path_root/'models'/'xgb_optuna.pkl'
#     model_xgb_old=load_model(model_path)
    
#     # select most important features
#     xgb_features=pd.DataFrame(model_xgb_old.feature_importances_, index=list(x.columns))
#     if feature_threshold==0:
#         xgb_best_features=list(xgb_features[xgb_features.values!=0].index)
#     else:        
#         xgb_best_features=list(xgb_features[xgb_features.values>=feature_threshold].index)
    
#     # cut dataset features
#     x=x[xgb_best_features]
#     X_train=X_train[xgb_best_features]
#     X_test=X_test[xgb_best_features]

# # %% Salzberg test

# # generate Y_rnd
# Y_rnd = np.random.permutation(Y_train)
# model.fit(X_train, Y_rnd)
# print('Salzberg test results:\n')
# print('training')
# score=performance_scores(model, X_train, Y_train, compact=compact_score)
# print(score)

# print('\ntest')
# score=performance_scores(model, X_test, Y_test, compact=compact_score)
# print(score)
