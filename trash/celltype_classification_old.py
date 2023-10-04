# %% Main
import sys
sys.path.append('G:/My Drive/PhD/CAPTUR3D_personal/03 Software e utilities/v0.2.0/Libraries')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flim_module as flim
import seaborn as sns
import re #for regular expressions
import sklearn
import sklearn.model_selection
import os

def load_R64_bkp(use_donor_DB=False):
    
    if use_donor_DB:
        print('select donor database:')
        df_donor=load_dat()
        filename=list(df_donor['filename'].values)
        df_donor=df_donor.drop(labels='filename', axis=1)
    else:
        print('select .R64 file:')
        filename=flim.decode.getfile('R64')
        # manual input user ID information (could be automatized)
        date=input('Date (e.g. 20210630):')
        islet=input('islet (PXHY): ')
        cell_type=input('cell type (alpha/beta): ')
        sex=input('donor sex: ')
        age=input('age: ')
        BMI=input('BMI: ')
        SI=input('Insulin Stimulatory Index: ')
        
        cell_number=[]
        glucose=[]
        
        for i in range(0,np.size(filename)):
            name=filename[i].split('.')[0]
            name=name.split('_')[-1]
            cell_number.append(name)
            glucose_tmp=re.findall('\d+', filename[i].split('_')[0])[-1]+'mM'
            glucose.append(glucose_tmp)
        
        # index selection
        
        df_donor={
            'date': date,
            'islet': islet,
            'cell_type': cell_type,
            'cell_number': cell_number,
            'glucose': glucose,
            'sex': sex,
            'age': age,
            'BMI': BMI,
            'insulin_SI': SI
            }
    
    ################ FEATURES CREATION ###################
    
    # intensity correction (bkg): not needed due to ROI masking
    
    # prepare data for features extractio
    dim3=np.size(filename)    
    
    lipofuscin_area=[]
    cell_area=[]
    g_barycenter_std=[]
    cell_circularity=[]
    cell_perimeter=[]
    
    df_g=pd.DataFrame([])
    df_s=pd.DataFrame([])
    df_intensity_all_rel=pd.DataFrame([])
    df_intensity_cytoplasm_rel=pd.DataFrame([])
    df_intensity_lipofuscin_rel=pd.DataFrame([])
    for i in range(0, dim3):
        g,s,intensity, namefile, g_har2, s_har2 =flim.decode.import_R64(har2=True, filename=filename[i])
        dim3=np.size(intensity, axis=0)
        threshold_high=flim.LipofuscinThresholdCalculus(intensity)
      
        ## calculate descriptive statistics on g and s matrices
        df_g_tmp=describe(g)
        df_g=pd.concat([df_g, df_g_tmp])
        
        df_s_tmp=describe(s)
        df_s=pd.concat([df_s, df_s_tmp])
        
        ## intensity
        intensity_rel=intensity/np.max(intensity) # normalize by max value
        df_intensity_all_rel_tmp=describe(intensity_rel)
        df_intensity_all_rel=pd.concat([df_intensity_all_rel, df_intensity_all_rel_tmp])
        
        intensity_lipofuscin_rel=np.where(intensity[i,:,:]>threshold_high[i], intensity[i,:,:], 0)/np.max(intensity[i,:,:])
        df_intensity_lipofuscin_rel_tmp=describe(intensity_lipofuscin_rel)
        df_intensity_lipofuscin_rel=pd.concat([df_intensity_lipofuscin_rel, df_intensity_lipofuscin_rel_tmp])
        
        intensity_cytoplasm_rel=np.where(intensity[i,:,:]<threshold_high[i], intensity[i,:,:], 0)/np.max(intensity[i,:,:])
        df_intensity_cytoplasm_rel_tmp=describe(intensity_cytoplasm_rel)
        df_intensity_cytoplasm_rel=pd.concat([df_intensity_cytoplasm_rel, df_intensity_cytoplasm_rel_tmp])
        
        # lipofuscin area (not in dataframe)
        lipofuscin_area.append(np.count_nonzero(intensity_lipofuscin_rel))
        
        # cell shape and area
        cell_area.append(np.count_nonzero(intensity[i,:,:]))
        cell_perimeter.append(flim.image.perimeter(intensity[i,:,:]))
        cell_circularity.append(flim.image.circularity(intensity[i,:,:]))  
        
    # lipofuscin realtive area calculus
    lipofuscin_area_rel=np.divide(lipofuscin_area, cell_area)    
    
    # compute barycenter and barycenter_std
    g_barycenter, s_barycenter=flim.barycenter(g, s)
    g_barycenter_std, s_barycenter_std=flim.barycenter_std(g, s)
    
    # oxphos (all e nolipo)
    threshold_low=30
    threshold_high=flim.LipofuscinThresholdCalculus(intensity)
    g_nolipo, s_nolipo=flim.IntensityThresholdCorrection(g, s, intensity, threshold_low, threshold_high)  
    g_nolipo_baryc, s_nolipo_baryc=flim.barycenter(g_nolipo, s_nolipo)
    oxphos_nolipo=flim.metabolism.gs_to_oxphos(g_nolipo_baryc, s_nolipo_baryc)
    oxphos=flim.metabolism.gs_to_oxphos(g_barycenter, s_barycenter)

    # adjust column names
    cols_g=list(df_g.columns)
    cols_s=list(df_s.columns)
    cols_int_all=list(df_intensity_all_rel.columns)
    cols_int_cyt=list(df_intensity_cytoplasm_rel.columns)
    cols_int_lipo=list(df_intensity_lipofuscin_rel.columns)
    for i in range(0, np.size(df_g.columns)):
        cols_g[i]='g_'+cols_g[i]
        cols_s[i]='s_'+cols_s[i]
        cols_int_all[i]='intensity_all_rel_'+cols_int_all[i]
        cols_int_cyt[i]='intensity_cytoplasm_rel_'+cols_int_cyt[i]
        cols_int_lipo[i]='intensity_lipofuscin_rel_'+cols_int_lipo[i]
        
    df_g.columns=cols_g
    df_s.columns=cols_s
    df_intensity_all_rel.columns=cols_int_all
    df_intensity_lipofuscin_rel.columns=cols_int_lipo
    df_intensity_cytoplasm_rel.columns=cols_int_cyt
        
    # ############### DATASET CREATION #################
    df={
        'g_barycenter': g_barycenter,
        's_barycenter': s_barycenter,
        'g_barycenter_std': g_barycenter_std,
        's_barycenter_std': s_barycenter_std,
        'cell_area': cell_area,
        'cell_perimeter': cell_perimeter,
        'cell_circularity': cell_circularity,
        'oxphos_nolipo': oxphos_nolipo,
        'oxphos': oxphos,
        'lipofuscin_area_rel': lipofuscin_area_rel
        }
    
    df=pd.DataFrame(df)
    df=df_donor.join(df)
    df=df.join(df_g)
    df=df.join(df_s)
    df=df.join(df_intensity_all_rel)
    df=df.join(df_intensity_cytoplasm_rel)
    df=df.join(df_intensity_lipofuscin_rel)
        
    # round at two decimal digits
    df=df.round(2)
    return df

def load_R64(use_donor_DB=False):
    
    if use_donor_DB:
        print('select donor database:')
        df_donor=load_dat()
        filename=list(df_donor['filename'].values)
        df_donor=df_donor.drop(labels='filename', axis=1)
    else:
        print('select .R64 file:')
        filename=flim.decode.getfile('R64')
        # manual input user ID information (could be automatized)
        date=input('Date (e.g. 20210630):')
        islet=input('islet (PXHY): ')
        cell_type=input('cell type (alpha/beta): ')
        sex=input('donor sex: ')
        age=input('age: ')
        BMI=input('BMI: ')
        SI=input('Insulin Stimulatory Index: ')
        
        cell_number=[]
        glucose=[]
        
        for i in range(0,np.size(filename)):
            name=filename[i].split('.')[0]
            name=name.split('_')[-1]
            cell_number.append(name)
            glucose_tmp=re.findall('\d+', filename[i].split('_')[0])[-1]+'mM'
            glucose.append(glucose_tmp)
        
        # index selection
        
        df_donor={
            'date': date,
            'islet': islet,
            'cell_type': cell_type,
            'cell_number': cell_number,
            'glucose': glucose,
            'sex': sex,
            'age': age,
            'BMI': BMI,
            'insulin_SI': SI
            }
    
    # convert date field in str
    df_donor['date'].astype(dtype=str)
    
    ################ FEATURES CREATION ###################
    
    # intensity correction (bkg): not needed due to ROI masking
    
    # prepare data for features extractio
    dim3=np.size(filename)    
    
    lipofuscin_area=[]
    cell_area=[]
    g_barycenter=[]
    g_barycenter_std=[]
    s_barycenter=[]
    s_barycenter_std=[]
    cell_circularity=[]
    cell_perimeter=[]
    oxphos_nolipo=[]
    
    df_g=pd.DataFrame([])
    df_s=pd.DataFrame([])
    df_intensity_all_rel=pd.DataFrame([])
    df_intensity_cytoplasm_rel=pd.DataFrame([])
    df_intensity_lipofuscin_rel=pd.DataFrame([])
    for i in range(0, dim3):
        print(str(i+1)+' of '+str(dim3))
        g,s,intensity, namefile, g_har2, s_har2 =flim.decode.import_R64(har2=True, filename=filename[i])
        threshold_high=flim.LipofuscinThresholdCalculus(intensity)
      
        ## calculate descriptive statistics on g and s matrices
        df_g_tmp=describe(g)
        df_g=pd.concat([df_g, df_g_tmp])
        
        df_s_tmp=describe(s)
        df_s=pd.concat([df_s, df_s_tmp])
        
        ## intensity
        intensity_rel=intensity/np.max(intensity) # normalize by max value
        df_intensity_all_rel_tmp=describe(intensity_rel)
        df_intensity_all_rel=pd.concat([df_intensity_all_rel, df_intensity_all_rel_tmp])
        
        intensity_lipofuscin_rel=np.where(intensity>threshold_high, intensity, 0)/np.max(intensity)
        df_intensity_lipofuscin_rel_tmp=describe(intensity_lipofuscin_rel)
        df_intensity_lipofuscin_rel=pd.concat([df_intensity_lipofuscin_rel, df_intensity_lipofuscin_rel_tmp])
        
        intensity_cytoplasm_rel=np.where(intensity<threshold_high, intensity, 0)/np.max(intensity)
        df_intensity_cytoplasm_rel_tmp=describe(intensity_cytoplasm_rel)
        df_intensity_cytoplasm_rel=pd.concat([df_intensity_cytoplasm_rel, df_intensity_cytoplasm_rel_tmp])
        
        # lipofuscin area (not in dataframe)
        lipofuscin_area.append(np.count_nonzero(intensity_lipofuscin_rel))
        
        # cell shape and area
        cell_area.append(np.count_nonzero(intensity))
        cell_perimeter.append(flim.image.perimeter(intensity[0,:,:]))
        cell_circularity.append(flim.image.circularity(intensity[0,:,:]))
        
        # compute barycenter and barycenter_std
        g_barycenter_tmp, s_barycenter_tmp=flim.barycenter(g, s)
        g_barycenter.append(g_barycenter_tmp)
        s_barycenter.append(s_barycenter_tmp)
        
        g_barycenter_std_tmp, s_barycenter_std_tmp=flim.barycenter_std(g, s)
        g_barycenter_std.append(g_barycenter_std_tmp)
        s_barycenter_std.append(s_barycenter_std_tmp)
        
        # oxphos (nolipo)
        threshold_low=30
        threshold_high=flim.LipofuscinThresholdCalculus(intensity)
        g_nolipo, s_nolipo=flim.IntensityThresholdCorrection(g, s, intensity, threshold_low, threshold_high)  
        g_nolipo_baryc, s_nolipo_baryc=flim.barycenter(g_nolipo, s_nolipo)
        oxphos_nolipo.append(flim.metabolism.gs_to_oxphos(g_nolipo_baryc, s_nolipo_baryc))
    
    # oxphos (without any corrections)
    oxphos=list(flim.metabolism.gs_to_oxphos(g_barycenter, s_barycenter))
    
    # lipofuscin realtive area calculus
    lipofuscin_area_rel=np.divide(lipofuscin_area, cell_area)    

    # adjust column names
    cols_g=list(df_g.columns)
    cols_s=list(df_s.columns)
    cols_int_all=list(df_intensity_all_rel.columns)
    cols_int_cyt=list(df_intensity_cytoplasm_rel.columns)
    cols_int_lipo=list(df_intensity_lipofuscin_rel.columns)
    
    for i in range(0, np.size(df_g.columns)):
        cols_g[i]='g_'+cols_g[i]
        cols_s[i]='s_'+cols_s[i]
        cols_int_all[i]='intensity_all_rel_'+cols_int_all[i]
        cols_int_cyt[i]='intensity_cytoplasm_rel_'+cols_int_cyt[i]
        cols_int_lipo[i]='intensity_lipofuscin_rel_'+cols_int_lipo[i]
        
    df_g.columns=cols_g
    df_s.columns=cols_s
    df_intensity_all_rel.columns=cols_int_all
    df_intensity_lipofuscin_rel.columns=cols_int_lipo
    df_intensity_cytoplasm_rel.columns=cols_int_cyt
    
    #reshape otherwise you'll obtain a list of lists
    g_barycenter=np.reshape(g_barycenter, np.size(g_barycenter, axis=0))
    g_barycenter_std=np.reshape(g_barycenter_std, np.size(g_barycenter_std, axis=0))
    s_barycenter=np.reshape(s_barycenter, np.size(s_barycenter, axis=0))
    s_barycenter_std=np.reshape(s_barycenter_std, np.size(s_barycenter_std, axis=0))
    oxphos=np.reshape(oxphos, np.size(oxphos, axis=0))
    oxphos_nolipo=np.reshape(oxphos_nolipo, np.size(oxphos, axis=0))
    
    # ############### DATASET CREATION #################
    df={
        'g_barycenter': g_barycenter,
        's_barycenter': s_barycenter,
        'g_barycenter_std': g_barycenter_std,
        's_barycenter_std': s_barycenter_std,
        'cell_area': cell_area,
        'cell_perimeter': cell_perimeter,
        'cell_circularity': cell_circularity,
        'oxphos_nolipo': oxphos_nolipo,
        'oxphos': oxphos,
        'lipofuscin_area_rel': lipofuscin_area_rel
        }
    
    # # adjust indices for join
    df_g.index=range(0, np.size(df_g, axis=0))
    df_s.index=range(0, np.size(df_s, axis=0))
    df_intensity_all_rel.index=range(0,np.size(df_intensity_all_rel, axis=0))
    df_intensity_cytoplasm_rel.index=range(0,np.size(df_intensity_cytoplasm_rel, axis=0))
    df_intensity_lipofuscin_rel.index=range(0,np.size(df_intensity_lipofuscin_rel, axis=0))
    
    
    df=pd.DataFrame(df)
    df=df_donor.join(df)
    df=df.join(df_g)
    df=df.join(df_s)
    df=df.join(df_intensity_all_rel)
    df=df.join(df_intensity_cytoplasm_rel)
    df=df.join(df_intensity_lipofuscin_rel)
        
    # round at two decimal digits
    df=df.round(2)
    return df

def load_dat(filename=-1):
    print('select .dat file:')
    if filename==-1:
        filename=flim.decode.getfile('dat')[0]
    df=pd.read_csv(filename)
    return df

def save_dat(df, filename=-1):
    if filename==-1:
        filename=flim.decode.asksavefile('dat')
    df.to_csv(filename, index=False)
    
def CreateDonorDB():
       
    # date=20210630
    # sex='M'
    # age=85
    # BMI=27.7
    # SI=1.37
    
    # date=20210714
    # sex='M'
    # age=46
    # BMI=23.67
    # SI=3.9
    
    # date=20210716
    # sex='M'
    # age=80
    # BMI=23.03
    # SI=2.32
    
    date=20210923
    sex='M'
    age=79
    BMI=26.81
    SI=5.35
    
    print('select .R64 file:')
    filename=flim.decode.getfile('R64')
    g, s, intensity, namefile = flim.decode.import_R64(filename=filename)
    
    islet=input('islet (PXHY): ')
    cell_type=input('cell type (alpha/beta): ')
    
    df={
        'filename':[],
        'date':[],
        'islet':[],
        'cell_type':[],
        'cell_number':[],
        'glucose':[],
        'sex':[],
        'age':[],
        'BMI':[],
        'insulin_SI':[]
        }
    
    dim3=np.size(filename)
    
    for i in range(0, dim3):
        df['filename'].append(filename[i])
        df['date'].append(date)
        df['islet'].append(islet)
        df['cell_type'].append(cell_type)
        df['cell_number'].append(i)
        df['glucose'].append(re.findall('\d+', filename[i].split('_')[0])[-1]+'mM')
        df['sex'].append(sex)
        df['age'].append(age)
        df['BMI'].append(BMI)
        df['insulin_SI'].append(SI)
    
    df=pd.DataFrame(df, index=range(0, dim3))
    return df
    
def Get_XY_FromDataFrame(df, target_column):
    target=df[target_column]
    df=df.drop(labels=target_column, axis=1)
    return df, target

def train_test_split(x,y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.2, stratify=y, random_state=1) 
    # X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size =.2, stratify=Y_train, random_state=1) 
    return X_train, X_test, Y_train, Y_test

def train_dev_test_split(x,y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.2, stratify=y, random_state=1) 
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size =.2, stratify=Y_train, random_state=1) 
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

def DecisionTree(x,y):
    from sklearn import tree
    x=flim.OnlyNumeric_FromDataFrame(x)
    # split training and testing sets
    df_train, df_test, target_train, target_test = sklearn.model_selection.train_test_split(x, y, test_size =.3, stratify=y, random_state=1) 
    # model choice and training
    model = tree.DecisionTreeClassifier(random_state=42) 
    model.fit(df_train, target_train)
    # prediction on testing set
    target_predict = model.predict(df_test)
    # evaluation report
    print('='*20,'Training Set Results','='*20)
    print(sklearn.metrics.classification_report(target_train, model.predict(df_train)))
    print('='*20,'Testing Set Results','='*20)
    report_testing_dtree = sklearn.metrics.classification_report(target_test, target_predict)
    print(report_testing_dtree)
    print('='*60)
    # Confusion matrix
    # sklearn.metrics.plot_confusion_matrix(model, df_test, target_test)

def XGBoost(x,y):
    from xgboost import XGBClassifier
    # labels=['date', 'islet', 'cell_number', 'glucose', 'sex']
    # labels=['date', 'islet', 'cell_number']
    # x['sex']=x['sex'].astype('category')
    # x=x.drop(labels=labels, axis=1)
    # split training and testing sets
    # X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.2, stratify=y, random_state=1) 
    
    # encode target vector
    # y=y=='beta'
    # Y_test=Y_test=='beta'
    # model choice and training
    model = XGBClassifier(use_label_encoder=False, class_weight='balanced', objective="binary:logistic", random_state=42)
    model.fit(x,y, eval_metric='logloss')
    return model

def LogisticRegression(x,y):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000).fit(x,y)
    return model

def performance_testing(model, x, y):
    # prediction on testing set
    Y_predict = model.predict(x)
    # evaluation report
    # print('='*20,'Performance Results','='*20)
    report_testing = sklearn.metrics.classification_report(y, Y_predict)
    print(report_testing)
    # print('='*60)
    # Confusion matrix
    # sklearn.metrics.plot_confusion_matrix(model, df_test, target_test)



def GetBestFeatures(X_train, X_dev, Y_train, Y_dev):
    
    # X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.2, stratify=y, random_state=1) 
    # X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size =.2, stratify=Y_train, random_state=1) 
    # model, X_train_dev, X_test, Y_train_dev, Y_test = XGBoost(x, y)
    # X_train, X_dev, Y_train, Y_dev = train_test_split(x, y)
    model=_XGBoost_learn(X_train, Y_train)
    # performance_testing(model, X_train, Y_train)
    
    x_columns=list(X_train.columns)
    metrics=[]
    
    a=pd.DataFrame([])
    a['columns']=X_train.columns
    a['score']=model.feature_importances_
    a=a.sort_values(by='score', axis=0, ascending=False)
    
    for i in range(np.size(x_columns), 0, -1):
        print('feature', np.size(x_columns)-(i-1), ' of ', np.size(x_columns))
        cols=list(a['columns'].iloc[0:i+1].values)
        X_train=X_train[cols]
        X_dev=X_dev[cols]
        # boosted decision tree
        model = _XGBoost_learn(X_train, Y_train)
        metrics.append(sklearn.metrics.precision_score(Y_dev, model.predict(X_dev), average=None))
    
    metrics=np.array(metrics)
    best=int(np.min(np.argwhere(metrics[:,0]==np.max(metrics[:,0]))))    
    cols=list(a['columns'].iloc[0:best+1].values)
    
    plt.figure(figsize=(10,4))
    plt.plot(metrics)
    plt.legend(['alpha', 'beta'])
    plt.show()
    
    print('max alpha precision: ',np.max(metrics[:,0]))
    print('max beta precision: ',np.max(metrics[:,1]))
    cols=list(a['columns'])   
    return cols, best

def FeatureWeights(model, X_train):    
    # plot features importance
    plt.figure(figsize=(20,4))
    pd.Series(dict(zip(X_train.columns,model.feature_importances_ ))).plot(kind='bar')
    
def Zscore(df):
    # normalize by z-score
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    df_scaled=df.iloc[:,6:]
    df_scaled=scaler.fit_transform(df_scaled)
    df_scaled=pd.DataFrame(df_scaled, columns=df.iloc[:,6:].columns)
    df=df.iloc[:,0:6].join([df_scaled])
    return df

def MinMaxScaler(df):
    # normalize by z-score
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    df_scaled=df.iloc[:,6:]
    df_scaled=scaler.fit_transform(df_scaled)
    df_scaled=pd.DataFrame(df_scaled, columns=df.iloc[:,6:].columns)
    df=df.iloc[:,0:6].join([df_scaled])
    return df
    
def describe(x):
    '''
    computes main descriptive statistics paramters. Returns a DataFrame object
    x: vector or matrix 
    '''
    if np.size(x)==0 or np.all(x==0):
        df={
            'min':np.nan,
            'max':np.nan,
            'mean':np.nan,
            'mode':np.nan,
            'std':np.nan,
            'CV':np.nan,
            'IQR':np.nan,
            'whisker_low':np.nan,
            'whisker_high':np.nan
            }
        return pd.DataFrame(df, index=[0])
    # initialize output variable
    df={}
    # reshape in 1D element
    x=flim.clean(x)
    # min, max
    df['min']=np.min(x)
    df['max']=np.max(x)
    # mean, mode
    df['mean']=np.mean(x)
    df['mode']=flim.mode(x)
    # std, se, cv
    df['std']=np.std(x)
    # df['SE']=np.array(df['std'])/np.sqrt(np.size(x)) #hidden because very low value, which rounded gives zero
    df['CV']=df['std']/df['mean']
    # percentiles, IQR, whiskers
    percent=[25, 50, 75, 99]
    for i in percent:
        df[str(i)]=np.percentile(x, i)
    df['IQR']=df['75']-df['25']
    df['whisker_low']=df['IQR']-1.5*df['25']
    df['whisker_high']=df['IQR']+1.5*df['75']
    # CI
    CI=[67, 95, 99]
    df_CI=flim.ConfidenceInterval(x, CI)
    df.update(df_CI)
    # create DataFrame
    df=pd.DataFrame(df, index=[0])
    df=np.round(df, 2)
    return df
    
# %% Load data
filename='G:/My Drive/PhD\CAPTUR3D_personal/00 Progetti/beta cell recognize/HI_features.dat'
# filename='C:/Users/Fabio/Desktop/beta cell recognize/HI_features_reduced.dat'
df=load_dat(filename)

# %% Exploratory Data Analysis

# pairplot
# df_pp=df.sort_values(by='cell_type', ascending=False)
# sns.pairplot(data=df_pp, hue='cell_type')

# # check missing values
# df.isna().sum() # -> only cells with zero lipofuscin display missing values

# # boxplot to assess xlabel variability
# if not os.path.exists('C:/Users/Fabio/Desktop/boxplot'):
#     os.mkdir('C:/Users/Fabio/Desktop/boxplot_')
# for i in list(df.columns)[10:]:
#     sns.boxplot(data=df, x='cell_type', y=i)
#     flim.decode.savefigure(name='/boxplot/boxplot_'+i,save=-1)
#     plt.show()

# xlabel='date'
# if not os.path.exists('C:/Users/Fabio/Desktop/boxplot_'+xlabel):
#     os.mkdir('C:/Users/Fabio/Desktop/boxplot_'+xlabel)
# for i in list(df.columns)[6:]:
#     sns.boxplot(data=x, x=xlabel, y=i, hue='cell_type')
#     flim.decode.savefigure(name='/boxplot_'+xlabel+'/boxplot_'+i,save=-1)
#     plt.show()

    
# # violin plot
# celltype='beta'
# sns.violinplot(data=df[df['cell_type']==celltype], x=xlabel, y='oxphos', hue="glucose", split=True)
# plt.title(celltype+' cells')
# flim.decode.savefigure(name=celltype,save=-1)

# # check variable distribution (normality, skewness)
# if not os.path.exists('C:/Users/Fabio/Desktop/hist'):
#     os.mkdir('C:/Users/Fabio/Desktop/hist')
# for i in list(df.columns)[6:]:
#     sns.kdeplot(df, x=i, hue='cell_type')
#     # sns.histplot(df, x=i, hue='cell_type')
#     plt.ylabel(i)
#     flim.decode.savefigure(name='/hist/hist_'+i,save=-1)
#     plt.show()

# # check variable distribution using kde
# if not os.path.exists('C:/Users/Fabio/Desktop/kde'):
#     os.mkdir('C:/Users/Fabio/Desktop/kde')
# for i in list(df.columns)[6:]:
#     sns.kdeplot(data=df, x=i, hue="cell_type")
#     plt.ylabel(i)
#     flim.decode.savefigure(name='/kde/kde_'+i,save=-1)
#     plt.show()

# size vs intensity
# sns.scatterplot(data=df, x='cell_area', y='intensity_all_rel_mean', hue='cell_type')

# %% pre-processing
# manual feature selection

# missing values handling
df=df.fillna(0)

# scaling
# df=Zscore(df)
df=MinMaxScaler(df)
df=df.round(2)

# categorical features encoding
dict_glucose={'2mM':0, '16mM':1}
df['glucose']=df['glucose'].replace(dict_glucose)
df=df.replace(['alpha', 'beta'], [0,1])

# get x and y
x,y=Get_XY_FromDataFrame(df, 'cell_type')
x=x.iloc[:,3:]
x=x.drop(labels='sex', axis=1)

# # outliers handling
# find outliers with zscore
# x_scaled=Zscore(x)
# x=x.where(np.abs(x_scaled)<=3)

# find outliers with LOF
from sklearn.neighbors import LocalOutlierFactor
for i in range(0, len(list(x.iloc[:,5:].columns))):
    clf = LocalOutlierFactor(n_neighbors=200)
    outliers=clf.fit_predict(np.reshape(x.iloc[:,i].values,(-1,1)))
    x.iloc[:,i]=x.iloc[:,i].where(outliers==1)


# outliers imputation
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(x)
x=imp.transform(x)

# # drop nans
# x=x.dropna()
# y=y.loc[x.index]
# df_s=df.loc[x.index]
# xlabel='date'
# if not os.path.exists('C:/Users/Fabio/Desktop/boxplot_'+xlabel):
#     os.mkdir('C:/Users/Fabio/Desktop/boxplot_'+xlabel)
# for i in list(df.columns)[6:]:
#     sns.boxplot(data=df_s, x=xlabel, y=i, hue='cell_type')
#     flim.decode.savefigure(name='/boxplot_'+xlabel+'/boxplot_'+i,save=-1)
#     plt.show()

# df_s['cell_type']=df_s['cell_type'].replace({0: 'alpha', 1:'beta'})
# for i in list(df_s.iloc[:,10:].columns):
#     sns.boxplot(data=df_s, x='cell_type', y=i)
#     flim.decode.savefigure(name='/boxplot/boxplot_'+i, save=-1)
#     plt.show()

# %% Unsupervised learning
#### PCA #######
# # normalize by z-score
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# df_scaled=df.iloc[:,5:]
# df_scaled=scaler.fit_transform(df_scaled)
# df_scaled=pd.DataFrame(df_scaled, columns=df.iloc[:,5:].columns)

# # choose number of components
# from sklearn.decomposition import PCA
# n_components=np.size(df_scaled, axis=1)
# pca10=PCA(n_components=n_components)
# pc_fit = pca10.fit_transform(df_scaled.values)

# # plot explained variance as function of number of components
# plt.figure(1, figsize=(14, 7))
# plt.bar(range(1,n_components+1,1), pca10.explained_variance_ratio_, alpha=0.5, align='center',
#         label='individual explained variance')
# plt.step(range(1,n_components+1,1),pca10.explained_variance_ratio_.cumsum(), where='mid',
#          label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.title("Around 95% of variance is explained by the Fisrt 10 components ");
# plt.legend(loc='best')
# plt.axhline(y=0.7, color='r', linestyle='-') # 70% of  explained variance
# plt.tight_layout()  

# # perform PCA
# n_components=3
# pca=PCA(n_components=n_components)
# pc = pca.fit_transform(df_scaled.values)
# columns=[]
# for i in range(1, n_components+1):
#     columns.append('PC'+str(i))
# pc_df=pd.DataFrame(pc, columns=columns)  
# pc_df['cell_type']=df['cell_type']
# # sns.scatterplot(x='PC1', y='PC2', data=pc2_df, hue='cell_type')

# # 3D plot
# from mpl_toolkits.mplot3d import Axes3D # 3D scatter plot
# fig = plt.figure(figsize=(12,7))
# ax = Axes3D(fig) 

# cmap = {'alpha':'orange','beta':'green'}
# ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=[cmap[c] for c in  pc_df['cell_type'].values],
#            marker='o', s=20)

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.view_init(30,-110)
# plt.show()

#### clustering ####

# %% Supervised learning

#### dataset splitting
# X_train, X_dev, X_test, Y_train, Y_dev, Y_test = train_dev_test_split(x, y)
X_train, X_test, Y_train, Y_test = train_test_split(x, y)

#### cross validation

#### grid search

# %%% Logistic Regression
print('Logistic Regression')
model_lr=LogisticRegression(X_train, Y_train)
performance_testing(model_lr, X_test, Y_test)


# %%% XGBoost

# select best features
# cols, index =GetBestFeatures(X_train, X_dev, Y_train, Y_dev)
# index=32
# cols=['intensity_cytoplasm_rel_CV',
#  'g_barycenter_std',
#  's_IQR',
#  'BMI',
#  'age',
#  'intensity_cytoplasm_rel_CI_95_max',
#  'intensity_cytoplasm_rel_CI_99_max',
#  'g_min',
#  'g_CI_67_min',
#  'intensity_lipofuscin_rel_CV',
#  'intensity_all_rel_50',
#  'g_IQR',
#  'intensity_cytoplasm_rel_IQR',
#  'g_75',
#  's_barycenter_std',
#  's_25',
#  'intensity_all_rel_CI_67_min',
#  's_CV',
#  'g_whisker_low',
#  'intensity_all_rel_std',
#  'g_CI_95_max',
#  'cell_perimeter',
#  'g_50',
#  's_CI_95_min',
#  'intensity_lipofuscin_rel_min',
#  'intensity_lipofuscin_rel_50',
#  'intensity_cytoplasm_rel_CI_67_max',
#  's_CI_99_max',
#  's_mode',
#  'intensity_cytoplasm_rel_99',
#  'intensity_all_rel_75',
#  'g_CI_95_min',
#  'intensity_all_rel_CI_95_min',
#  'intensity_all_rel_IQR',
#  'intensity_all_rel_whisker_low',
#  'g_whisker_high',
#  'intensity_cytoplasm_rel_75',
#  'cell_area',
#  's_CI_95_max',
#  'intensity_cytoplasm_rel_max',
#  's_whisker_low',
#  'g_CI_99_min',
#  'g_barycenter',
#  'cell_circularity',
#  'intensity_all_rel_CI_67_max',
#  's_CI_67_min',
#  'intensity_all_rel_mode',
#  'intensity_all_rel_CI_95_max',
#  'intensity_lipofuscin_rel_mode',
#  'intensity_cytoplasm_rel_whisker_low',
#  'intensity_lipofuscin_rel_75',
#  'intensity_lipofuscin_rel_whisker_high',
#  'intensity_all_rel_CI_99_max',
#  's_max',
#  'intensity_all_rel_25',
#  'g_max',
#  'intensity_all_rel_99',
#  'g_99',
#  'g_mode',
#  'g_CI_99_max',
#  's_99',
#  'intensity_lipofuscin_rel_CI_95_max',
#  's_whisker_high',
#  's_min',
#  'intensity_lipofuscin_rel_CI_99_max',
#  's_CI_99_min',
#  's_CI_67_max',
#  'intensity_cytoplasm_rel_CI_67_min',
#  'g_CV',
#  'intensity_all_rel_min',
#  's_barycenter',
#  'intensity_lipofuscin_rel_IQR',
#  'insulin_SI',
#  'intensity_all_rel_CV',
#  'intensity_lipofuscin_rel_99',
#  'lipofuscin_area_rel',
#  'intensity_cytoplasm_rel_50',
#  'intensity_lipofuscin_rel_mean',
#  's_50',
#  'g_25',
#  's_75',
#  'intensity_lipofuscin_rel_whisker_low',
#  'intensity_cytoplasm_rel_std',
#  'intensity_cytoplasm_rel_CI_95_min',
#  'intensity_cytoplasm_rel_mean',
#  'oxphos',
#  'intensity_all_rel_CI_99_min',
#  'intensity_lipofuscin_rel_CI_67_max',
#  'intensity_lipofuscin_rel_25',
#  'intensity_lipofuscin_rel_std',
#  'oxphos_nolipo',
#  'intensity_cytoplasm_rel_whisker_high',
#  'g_CI_67_max',
#  'intensity_all_rel_mean',
#  'intensity_lipofuscin_rel_CI_95_min',
#  'intensity_cytoplasm_rel_CI_99_min',
#  'intensity_lipofuscin_rel_CI_67_min',
#  'glucose',
#  'intensity_cytoplasm_rel_25',
#  'intensity_cytoplasm_rel_mode',
#  'intensity_all_rel_whisker_high',
#  'intensity_lipofuscin_rel_max',
#  'intensity_cytoplasm_rel_min',
#  'intensity_all_rel_max',
#  's_std',
#  's_mean',
#  'g_std',
#  'g_mean',
#  'intensity_lipofuscin_rel_CI_99_min']

# cols=cols[0:index+1]

# x=x[cols]
# X_train, X_test, Y_train, Y_test = train_test_split(x, y)
print('XGBoost')
model_xgb = XGBoost(X_train, Y_train)
performance_testing(model_xgb, X_test, Y_test)

# %%% SVM


# %% Error analysis
# Y_pred=model.predict(X_test)
# # create misclassified alpha dataframe
# Y_pred=pd.Series(Y_pred, index=Y_test.index)
# Y_error=pd.concat([Y_test, Y_pred], axis=1)
# Y_error.columns=['test', 'pred']
# # Y_error=Y_error[np.logical_xor(Y_error['test'],Y_error['pred'])]
# # Y_error=Y_error[Y_error['test']=='alpha']
# # Y_error=Y_error[Y_error['pred']=='beta']
# Y_error=Y_error[np.logical_xor(Y_error['test'],Y_error['pred'])]
# # Y_error=Y_error.replace([0,1],['alpha', 'beta'])
# df_error=df.loc[Y_error.index]

