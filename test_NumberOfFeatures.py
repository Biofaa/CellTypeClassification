# %% Libraries and functions
import sys
from pathlib import Path
path_root=Path.cwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flim_module as flim
import seaborn as sns
import re #for regular expressions
import sklearn
import sklearn.model_selection
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, classification_report, confusion_matrix 
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import ElasticNet
import joblib
import imblearn

def load_R64_bkp(use_donor_DB=False):
    '''
    Load simFCS R64 files, computes features and creates a DataFrame.
    If use_donor_DB=true, the function takes as input a csv with a list of file paths
    
    '''
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
    
    # prepare data for features extraction
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
        intensity_rel=intensity
        # intensity_rel=intensity/np.median(intensity) # normalize by max value
        df_intensity_all_rel_tmp=describe(intensity_rel)
        df_intensity_all_rel=pd.concat([df_intensity_all_rel, df_intensity_all_rel_tmp])
        
        # intensity_lipofuscin_rel=np.where(intensity>threshold_high, intensity, 0)/np.median(intensity)
        intensity_lipofuscin_rel=np.where(intensity>threshold_high, intensity, 0)
        df_intensity_lipofuscin_rel_tmp=describe(intensity_lipofuscin_rel)
        df_intensity_lipofuscin_rel=pd.concat([df_intensity_lipofuscin_rel, df_intensity_lipofuscin_rel_tmp])
        
        # intensity_cytoplasm_rel=np.where(intensity<threshold_high, intensity, 0)/np.median(intensity)
        intensity_cytoplasm_rel=np.where(intensity<threshold_high, intensity, 0)
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

def load_R64(use_donor_DB=False):
    '''
    Load simFCS R64 files, computes features and creates a DataFrame.
    If use_donor_DB=true, the function takes as input a csv with a list of file paths
    
    '''
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
    
    # prepare data for features extraction
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
    
    g_har2_barycenter=[]
    g_har2_barycenter_std=[]
    s_har2_barycenter=[]
    s_har2_barycenter_std=[]
    oxphos_nolipo_har2=[]
    df_g_har2=pd.DataFrame([])
    df_s_har2=pd.DataFrame([])
    
    for i in range(0, dim3):
        print(str(i+1)+' of '+str(dim3))
        g,s,intensity, namefile, g_har2, s_har2 =flim.decode.import_R64(har2=True, filename=filename[i])
        threshold_high=flim.LipofuscinThresholdCalculus(intensity)
      
        ## calculate descriptive statistics on g and s matrices
        df_g_tmp=describe(g)
        df_g=pd.concat([df_g, df_g_tmp])
        
        df_g_har2_tmp=describe(g_har2)
        df_g_har2=pd.concat([df_g_har2, df_g_har2_tmp])
        
        df_s_tmp=describe(s)
        df_s=pd.concat([df_s, df_s_tmp])
        
        df_s_har2_tmp=describe(s_har2)
        df_s_har2=pd.concat([df_s_har2, df_s_har2_tmp])
        
        ## intensity
        intensity_rel=intensity
        # intensity_rel=intensity/np.median(intensity) # normalize by max value
        df_intensity_all_rel_tmp=describe(intensity_rel)
        df_intensity_all_rel=pd.concat([df_intensity_all_rel, df_intensity_all_rel_tmp])
        
        # intensity_lipofuscin_rel=np.where(intensity>threshold_high, intensity, 0)/np.median(intensity)
        intensity_lipofuscin_rel=np.where(intensity>threshold_high, intensity, 0)
        df_intensity_lipofuscin_rel_tmp=describe(intensity_lipofuscin_rel)
        df_intensity_lipofuscin_rel=pd.concat([df_intensity_lipofuscin_rel, df_intensity_lipofuscin_rel_tmp])
        
        # intensity_cytoplasm_rel=np.where(intensity<threshold_high, intensity, 0)/np.median(intensity)
        intensity_cytoplasm_rel=np.where(intensity<threshold_high, intensity, 0)
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
        
        g_har2_barycenter_tmp, s_har2_barycenter_tmp=flim.barycenter(g_har2, s_har2)
        g_har2_barycenter.append(g_har2_barycenter_tmp)
        s_har2_barycenter.append(s_har2_barycenter_tmp)
        
        g_har2_barycenter_std_tmp, s_har2_barycenter_std_tmp=flim.barycenter_std(g_har2, s_har2)
        g_har2_barycenter_std.append(g_har2_barycenter_std_tmp)
        s_har2_barycenter_std.append(s_har2_barycenter_std_tmp)
        
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
    cols_g_har2=list(df_g_har2.columns)
    cols_s_har2=list(df_s_har2.columns)
    cols_int_all=list(df_intensity_all_rel.columns)
    cols_int_cyt=list(df_intensity_cytoplasm_rel.columns)
    cols_int_lipo=list(df_intensity_lipofuscin_rel.columns)
    
    for i in range(0, np.size(df_g.columns)):
        cols_g[i]='g_'+cols_g[i]
        cols_s[i]='s_'+cols_s[i]
        cols_g_har2[i]='g_har2_'+cols_g_har2[i]
        cols_s_har2[i]='s_har2_'+cols_s_har2[i]
        cols_int_all[i]='intensity_all_rel_'+cols_int_all[i]
        cols_int_cyt[i]='intensity_cytoplasm_rel_'+cols_int_cyt[i]
        cols_int_lipo[i]='intensity_lipofuscin_rel_'+cols_int_lipo[i]
        
    df_g.columns=cols_g
    df_s.columns=cols_s
    df_g_har2.columns=cols_g_har2
    df_s_har2.columns=cols_s_har2
    df_intensity_all_rel.columns=cols_int_all
    df_intensity_lipofuscin_rel.columns=cols_int_lipo
    df_intensity_cytoplasm_rel.columns=cols_int_cyt
    
    #reshape otherwise you'll obtain a list of lists
    g_barycenter=np.reshape(g_barycenter, np.size(g_barycenter, axis=0))
    g_barycenter_std=np.reshape(g_barycenter_std, np.size(g_barycenter_std, axis=0))
    s_barycenter=np.reshape(s_barycenter, np.size(s_barycenter, axis=0))
    s_barycenter_std=np.reshape(s_barycenter_std, np.size(s_barycenter_std, axis=0))
    
    g_har2_barycenter=np.reshape(g_har2_barycenter, np.size(g_barycenter, axis=0))
    g_har2_barycenter_std=np.reshape(g_har2_barycenter_std, np.size(g_barycenter_std, axis=0))
    s_har2_barycenter=np.reshape(s_har2_barycenter, np.size(s_barycenter, axis=0))
    s_har2_barycenter_std=np.reshape(s_har2_barycenter_std, np.size(s_barycenter_std, axis=0))
    
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
        'lipofuscin_area_rel': lipofuscin_area_rel,
        'g_har2_barycenter': g_har2_barycenter,
        's_har2_barycenter': s_har2_barycenter,
        'g_har2_barycenter_std': g_har2_barycenter_std,
        's_har2_barycenter_std': s_har2_barycenter_std
        }
    
    # # adjust indices for join
    df_g.index=range(0, np.size(df_g, axis=0))
    df_s.index=range(0, np.size(df_s, axis=0))
    df_g_har2.index=range(0, np.size(df_g_har2, axis=0))
    df_s_har2.index=range(0, np.size(df_s_har2, axis=0))
    df_intensity_all_rel.index=range(0,np.size(df_intensity_all_rel, axis=0))
    df_intensity_cytoplasm_rel.index=range(0,np.size(df_intensity_cytoplasm_rel, axis=0))
    df_intensity_lipofuscin_rel.index=range(0,np.size(df_intensity_lipofuscin_rel, axis=0))
    
    
    df=pd.DataFrame(df)
    df=df_donor.join(df)
    df=df.join(df_g)
    df=df.join(df_s)
    df=df.join(df_g_har2)
    df=df.join(df_s_har2)
    df=df.join(df_intensity_all_rel)
    df=df.join(df_intensity_cytoplasm_rel)
    df=df.join(df_intensity_lipofuscin_rel)
        
    # round at two decimal digits
    df=df.round(2)
    return df



def load_dat(filename=-1):
    '''
    Load the dataset
    '''
    print('select .dat file:')
    if filename==-1:
        filename=flim.decode.getfile('dat')[0]
    df=pd.read_csv(filename)
    return df

def save_dat(df, filename=-1):
    '''
    Save the dataset
    '''
    if filename==-1:
        filename=flim.decode.asksavefile('dat')
    df.to_csv(filename, index=False)
    
def CreateDonorDB():
    '''
    creates a DataFrame with donor identifiers
    '''   
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
    '''
    splits the dataset into X (features) and Y (target)
    '''
    target=df[target_column]
    df=df.drop(labels=target_column, axis=1)
    return df, target

def train_test_split(x,y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.25, stratify=y, random_state=42) 
    # X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size =.2, stratify=Y_train, random_state=1) 
    return X_train, X_test, Y_train, Y_test

def train_dev_test_split(x,y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.3, stratify=y, random_state=42) 
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size =.3, stratify=Y_train, random_state=1) 
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
    
    # WARNING: i did not implemented  class_weight='balanced' since in my
    # trials it did not affect the score but reaised a warning
    model = XGBClassifier(use_label_encoder=False, objective="binary:logistic", random_state=42)
    model.fit(x,y, eval_metric='logloss')
    return model

# def LogisticRegression(x,y):
#     from sklearn.linear_model import LogisticRegression

#     model = LogisticRegression(max_iter=1000).fit(x,y)
#     return model

def performance_testing(model, x, y):
    '''
    calculates precision, recall, f1 and accuracy on x
    '''
    # prediction on testing set
    Y_predict = model.predict(x)
    # evaluation report
    # print('='*20,'Performance Results','='*20)
    report_testing = sklearn.metrics.classification_report(y, Y_predict)
    print(report_testing)
    # print('='*60)
    # Confusion matrix
    # sklearn.metrics.plot_confusion_matrix(model, df_test, target_test)

def performance_scores(model, x, y, compact=False):
    '''
    calculates precision, recall, f1, accuracy, ROC_AUC, Matthews Correlation Coefficient, balanced accuracy on x
    if compact=True, the global metrics will be shown, otherwise the score is given for each class
    '''
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
    from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
    # prediction on testing set
    Y_predict = model.predict(x)
    
    if compact:
        scores={
            'precision':precision_score(y, Y_predict),
            'recall': recall_score(y, Y_predict),
            'f1': f1_score(y, Y_predict),
            'accuracy': accuracy_score(y, Y_predict),
            'roc_auc': roc_auc_score(y, Y_predict),
            'mcc': matthews_corrcoef(y, Y_predict),
            'cohen_kappa': cohen_kappa_score(y, Y_predict),
            'balanced_accuracy':balanced_accuracy_score(y, Y_predict),
            }
        
        df=pd.DataFrame(scores, index=[0])
        
    else:
        scores={
            'precision':precision_score(y, Y_predict, average=None),
            'recall': recall_score(y, Y_predict, average=None),
            'f1': f1_score(y, Y_predict, average=None),
            # 'balanced_accuracy':balanced_accuracy_score(y, Y_predict, average=None),
            'roc_auc': roc_auc_score(y, Y_predict, average=None),
            # 'matthews': matthews_corrcoef(y, Y_predict, average=None)
            }
        
        df=pd.DataFrame(scores, index=np.arange(0, y.drop_duplicates().count()))

    # precision=precision_score(y, Y_predict, average=None)
    # recall=precision_score(y, Y_predict)
    # f1=f1_score(y, Y_predict)
    # balanced_accuracy=balanced_accuracy_score(y, Y_predict)
    # roc_auc=roc_auc_score(y, Y_predict)
    # mcc=matthews_corrcoef(y, Y_predict)
    
    # df=pd.DataFrame(scores, index=np.arange(0, y.drop_duplicates().count()))
    df=np.round(df, 2).T
    return df
    

# def GetBestFeatures(X_train, X_dev, Y_train, Y_dev):
    
#     # X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size =.2, stratify=y, random_state=1) 
#     # X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size =.2, stratify=Y_train, random_state=1) 
#     # model, X_train_dev, X_test, Y_train_dev, Y_test = XGBoost(x, y)
#     # X_train, X_dev, Y_train, Y_dev = train_test_split(x, y)
#     model=_XGBoost_learn(X_train, Y_train)
#     # performance_testing(model, X_train, Y_train)
    
#     x_columns=list(X_train.columns)
#     metrics=[]
    
#     a=pd.DataFrame([])
#     a['columns']=X_train.columns
#     a['score']=model.feature_importances_
#     a=a.sort_values(by='score', axis=0, ascending=False)
    
#     for i in range(np.size(x_columns), 0, -1):
#         print('feature', np.size(x_columns)-(i-1), ' of ', np.size(x_columns))
#         cols=list(a['columns'].iloc[0:i+1].values)
#         X_train=X_train[cols]
#         X_dev=X_dev[cols]
#         # boosted decision tree
#         model = _XGBoost_learn(X_train, Y_train)
#         metrics.append(sklearn.metrics.precision_score(Y_dev, model.predict(X_dev), average=None))
    
#     metrics=np.array(metrics)
#     best=int(np.min(np.argwhere(metrics[:,0]==np.max(metrics[:,0]))))    
#     cols=list(a['columns'].iloc[0:best+1].values)
    
#     plt.figure(figsize=(10,4))
#     plt.plot(metrics)
#     plt.legend(['alpha', 'beta'])
#     plt.show()
    
#     print('max alpha precision: ',np.max(metrics[:,0]))
#     print('max beta precision: ',np.max(metrics[:,1]))
#     cols=list(a['columns'])   
#     return cols, best

def FeatureWeights(model, X_train): 
    '''
    calculates feature importance in models that have .feature_importances_ method
    '''
    # plot features importance
    plt.figure(figsize=(20,4))
    pd.Series(dict(zip(X_train.columns,model.feature_importances_ ))).plot(kind='bar')

class scaling:  
    def Zscore(df):
        # normalize by z-score
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()
        # df_scaled=df.iloc[:,6:]
        df_scaled=scaler.fit_transform(df)
        df_scaled=pd.DataFrame(df_scaled, columns=df.columns)
        # df=df.iloc[:,0:6].join([df_scaled])
        return df
    
    def MinMaxScaler(df):
        # normalize by z-score
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        # df_scaled=df.iloc[:,6:]
        df_scaled=scaler.fit_transform(df)
        df_scaled=pd.DataFrame(df_scaled, columns=df.columns)
        # df=df.iloc[:,0:6].join([df_scaled])
        return df_scaled
    
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

def DataTransform(x, y):
    '''performs data transformation
        - encoding of categorical features
        - missing values handling
        - outliers handling
        - scaling
    '''
    #### manual feature selection
    
    #### categorical features encoding
    dict_glucose={'2mM':0, '16mM':1}
    x['glucose']=x['glucose'].replace(dict_glucose)
    y=y.replace(['alpha', 'beta'], [0,1])
    
    #### missing values handling
    x=x.fillna(0)
    
    #### outliers handling
    # # find outliers with zscore
    # x_scaled=Zscore(x)
    # x=x.where(np.abs(x_scaled)<=3)

    # # find outliers with LOF
    from sklearn.neighbors import LocalOutlierFactor
    for i in range(0, len(list(x.iloc[:,5:].columns))):
        clf = LocalOutlierFactor(n_neighbors=200)
        outliers=clf.fit_predict(np.reshape(x.iloc[:,5+i].values,(-1,1)))
        x.iloc[:,5+i]=x.iloc[:,5+i].where(outliers==1)
        
    # outliers imputation
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_imputed=imp.fit_transform(x)
    x=pd.DataFrame(x_imputed, columns=list(x.columns))
        
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
    return x, y

class MultiCollinearity:
    '''
    perform feature selection by removing highly correlated features
    '''
    def fit(df, threshold=0.8):
        # create correlation matrix
        corr_matrix = df.corr().abs()
        
        # Select upper triangle of correlation matrix
        u = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in u.columns if any(u[column] > threshold) or any(u[column] < -(threshold))]
        return to_drop
        
    def transform(df, to_drop):
        # drop highly correlated features
        df = df.drop(df[to_drop], axis=1)
        return df
    
    def fit_transform(df, threshold=0.8):
        to_drop=MultiCollinearity.fit(df, threshold)
        df=MultiCollinearity.transform(df, to_drop)
        return df
    
class MyElasticNet:
    def fit(X, Y, cv=5):
        # ElasticNet
        alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]
        l1_ratios = [1, 0.9, 0.8, 0.7, 0.5]
        
        import itertools
        from sklearn.model_selection import cross_val_score 
        
        elastics=[]
        cv_elastic=[]
        
        idx=list(itertools.product(alphas, l1_ratios))
        
        for (alpha, l1_ratio) in idx:
            elastics.append(ElasticNet(alpha = alpha, l1_ratio=l1_ratio, max_iter=5000))
            cv_elastic.append(cross_val_score(elastics[-1], X_train, Y_train, cv=cv, scoring='roc_auc').mean())
        
        # plot results to find best alpha and l1_ratio     
        # plt.figure(figsize=(10,2))
        # pd.Series(cv_elastic, index=idx).plot()
        
        cv_elastic=pd.Series(cv_elastic)
        i=cv_elastic[cv_elastic==np.max(np.abs(cv_elastic))].index.values[-1]
        alpha=idx[i][0]
        l1_ratio=idx[i][1]
        # alpha=0.01
        # l1_ratio=0.5
    
        elastic=ElasticNet(alpha = 0.01, l1_ratio=0.5)
        elastic.fit(X_train, Y_train)
        coefs=pd.Series(elastic.coef_, index = X_train.columns)
        coefs=list(coefs[np.abs(coefs)>0].index)
        return coefs
    
    def transform(X, coefs):
        X=X[coefs]
        return X
    
    def fit_transform(X):
        coefs=MyElasticNet.fit()
        X_reduced=MyElasticNet.transform(X, coefs)
        return X_reduced
    
class MyLasso():
    def fit(X, Y, cv):
        from sklearn.linear_model import LassoCV
        alphas=np.arange(0, 1, 0.01) #set lambda parameter of regularization
        model_lasso = LassoCV(alphas=alphas, cv=cv, n_jobs=-1)
        model_lasso.fit(X, Y)
        coefs=pd.Series(model_lasso.coef_, index = X.columns)
        coefs=list(coefs[np.abs(coefs)>0].index)
        return coefs
    def transform(X, coefs):
        X=X[coefs]
        return X
    def fit_transform(X, Y):
        coefs=MyLasso.fit(X, Y)
        X=MyLasso.transform(X, coefs)
        return X, Y
    
def save_model(model, filename=-1):
    '''
    Save a trained model on .pkl file
    '''
    if filename==-1:
        filename=flim.decode.asksavefile('pkl')
    joblib.dump(model, filename)
    
def load_model(filename=-1):
    '''
    loads a trained model from .pkl file
    '''
    print('choose a saved model to open (.pkl)')
    if filename==-1:
        filename=flim.decode.getfile('pkl')[0]
    model=joblib.load(filename)
    return model
    
    
#%% config variables
compact_score=False #if false, you'll obtain a separate score for alpha and beta cells
reduce_dataset=True
xgb_feature_selection=True
save_model_bool=True
model_name='xgb_optuna_FeatureSelection_0.001.pkl'
feature_threshold=None

# %% Load data

# load dataset
filename=path_root/'data'/'HI_features.dat'
df=load_dat(filename)
# load model
model_path=path_root/'models'/model_name
model = load_model(model_path)

# clean dataset by hand
if reduce_dataset:
    # load rows to exclude from csv/xlsx file
    filename_todrop=path_root/'data'/'ErrorAnalysis_rowstodrop.csv'
    df_todrop=pd.read_csv(filename_todrop)
    df=pd.concat([df, df_todrop]).drop_duplicates(subset=list(df_todrop.columns), keep=False)

# %% pre-processing

#### get x and y (extract only features)
x,y=Get_XY_FromDataFrame(df, 'cell_type')
x=x.iloc[:,3:]
x=x.drop(labels='sex', axis=1)

#### dataset splitting
X_train, X_test, Y_train, Y_test = train_test_split(x, y)

#### transform data: scaling, categorical features encoding etc
X_train, Y_train = DataTransform(X_train, Y_train)
X_test, Y_test = DataTransform(X_test, Y_test)

#### drop almost all features
# cols=['g_barycenter_std', 'g_CV', 'g_IQR', 'g_std']
# cols=['g_barycenter_std', 'g_CV', 'g_IQR', 'g_std', 'cell_circularity', 'g_99', 'g_CI_95_max', 'g_max', 'g_mean', 'g_mode', 'g_whisker_high', 'intensity_cytoplasm_rel_CV', 'lipofuscin_area_rel', 's_mode']
# X_train=X_train[cols]
# X_test=X_test[cols]

#### scaling
# Zscore
# x=scaling.Zscore(x)
# x=x.dropna()
# y=y.loc[x.index]

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test=pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#### cross validation
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

#### Feature Selection
# Multicollinearity
# to_drop=MultiCollinearity.fit(X_train, 0.90)
# X_train=MultiCollinearity.transform(X_train, to_drop)
# X_test=MultiCollinearity.transform(X_test, to_drop)

# Elastic Net
# coefs=MyElasticNet.fit(X_train, Y_train, cv)
# X_train=MyElasticNet.transform(X_train, coefs)
# X_test=MyElasticNet.transform(X_test, coefs)

# LASSO
# coefs=MyLasso.fit(X_train, Y_train, cv)
# X_train=MyLasso.transform(X_train, coefs)
# X_test=MyLasso.transform(X_test, coefs)

#### Balance classes
# SMOTE
oversample = imblearn.over_sampling.SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)

# %% performance evaluation

if xgb_feature_selection:
    # load original xgb model
    model_path=path_root/'models'/'xgb_optuna.pkl'
    model_xgb_old=load_model(model_path)
    
    # select most important features
    xgb_features=pd.DataFrame(model_xgb_old.feature_importances_, index=list(x.columns))
    if feature_threshold==0:
        xgb_best_features=list(xgb_features[xgb_features.values!=0].index)
    elif feature_threshold==None:
        xgb_best_features=list(xgb_features.index)
    else:        
        xgb_best_features=list(xgb_features[xgb_features.values>=feature_threshold].index)
    
    # cut dataset features
    x=x[xgb_best_features]
    X_train=X_train[xgb_best_features]
    X_test=X_test[xgb_best_features]

# compute score
print('\nnumber of features: ', np.shape(X_train)[1])
print('model: '+model_name)
print('\ntraining')
score=performance_scores(model, X_train, Y_train, compact=compact_score)
print(score)

print('\ntest')
score=performance_scores(model, X_test, Y_test, compact=compact_score)
print(score)
    
