# basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

# other libraries
import pydotplus
from numpy import loadtxt
from xgboost import XGBClassifier
from IPython.display import Image  
from imblearn.ensemble import BalancedRandomForestClassifier 

# sci-kit learn libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.feature_selection import rfe
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.externals.six import StringIO

def check_correlation(dataframe,lst,threshold):
    return sum((abs(dataframe[lst].corr()>=threshold)).sum()>1)==0

def create_num_list():
    """
    Create a list of numerical features
    """
    return ['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']

def filter_outliers(dataframe,threshold,target_variable):
    """
    Input a data frame and filter outliers based on a threshold while not looking at the target variable
    """
    dataframe = dataframe[(np.abs(stats.zscore(dataframe.drop(target_variable,axis=1))) <= threshold).all(axis=1)]
    return dataframe
    
def separate_x_and_y(dataframe, target_variable):
    """
    Input a data frame and separate into predictors (X) and target (y)
    """
    X = dataframe.drop(target_variable,axis=1)
    y = dataframe[target_variable]
    return X, y

def separate_num_and_cat(dataframe,num_list):
    """
    Input a list of numerical features to separate the larger data frame into categorical and numerical features
    """
    df_num = dataframe[num_list]
    df_cat = dataframe.drop(num_list,axis=1)
    return df_num, df_cat

def scale_data(dataframe):
    """
    Input a data frame to scale the data
    """
    scaler = MinMaxScaler()
    for col in dataframe.columns:
        if (dataframe[col]>=1).sum() >0:
            dataframe[col] = scaler.fit_transform(dataframe[[col]])
    return dataframe

def combine_dataframes(df1,df2):
    """
    Input two dataframes to be merged
    """
    new_df = df1.copy()
    for col in df2.columns:
        new_df[col]=df2[col]

def reduce_features(dataframe,rs,target_variable,min_feat):
    """
    Input a data frame and reduce un-needed features
    """
    X = dataframe.drop(target_variable,axis=1)
    y = dataframe[target_variable]
    rfc = RandomForestClassifier(random_state=rs)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(8), scoring='accuracy',min_features_to_select=min_feat)
    rfecv.fit(X, y)
    support_list = list(rfecv.support_)
    importance = []
    for val, sup in list(zip(X.columns,support_list)):
        if sup == True:
            importance.append(val)
    xydf = pd.concat([X[importance],dataframe[target_variable]],axis=1)
    return xydf
        
def transform_data(dataframe):
    """
    Input data to normalize using Box-Cox transform
    """
    for col in dataframe.columns:
        dataframe[col]=list(stats.boxcox(abs(dataframe[col]+0.5)))[0]
    return dataframe
        
def validation_split(X,y,rs,ts=0.25):
    """
    Input X and y data to split into train and test sections
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=rs,test_size=ts)
    return X_train, X_test, y_train, y_test

class models():
    
    train_results = {'model':[],'accuracy':[],'precision':[],'recall':[],'f1':[]}
    test_results = {'model':[],'accuracy':[],'precision':[],'recall':[],'f1':[]}
    
    def logistic_regression_model(X_tr,y_tr,X_val,y_val,rs):
        """
        Input data and see logistic regression modeling results
        """
        logreg = LogisticRegression(random_state=rs)
        logreg.fit(X_tr, y_tr)
        Y_pred = logreg.predict(X_val)
        acc_log = round(logreg.score(X_val, y_val) * 100, 2)
        p_score_log = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_log = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_log = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_log,p_score_log,r_score_log,f1_log]
        if X_tr.equals(X_val):
            models.train_results['model'].append('logistic regression')
            models.train_results['accuracy'].append(acc_log)
            models.train_results['precision'].append(p_score_log)
            models.train_results['recall'].append(r_score_log)
            models.train_results['f1'].append(f1_log)
        else:
            models.test_results['model'].append('logistic regression')
            models.test_results['accuracy'].append(acc_log)
            models.test_results['precision'].append(p_score_log)
            models.test_results['recall'].append(r_score_log)
            models.test_results['f1'].append(f1_log)
        return scores, cm
            
        
    def support_vector_machine_model(X_tr,y_tr,X_val,y_val,rs):
        """
        Input data and see support vector machine modeling results
        """
        svc = SVC(random_state=rs)
        svc.fit(X_tr, y_tr)
        Y_pred = svc.predict(X_val)
        acc_svc = round(svc.score(X_val, y_val) * 100, 2)
        p_score_svc = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_svc = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_svc = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_svc,p_score_svc,r_score_svc,f1_svc]
        if X_tr.equals(X_val):
            models.train_results['model'].append('support vector machine')
            models.train_results['accuracy'].append(acc_svc)
            models.train_results['precision'].append(p_score_svc)
            models.train_results['recall'].append(r_score_svc)
            models.train_results['f1'].append(f1_svc)
        else:
            models.test_results['model'].append('support vector machine')
            models.test_results['accuracy'].append(acc_svc)
            models.test_results['precision'].append(p_score_svc)
            models.test_results['recall'].append(r_score_svc)
            models.test_results['f1'].append(f1_svc)
        return scores, cm

            
    def knn_model(X_tr,y_tr,X_val,y_val,min_n=3):
        """
        Input data and see support knn modeling results
        """
        knn = KNeighborsClassifier(n_neighbors = min_n)
        knn.fit(X_tr, y_tr)
        Y_pred = knn.predict(X_val)
        acc_knn = round(knn.score(X_val, y_val) * 100, 2)
        p_score_knn = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_knn = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_knn = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_knn,p_score_knn,r_score_knn,f1_knn]
        if X_tr.equals(X_val):
            models.train_results['model'].append('knn')
            models.train_results['accuracy'].append(acc_knn)
            models.train_results['precision'].append(p_score_knn)
            models.train_results['recall'].append(r_score_knn)
            models.train_results['f1'].append(f1_knn)
        else:
            models.test_results['model'].append('knn')
            models.test_results['accuracy'].append(acc_knn)
            models.test_results['precision'].append(p_score_knn)
            models.test_results['recall'].append(r_score_knn)
            models.test_results['f1'].append(f1_knn)
        return scores, cm
            
    def gaussian_naive_bayes_model(X_tr,y_tr,X_val,y_val):
        """
        Input data and see gaussian naive bayes modeling results
        """
        gaussian = GaussianNB()
        gaussian.fit(X_tr, y_tr)
        Y_pred = gaussian.predict(X_val)
        acc_gaussian = round(gaussian.score(X_val, y_val) * 100, 2)
        p_score_gaussian = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_gaussian = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_gaussian = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_gaussian,p_score_gaussian,r_score_gaussian,f1_gaussian]
        if X_tr.equals(X_val):
            models.train_results['model'].append('gaussian naive bayes')
            models.train_results['accuracy'].append(acc_gaussian)
            models.train_results['precision'].append(p_score_gaussian)
            models.train_results['recall'].append(r_score_gaussian)
            models.train_results['f1'].append(f1_gaussian)
        else:
            models.test_results['model'].append('gaussian naive bayes')
            models.test_results['accuracy'].append(acc_gaussian)
            models.test_results['precision'].append(p_score_gaussian)
            models.test_results['recall'].append(r_score_gaussian)
            models.test_results['f1'].append(f1_gaussian)
        return scores, cm
            
    def linear_svc_model(X_tr,y_tr,X_val,y_val,rs):
        """
        Input data and see linear support vector machine modeling results
        """
        lsvc = LinearSVC(random_state=rs)
        lsvc.fit(X_tr, y_tr)
        Y_pred = lsvc.predict(X_val)
        acc_linear_svc = round(lsvc.score(X_val, y_val) * 100, 2)
        p_score_lsvc = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_lsvc = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_lsvc = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_linear_svc,p_score_lsvc,r_score_lsvc,f1_lsvc]
        if X_tr.equals(X_val):
            models.train_results['model'].append('linear svc')
            models.train_results['accuracy'].append(acc_linear_svc)
            models.train_results['precision'].append(p_score_lsvc)
            models.train_results['recall'].append(r_score_lsvc)
            models.train_results['f1'].append(f1_lsvc)
        else:
            models.test_results['model'].append('linear svc')
            models.test_results['accuracy'].append(acc_linear_svc)
            models.test_results['precision'].append(p_score_lsvc)
            models.test_results['recall'].append(r_score_lsvc)
            models.test_results['f1'].append(f1_lsvc)
        return scores, cm
    
    def stochastic_gradient_descent_model(X_tr,y_tr,X_val,y_val,rs):
        """
        Input data and see stochastic gradient descent modeling results
        """
        sgd = SGDClassifier(random_state=rs)
        sgd.fit(X_tr, y_tr)
        Y_pred = sgd.predict(X_val)
        acc_sgd = round(sgd.score(X_val, y_val) * 100, 2)
        p_score_sgd = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_sgd = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_sgd = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm =(confusion_matrix(y_val, Y_pred))
        scores = [acc_sgd,p_score_sgd,r_score_sgd,f1_sgd]
        if X_tr.equals(X_val):
            models.train_results['model'].append('stochastic gradient descent')
            models.train_results['accuracy'].append(acc_sgd)
            models.train_results['precision'].append(p_score_sgd)
            models.train_results['recall'].append(r_score_sgd)
            models.train_results['f1'].append(f1_sgd)
        else:
            models.test_results['model'].append('stochastic gradient descent')
            models.test_results['accuracy'].append(acc_sgd)
            models.test_results['precision'].append(p_score_sgd)
            models.test_results['recall'].append(r_score_sgd)
            models.test_results['f1'].append(f1_sgd)
        return scores, cm
            
    def decision_tree_model(X_tr,y_tr,X_val,y_val,rs):
        """
        Input data and see support decision tree modeling results
        """
        decision_tree = DecisionTreeClassifier(random_state=rs)
        decision_tree.fit(X_tr, y_tr)
        Y_pred = decision_tree.predict(X_val)
        acc_dt = round(decision_tree.score(X_val, y_val) * 100, 2)
        p_score_dt = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_dt = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_dt = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_dt,p_score_dt,r_score_dt,f1_dt]
        if X_tr.equals(X_val):
            models.train_results['model'].append('decision tree')
            models.train_results['accuracy'].append(acc_dt)
            models.train_results['precision'].append(p_score_dt)
            models.train_results['recall'].append(r_score_dt)
            models.train_results['f1'].append(f1_dt)
        else:
            models.test_results['model'].append('decision tree')
            models.test_results['accuracy'].append(acc_dt)
            models.test_results['precision'].append(p_score_dt)
            models.test_results['recall'].append(r_score_dt)
            models.test_results['f1'].append(f1_dt)
        return scores, cm
            
    def random_forest_model(X_tr,y_tr,X_val,y_val,rs,n_e=100):
        """
        Input data and see random forest modeling results
        """
        random_forest = RandomForestClassifier(random_state=rs,n_estimators=n_e)
        random_forest.fit(X_tr, y_tr)
        Y_pred = random_forest.predict(X_val)
        random_forest.score(X_val, y_val)
        acc_rf = round(random_forest.score(X_val, y_val) * 100, 2)
        p_score_rf = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_rf = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_rf = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_rf,p_score_rf,r_score_rf,f1_rf]
        if X_tr.equals(X_val):
            models.train_results['model'].append('random forest')
            models.train_results['accuracy'].append(acc_rf)
            models.train_results['precision'].append(p_score_rf)
            models.train_results['recall'].append(r_score_rf)
            models.train_results['f1'].append(f1_rf)
        else:
            models.test_results['model'].append('random forest')
            models.test_results['accuracy'].append(acc_rf)
            models.test_results['precision'].append(p_score_rf)
            models.test_results['recall'].append(r_score_rf)
            models.test_results['f1'].append(f1_rf)
        return scores, cm
                            
    def XGBoost_model(X_tr,y_tr,X_val,y_val,rs):
        """
        Input data and see XGBoost model results
        """
        xgb = XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=140)
        xgb.fit(X_tr,y_tr)
        Y_pred = xgb.predict(X_val)
        acc_xgb = round(xgb.score(X_val, y_val) * 100, 2)
        p_score_xgb = round(precision_score(y_val,Y_pred,average='micro')*100,2)
        r_score_xgb = round(recall_score(y_val,Y_pred,average='micro')*100,2)
        f1_xgb = round(f1_score(y_val,Y_pred,average='micro')*100,2)
        cm=(confusion_matrix(y_val, Y_pred))
        scores = [acc_xgb,p_score_xgb,r_score_xgb,f1_xgb]
        if X_tr.equals(X_val):
            models.train_results['model'].append('xgboost')
            models.train_results['accuracy'].append(acc_xgb)
            models.train_results['precision'].append(p_score_xgb)
            models.train_results['recall'].append(r_score_xgb)
            models.train_results['f1'].append(f1_xgb)
        else:
            models.test_results['model'].append('xgboost')
            models.test_results['accuracy'].append(acc_xgb)
            models.test_results['precision'].append(p_score_xgb)
            models.test_results['recall'].append(r_score_xgb)
            models.test_results['f1'].append(f1_xgb)
        return scores, cm
                            
def create_summary_dataframe(model_results):
    """
    Input model results and see a summary data frame
    """
    summary_df = pd.DataFrame(model_results)
    return summary_df
                            
def feature_importance_dataframe(X_tr,y_tr,rs):
    """
    Input data to see most indicative features in predicting target class
    """
    rfc = RandomForestClassifier(n_estimators=400,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',
                                           random_state=rs)
    rfc.fit(X_tr,y_tr)
    importances = rfc.feature_importances_
    importance_df = pd.DataFrame(importances).T
    importance_df.columns = X_tr.columns
    importance_df.T
    importance_df_sorted=pd.DataFrame(importance_df.T[0].sort_values())
    return importance_df_sorted
                   
def draw_decision_tree(rs,X_tr,y_tr,depth):
    """
    Input a set of data and see the decision tree for selecting values
    """
    dot_data = StringIO()
    dt = DecisionTreeClassifier(random_state=rs)
    dt.fit(X_tr,y_tr)
    export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,max_depth=depth,feature_names=X_tr.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())