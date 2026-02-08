import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from mrmr import mrmr_classif

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



import shap
import umap

import joblib
import os
import shutil

import seaborn as sns
import colorcet as cc

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from sklearn.cluster import AgglomerativeClustering

from scipy.special import softmax
import matplotlib.ticker as mtick

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.utils import to_categorical


def read_data(path_dataset, path_codebook):
    df = pd.read_stata(path_dataset, index_col=None)
    sheet1 = "codebook"
    df_codebook = pd.read_excel(path_codebook, header=1, sheet_name = sheet1,engine='openpyxl')

    return df, df_codebook

def data_preprocessing(df_data,df_codebook,non_relevant_vars,outcomes):

    # Preselect features
    list_of_values = ["YES"]
    df_YES = df_codebook[df_codebook['Ximena Selected'].isin(list_of_values)]
    list_variables = list(df_YES['Variable name'].values)

    df_data = df_data[list_variables]
    
    # Remove samples
    df_data = df_data[(df_data['survey']==1.0)]
    df_data = df_data.reset_index(drop = True)

    # Remove features
    df_data.drop(non_relevant_vars,axis=1,inplace=True)

    # Rename features
    name_dic = dict(zip(df_codebook["Variable name"], df_codebook["New variable name"]))
    for i in list(df_data.columns):
        if i not in outcomes:
            df_data.rename({i: name_dic[i]}, axis=1, inplace=True)
        
    return df_data

def set_target(df_data,outcomes,thr_disc_1,thr_disc_2):
    X = df_data.drop(columns=outcomes, axis=1)
    y_aux = df_data[outcomes]
    for col in y_aux.columns:
        y_aux[col] = pd.cut(y_aux[col], bins=[-np.inf,thr_disc_1[col],thr_disc_2[col], np.inf], labels=[0,1,2]).to_numpy()

    return X,y_aux

def data_encoding(X, additional_categ_var):

    for i in additional_categ_var:
        X[i].astype('category')

    X_categ = X.select_dtypes('category')
    categ_vars = list(X_categ.columns)
    num_vars = [x for x in list(X.columns) if x not in categ_vars]

    categories = []
    ord_categ_var = ['COV_NEWS','RES_OUTDOOR_USE','GLOVES','MASK','DISINFECTION','REGU_IMP','CONFIN_BUY','PHYSACT_CHANGES','WEIGHT_CHANGE','HEALTH_STATUS','SLEEP_CHANGES']

    for i in range(len(categ_vars)):
        categ = categ_vars[i]
        if categ not in ord_categ_var:
            labels = X_categ[categ].cat.categories.tolist()
        elif categ == ord_categ_var[0]:
            labels = ['Less than once a week', '1 time a week', 'Two or three times a week', '1 time a day', 'Several times a day']
        elif categ == ord_categ_var[1]:
            labels = ['Available space. Never', 'Not available outdoor space.', 'Available space. Rarely' ,'Available space. Sometimes', 'Available space. Often.']
        elif categ == ord_categ_var[2]:
            labels = ['Never','Yes, but only a few times', 'Yes, almost every time I go out', 'Yes, systematically every time']
        elif categ == ord_categ_var[3]:
            labels = ['Never', 'Yes, but only a few times', 'Yes, almost every time I go out', 'Yes, systematically every time']
        elif categ == ord_categ_var[4]:
            labels = ['Never', 'Yes, but only a few times', 'Yes, almost every time I go out', 'Yes, systematically every time']
        elif categ == ord_categ_var[5]:
            labels = ['No, they are not important', 'I think sometimes they can be skipped', 'Yes, I think it is very important']
        elif categ == ord_categ_var[6]:
            labels = ['No, I expect or replace it with something else', 'No, I always find what I need', 'Yes, sometimes', 'Yes, always']
        elif categ == ord_categ_var[7]:
            labels = ['I do much less physical activity', 'I do less physical activity', 'I do a little less physical activity', 'No changes', 'I do a little more physical activity', 'I do more physical activity', 'I do a lot more physical activity']
        elif categ ==ord_categ_var[8]:
            labels = ['It has decreased', 'It is the same', 'It has increased']
        elif categ == ord_categ_var[9]:
            labels = ['Bad', 'Regular', 'Good', 'Very good', 'Excellent']
        elif categ == ord_categ_var[10]:
            labels = ['Yes, it has decreased', "No, it's the usual", 'Yes, it has increased']
    
        categories.append(labels)

    categ_encoder = OrdinalEncoder(categories=categories, handle_unknown = 'use_encoded_value', unknown_value = np.nan)
    aux = categ_encoder.fit_transform(X_categ)
    X[categ_vars] = aux

    return X, num_vars, categ_vars

def data_imputation(X_train, X_test, categ_vars):
    
    # Columns to KNN impute: ALL
    to_knn_impute = list(X_train.columns)

    # Define and fit imputer with X_train
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean').fit(X_train[to_knn_impute])

    # Apply imputer (already fitted with training data) to X_train data
    imputed_data = imputer.transform(X_train[to_knn_impute])
    X_train[to_knn_impute] = imputed_data

    # Apply imputer (already fitted with training data) to X_test data
    imputed_data_test = imputer.transform(X_test[to_knn_impute])
    X_test[to_knn_impute] = imputed_data_test

    # Round categorical variables
    for i in categ_vars:
      X_train[i] = X_train[i].round(decimals = 0)
      X_test[i] = X_test[i].round(decimals=0)

    return X_train, X_test

def data_one_hot_encoding(X_train, X_test, features):
    df = pd.concat([X_train,X_test],keys=['train','test'])

    for col in features:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df.drop([col], axis=1, inplace=True)
    
    #if col == "FOLLOW_DEP_TREAT":
    #    df.drop([col+'_0.0'],axis=1, inplace=True)

    X_train = df.loc['train']
    X_test = df.loc['test']

    return X_train, X_test

def data_scale(X_train, X_test):

    scaler = StandardScaler()
    features = X_train

    scaler = StandardScaler().fit(features.values)
    features_scaled = scaler.transform(features.values)

    # Create new normalized dataframe
    X_train_scaled = pd.DataFrame(index= X_train.index, columns=X_train.columns.values)
    X_train_scaled[X_train.columns.values] = features_scaled 

    #Apply scaler to test

    # fit with Scaler
    features_test = X_test
    features_test_scaled = scaler.transform(features_test.values)

    # Create new normalized dataframe
    X_test_scaled = pd.DataFrame(index = X_test.index, columns=X_test.columns.values)
    X_test_scaled[X_test.columns.values] = features_test_scaled 

    return X_train_scaled, X_test_scaled

    ############ PARAMETERS ############

def data_feature_selection(X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, algorithm, n_features):
    
    corr_matrix = X_train.corr().abs()

    # select upper traingle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

    X_train_red = X_train.drop(to_drop,axis=1)
    X_test_red = X_test.drop(to_drop,axis=1)
    X_train_scaled_red = X_train_scaled.drop(to_drop,axis=1)
    X_test_scaled_red = X_test_scaled.drop(to_drop,axis=1)

    if algorithm == "mrmr":
        selected_features = mrmr_classif(X=X_train_red, y=y_train, K= n_features,show_progress = False)
        to_drop = list(set(list(X_train_red.columns)) - set(selected_features))
    
    else:
        print("Not implemented")

    X_train_red = X_train_red.drop(to_drop,axis=1)
    X_test_red = X_test_red.drop(to_drop,axis=1)
    X_train_scaled_red = X_train_scaled_red.drop(to_drop,axis=1)
    X_test_scaled_red = X_test_scaled_red.drop(to_drop,axis=1)

    #print("Selected features: ", X_train_red.columns)

    return X_train_red, X_train_scaled_red, X_test_red, X_test_scaled_red

def CustomizedScorer(y_test, y_pred):
    

    #cost_aux = roc_auc_score(y_test, y_pred, multi_class='ovr', average=None)
    
    cm = confusion_matrix(y_test, y_pred)
    aux2 = np.sum(cm[2,:])
    aux1 = np.sum(cm[1,:])
    aux0 = np.sum(cm[0,:])

    #cost = (aux2*(cm[2,0]+cm[2,1]) + aux1*cm[1,0])/(aux2+aux1)
    #cost = cm[2,0]/aux2 + cm[2,1]/aux2
    #cost = cm[2,0]/aux2
    #cost = 1/aux2*cm[2,2] + 1/aux1*cm[1,1]
    #cost = ((1/aux2)*cm[2,2] + (1/aux1)*cm[1,1])/2
    #cost = (1/aux2)*cm[2,2] + (1/aux1)*cm[1,1]
    #cost = 2/3*cm[2,0]/aux2 + 1/3*cm[2,1]/aux2
    #cost = cm[2,0]/aux2
    #cost = 1/3(*(cm[2,0]+cm[2,1])/aux2 + (cm[1,0]+cm[1,2])/aux1)
    cost = ((cm[2,0]+cm[2,1])/aux2 + (cm[0,2]+cm[0,1])/aux0 + (cm[1,0]+cm[1,2])/aux1)/3
    #cost = (cost_aux[0] + cost_aux[2])/2
    
    return cost

def grid_search(X_train,X_train_scaled, y_train, model,model_name,target,n_iter,cv_t2, n_features):

    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    custom_score = make_scorer(CustomizedScorer, greater_is_better = False)

    scorer = "roc_auc_ovo"
    #scorer = "balanced_accuracy"

    if model_name == "XGBoost":

        parameters = {
            'max_depth': range(3,15,3),
            'n_estimators': range(15,55,10),
            'learning_rate': np.linspace(0.01,0.75,5),
            'reg_lambda': np.linspace(0,10,5),
            #'min_split_loss': np.linspace(0,25,5)
        }
        
        randomized_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            scoring = scorer,
            cv = cv_t2,
            verbose = False,
            n_iter = n_iter
        )

        clf = randomized_search.fit(X_train_scaled, np.ravel(y_train),sample_weight=classes_weights)

    elif model_name == "Support Vector Machines":

        parameters = {
            'C': np.exp(np.linspace(np.log(0.001),np.log(2),25))
        }

        # RandomizeSearchCV
        randomized_search = RandomizedSearchCV(
            estimator = model,
            param_distributions=parameters,
            scoring=scorer,
            cv = cv_t2,
            verbose = False,
            n_iter = n_iter
        )

        clf = randomized_search.fit(X_train_scaled, np.ravel(y_train),sample_weight=classes_weights)
    
    elif model_name == "Multi-Layer Perceptron":
        
        weights = class_weight.compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
        
        class_weights = {k: v for k, v in zip([0,1,2], weights)}
        
        # learning algorithm parameters
        learning_rate= np.exp(np.linspace(np.log(1e-5),np.log(0.1),10))

        # activation
        activation=['sigmoid']

        # numbers of layers
        n_layers = [1,2,3]

        # dropout and regularisation
        dropout = [0, 0.1, 0.2]
        l1 = [0.01]
        l2 = [0.01]

        n_features = n_features

        # dictionary summary
        param_grid = dict(
            n_layers=n_layers, 
            act=activation, 
            l1=l1, 
            l2=l2, 
            learning_rate=learning_rate, 
            dropout=dropout,
            n_features = n_features,
        )

        # RandomizeSearchCV
        randomized_search = RandomizedSearchCV(
            estimator = model,
            param_distributions=param_grid,
            scoring = scorer,
            cv = cv_t2,
            verbose = False,
            n_iter = n_iter
        )

        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)
        dummy_y = to_categorical(encoded_Y)

        clf = randomized_search.fit(X_train_scaled, dummy_y, class_weight = class_weights)
        #clf = randomized_search.fit(X_train_scaled, dummy_y)


    elif model_name == "Naive Bayes":

        parameters = {
            'var_smoothing': np.exp(np.linspace(np.log(1e-9),np.log(1000),50))
        }

        # RandomizeSearchCV
        randomized_search = RandomizedSearchCV(
            estimator = model,
            param_distributions=parameters,
            scoring = scorer,
            cv = cv_t2,
            verbose = False,
            n_iter = n_iter
        )

        clf = randomized_search.fit(X_train_scaled, np.ravel(y_train),sample_weight=classes_weights)

    elif model_name == "Random Forest":
        parameters = {
            'n_estimators': range(5,100,10),
            'max_depth': range(5,50,5),
            'min_samples_split': np.linspace(0,1,5)
        }

        # RandomizeSearchCV
        randomized_search = RandomizedSearchCV(
            estimator = model,
            param_distributions=parameters,
            scoring = scorer,
            cv = cv_t2,
            verbose = False,
            n_iter = n_iter
        )

        clf = randomized_search.fit(X_train_scaled, np.ravel(y_train),sample_weight=classes_weights)

    elif model_name == "Logistic Regression":
        
        parameters = {
            'penalty': ["l2", None],
            'C': np.exp(np.linspace(np.log(0.001),np.log(2),25)),
            'solver': ["lbfgs"]
        }

        # RandomizeSearchCV
        randomized_search = RandomizedSearchCV(
            estimator = model,
            param_distributions=parameters,
            scoring = scorer,
            cv = cv_t2,
            verbose = False,
            n_iter = n_iter
        )

        clf = randomized_search.fit(X_train_scaled, np.ravel(y_train),sample_weight=classes_weights)
    
    best_params = clf.best_params_    
    best_model = clf.best_estimator_
    best_score = clf.cv_results_

    return best_model, best_params, best_score

def compute_confussion_matrix(target,n_batch,n_cv,CONFIG):

    colors = {"G_depressionscore": "#0062BE", "G_anxietyscore": "#D18F00", "G_totalscore": "#5DAF31"}
    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}

    l_cm = []

    for batch in range(n_batch):
        for cv in range(n_cv):
            
            # load data
            path_data = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/data_scaled_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"

            data = pd.read_csv(path_data, index_col=[0,1])
            data_test = data.loc['test']
            y_test = data_test[target]
            X_test = data_test.drop([target],axis=1)

            # load model
            model = joblib.load(CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/model_"+str(batch+1)+ "_cv_"+str(cv+1)+".sav")
                
            # aggregated confusion matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            l_cm.append(cm)

    cm_mean = np.zeros((3,3))
    cm_std = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            aux = []
            aux2 = []
            for cm in l_cm:
                aux.append(cm[i,j]/np.sum(cm[i,:]))
                
            cm_mean[i,j] = np.mean(aux)
            cm_std[i,j] = np.std(aux)        
    
    l_plot = cm_mean.tolist()

    for i in range(3):
        for j in range(3):
            l_plot[i][j] = str(np.round(cm_mean[i,j],2)) + "%" +"\n"+ "("+ r'$\pm$ ' + str(np.round(cm_std[i,j],2)) + ")"
    
    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_mean, annot=l_plot, fmt='', cmap=sns.light_palette(colors[target], as_cmap=True),vmin=0, vmax=1,ax = ax)

    ax.set_xlabel('\nPredicted label')
    ax.set_ylabel('True label');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['None','Mild', 'Severe'])
    ax.yaxis.set_ticklabels(['None','Mild', 'Severe'])

    # save figure
    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/confusion_matrix_"+sufix[target]+".svg", format='svg', dpi=1200,bbox_inches="tight")

def compute_roc_auc_performance(targets,n_batch,n_cv,CONFIG):

    l1 =[]
    l2 =[]
    l3 =[]

    for target in targets:
        for batch in range(n_batch):
            for cv in range(n_cv):
            
                # load data
                path_data = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/data_scaled_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"
                data = pd.read_csv(path_data, index_col=[0,1])
                data_test = data.loc['test']
                y_test = data_test[target]
                X_test = data_test.drop([target],axis=1)

                # load model
                model = joblib.load(CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/model_"+str(batch+1)+ "_cv_"+str(cv+1)+".sav")

                # aggregated confusion matrix
                y_prob = model.predict_proba(X_test)
                value = roc_auc_score(y_test, y_prob,multi_class="ovr", average=None)

                l1.append(target)
                l2.append(value[0])
                l3.append('roc0')

                l1.append(target)
                l2.append(value[1])
                l3.append('roc1')

                l1.append(target)
                l2.append(value[2])
                l3.append('roc2')

    fig, ax = plt.subplots(figsize=(8, 4))
    df_plot = pd.DataFrame([l1,l2,l3],index = ['target','score','class']).transpose()

    my_pal = {"G_depressionscore": "#009BFF", "G_anxietyscore": "#D18F00", "G_totalscore":"#5DAF31"}

    sns.set_style("whitegrid")
    sns.boxplot(x="class", y="score",hue = "target", data=df_plot,whis=10,ax = ax, palette=my_pal,linewidth=0.75) 
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    legend_handles, _= ax.get_legend_handles_labels()
    ax.legend(legend_handles, ['Depression','Anxiety','Self-perceived stress'],fontsize = '10', bbox_to_anchor=(1.02,1),loc='upper left',borderaxespad=0)
    ax.xaxis.set_ticklabels(['None','Mild', 'Severe'])
    ax.set_xlabel('Severity Class')
    ax.set_ylabel('AUROC_ovr')

    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/roc_auc.svg", format='svg', dpi=1200,bbox_inches="tight")

def compute_features(target, n_batch, n_cv,CONFIG):
    d_freq = {}

    for batch in range(n_batch):
        for cv in range(n_cv):

            path_data = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name +"/data_scaled_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"
            data = pd.read_csv(path_data, index_col=[0,1])

            data_test = data.loc['test']
            X_test = data_test.drop([target],axis=1)

            for var in X_test.columns:
                if var in d_freq:
                    d_freq[var] = d_freq[var] + 1
                else:
                    d_freq[var] = 1
    
    aux1 = np.array(list(d_freq.values()))
    aux2 = np.array(list(d_freq.keys()))
    sol = np.where(aux1 == n_cv*n_batch)[0]

    selected_features = list(aux2[sol])

    return selected_features, d_freq

def compute_shap_values(target,n_batch,n_cv,CONFIG):
    
    selected_features, d_freq = compute_features(target, n_batch, n_cv,CONFIG)
    features_freq = {}

    shap_sets = []
    X_test_sets = []
    y_pred_sets = []
    y_prob_sets = []

    d_shap = {}

    for batch in range(n_batch):
        for cv in range(n_cv):

            # load data and model
            path_data = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name +"/data_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"
            path_data_scaled = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name +"/data_scaled_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"

            data = pd.read_csv(path_data, index_col=[0,1])
            data_scaled = pd.read_csv(path_data_scaled, index_col=[0,1])

            model = joblib.load(CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name +"/model_"+str(batch+1)+ "_cv_"+str(cv+1)+".sav")
            
            data_test = data.loc['test']
            data_test_scaled = data_scaled.loc['test']

            y_test = data_test[target]
            X_test = data_test.drop([target],axis=1)
            X_test_scaled = data_test_scaled.drop([target],axis=1)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)

            for var in X_test.columns:
                if var in features_freq:
                    features_freq[var] = features_freq[var] + 1
                else:
                    features_freq[var] = 1

            # compute shap values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled, tree_limit = -1)

            values0 = shap_values[0]
            shap_mean = np.abs(values0).mean(0)
            df_shap = pd.DataFrame(shap_mean, index = X_test_scaled.columns, columns = [target+"_0"])

            values1 = shap_values[1]
            shap_mean = np.abs(values1).mean(0)
            df_shap[target+"_1"] = shap_mean

            values2 = shap_values[2]
            shap_mean = np.abs(values2).mean(0)
            df_shap[target+"_2"] = shap_mean

            # Select shap values for class=2 and common features
            pos = np.where(y_test.values == 2)[0]
            cols = [X_test_scaled.columns.get_loc(i) for i in selected_features]
            values = values2[pos,:]
            shap_sets.append(values[:,cols])

            for var in list(df_shap.index):
                if var in d_shap:
                    d_shap[var] = d_shap[var] + df_shap.loc[var,target+"_2"]
                else:
                    d_shap[var] = df_shap.loc[var,target+"_2"]


            # Select y_pred, y_prob and X_test  
            aux_x = X_test[selected_features].values  ## X_test is not scaled !!!!!!!!!!!!
            y_pred_sets.append(y_pred[pos])
            y_prob_sets.append(y_prob[pos,2])
            X_test_sets.append(aux_x[pos,:])
            
    for key in d_shap.keys():
        d_shap[key] = d_shap[key]/features_freq[key]

    shap_cv = np.concatenate(shap_sets)
    X_test_cv = np.concatenate(X_test_sets)
    y_pred_cv = np.concatenate(y_pred_sets)
    y_prob_cv = np.concatenate(y_prob_sets)

    df_mapper = pd.DataFrame(shap_cv,columns = ["shap_" + str(i) for i in selected_features])
    df_aux = pd.DataFrame(X_test_cv,columns = selected_features)
    df_mapper = pd.concat([df_mapper,df_aux],axis=1)
    df_mapper['y_pred'] = y_pred_cv
    df_mapper['y_prob'] = y_prob_cv

    return df_mapper, selected_features, d_freq, d_shap

def compute_UMAP(df_shap,target,selected_features,n_neighbors,spread,min_dist,ran):
    
    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}

    # UMAP
    mapper = umap.UMAP(random_state=ran,n_neighbors = n_neighbors,spread = spread, min_dist = min_dist, n_components=2).fit_transform(df_shap[["shap_" + str(i) for i in selected_features]])
    df_mapper = pd.DataFrame(mapper, columns = ['x','y'])

    df_mapper = pd.concat([df_mapper,df_shap],axis=1)

    colors = ["#00BDFF", "#FF0000"] # first color is black, last is red
    cm = LinearSegmentedColormap.from_list("Custom", colors)
    
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.scatterplot(data=df_mapper, x="x", y="y",hue="y_prob",alpha=0.6,s=25,edgecolors='None',palette=cm)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()

    axx = ax.figure.colorbar(sm, ax=ax)

    axx.set_label(label = "Severe Probability",size = 12)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/umap_"+ sufix[target]+"_probs.svg", format='svg', dpi=1200,bbox_inches="tight")


    return df_mapper

def compute_clustering(df_mapper,target,n_clu):

    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}

    # Clustering
    kmeans = AgglomerativeClustering(linkage="single",n_clusters=n_clu)
   
    #predict the labels of clusters.
    label = kmeans.fit_predict(df_mapper[['x','y']])

    palette = sns.color_palette(cc.glasbey_light, n_colors=n_clu)

    df_mapper['cluster'] = label

    fig, ax = plt.subplots(figsize=(4.75, 4))
    sns.set_style("white")
    ax = sns.scatterplot(data=df_mapper, x="x", y="y",hue="cluster",alpha=1,s=25,edgecolors='None',palette=palette)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    legend_handles, _= ax.get_legend_handles_labels()
    ax.legend(legend_handles, [str(i+1) for i in range(n_clu)],fontsize = '12', bbox_to_anchor=(1.02,1),loc='upper left',borderaxespad=0,title = r'$\bf{Cluster}$',title_fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/clusters_umap_"+ sufix[target]+".svg", format='svg', dpi=1200,bbox_inches="tight")
    
    return df_mapper

def compute_risk_profiles(df_mapper,target,selected_features,raw = True):

    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}

    l = []
    n_clu = len(df_mapper['cluster'].unique())

    n2 = ["shap_" + str(i) for i in selected_features]
    percen = []
    # compute risk factors
    for c in range(n_clu):

        identified = 0
        df_aux = df_mapper[df_mapper['cluster']==c]
        percen.append(np.round(len(df_aux)/len(df_mapper)*100,1))

        for i in range(len(n2)):
            df_pos = df_aux[df_aux[n2[i]]>=0.01]
            pos_ratio = len(df_pos)/len(df_aux)
            inf_pos = np.min(df_pos[selected_features[i]])
            sup_pos = np.max(df_pos[selected_features[i]])

            if pos_ratio >= 0.9 and len(df_pos[n2[i]].values) > 0.0:
                l_aux = [c,selected_features[i],pos_ratio,np.mean(df_pos[n2[i]].values),inf_pos,sup_pos]
                l.append(l_aux)

                identified = 1
                
        if identified == 0:
            l_aux = [c,'None',0,0,0,0]
            l.append(l_aux)

    df_aux = pd.DataFrame(l,columns = ['cluster','risk_factor','prevalence','mean_risk','min_thr','max_thr'])
    df_plot = pd.pivot_table(df_aux,index = 'cluster',columns = 'risk_factor', values = 'mean_risk',fill_value = 0)
    df_plot.drop(columns = ['None'],inplace = True)

    df_plot = df_plot.reindex(df_plot.mean().sort_values(ascending = False).index, axis=1)

    # stacked circular barplot
    labels = ['Cluster ' + str(i+1) +"\n"+ "("+str(percen[i])+"%)"  for i in range(n_clu)] 
    #['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5','Cluster 6','Cluster 7', 'Cluster 8']

    data = df_plot.values.transpose()

    # Calculate angles for each sector
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    # Create figure and axis objects
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.grid(color='gray', linestyle='dashed',alpha = 0.5)

    theta_vals = np.linspace(0, 2 * np.pi, 100)
    r_vals = np.ones_like(theta_vals)*0
    ax.plot(theta_vals, r_vals, '--', color='k',linewidth=1,alpha=0.5)

    if raw == True:
        label_n = list(df_plot.columns)
        df_labels = df_aux.groupby('risk_factor').max()
        label_n = [risk_factor + " [" + str(np.round(df_labels.loc[risk_factor]['min_thr'],0)) + ", "+ str(np.round(df_labels.loc[risk_factor]['max_thr'],0)) + "]" for risk_factor in list(df_plot.columns)]
    else:
        if target == "G_depressionscore":
            label_n = ["HEALTH_STATUS = bad or regular","DUKE_IDX >= 18","PHYSACT_CHANGES = much less or less","CHR_DEP = yes", "N_SYMP_CUR >= 2", "CONFIN_BUY = yes","CONFIN_CONFLICTS = yes"]
        elif target == "G_anxietyscore":
            label_n = ["HEALTH_STATUS = bad or regular","DUKE_IDX >= 18","CHR_DEP = yes","CONFIN_BUY = yes","N_SYMP_CUR >= 1","CONFIN_RULES >= 4", "SLEEP_CHANGES = less hours"]
        elif target == "G_totalscore":
            label_n = ["DUKE_IDX >= 16","SLEEP_CHANGES = less hours","CONFIN_BUY = yes","CHR_DEP = yes","HEALTH_STATUS = bad or regular","GEN = female"]



    # Add bars
    bottom = np.zeros(len(labels))
    line1 = []

    if raw == True:
        palette = sns.color_palette(cc.glasbey_light, n_colors=25)
        colors = list(map(mpl.colors.rgb2hex, palette))        
        for i, row in enumerate(data):
            line = ax.bar(angles, row, width=0.5, bottom=bottom, alpha=0.7, label = label_n[i],color = colors[i], linewidth = 0.5)
            bottom += row
            line1.append(line)
    else:
        #labels[1] = 'Cluster ' + str(1+1) + ": non-characterized" +"\n"+ "("+str(percen[1])+"%)"  
        colors = {'DUKE_IDX': '#FF8000','HEALTH_STATUS':'#61FF00','CHR_DEP': '#EC00FF','PHYSACT_CHANGES': '#89663C', 'CONFIN_BUY': '#00FFF3', 'N_SYMP_CUR':'#056161','HOURS_SLEEP':'#000000','SLEEP_CHANGES':'#C9CF00','FOLLOW_DEP_TREAT_0.0':'#CB6FFE','GEN':'#5100FF', 'CONFIN_CONFLICTS': '#B662BF','CONFIN_RULES': '#0400FF'}
        for i, row in enumerate(data):
            line = ax.bar(angles, row, width=0.5, bottom=bottom, alpha=0.7, label = label_n[i],color = colors[list(df_plot.columns)[i]], linewidth = 0.5)
            bottom += row
            line1.append(line)


    ax.set_xticks(angles)
    ax.set_xticklabels(labels,fontweight ='bold',fontsize =12)
    ax.legend(handles=line1, bbox_to_anchor=(1.3,1),loc='upper left',borderaxespad=0,title = r'$\bf{Risk}$ $\bf{factors}$',title_fontsize=14,fontsize = 14,shadow=True, borderpad=1, alignment='left')
    ax.tick_params(axis='both', which='major', pad=18) 

    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/profiles_"+ sufix[target]+".svg", format='svg', dpi=1200,bbox_inches="tight")

    return df_aux,df_plot, label_n

def compute_feature_selection_frequency(d_freq, target):
    
    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}
    colors = {"G_depressionscore": "#0062BE", "G_anxietyscore": "#D18F00", "G_totalscore": "#5DAF31"}

    fig, ax = plt.subplots(figsize=(16, 6))

    barWidth = 0.3
    br1 = range(len(d_freq.values()))
    br2 = [x + barWidth for x in br1]

    ax.bar(br1, np.array(list(d_freq.values()))/5, color = colors[target], width = barWidth, label = "Selection frequency")
    plt.grid(axis = 'y')

    ax.set_xticks([r + barWidth/2 for r in br1])
    ax.set_xticklabels(list(d_freq.keys()), rotation=40,ha='right',va='top', fontsize = 12)
    ax.set_xlabel('Features', fontweight ='bold', fontsize = 16)
    ax.set_ylim([0, 1.1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))


    ax.set_ylabel('Feature selection', fontweight ='bold', fontsize = 16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/selection_freq_"+ sufix[target]+".svg", format='svg', dpi=1200,bbox_inches="tight")

def compute_pred_per_cluster(df_mapper,target,n_clu):
    
    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}
    palettes = {"G_depressionscore": ['#6AB7FF','#0083FF','#004280'], "G_anxietyscore": ["#FFE498","#FFBD00","#D18F00"], "G_totalscore": ["#DBFF96","#9EDE80","#5DAF31"]}

    df_aux = df_mapper.groupby('cluster')['y_pred'].value_counts(normalize = True)
    df_index = df_aux.index

    l1 = []
    l2 = []
    l3 = []

    for i in range(len(df_aux)):
        l1.append(df_aux.iloc[i])
        l2.append(str(df_index[i][0]))
        l3.append(df_index[i][1])

    df_plot_freqs = pd.DataFrame([l1,l2,l3],index = ["freq","cluster","label"]).transpose()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data = df_plot_freqs,x = "cluster",y = "freq", hue = "label",ax=ax,width = 0.7, palette= palettes[target])
    legend_handles, _= ax.get_legend_handles_labels()
    ax.legend(legend_handles, ["No", "Mild", "Severe"],fontsize = '10',title = r'$\bf{Prediction}$',title_fontsize = 12, bbox_to_anchor=(1.02,1),loc='upper left',borderaxespad=0)

    ax.set_xlabel('Cluster', fontsize = 12)
    ax.xaxis.set_ticklabels([str(i+1) for i in range(n_clu)])

    ax.set_ylabel('', fontsize = 12);

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(labelsize = 10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/predictions_per_cluster_"+sufix[target]+".svg", format='svg', dpi=1200,bbox_inches="tight")
  
def compute_shap_local_exp(df_mapper,target,cluster,prediction,n_batch,n_cv,thr,CONFIG,pos_id_n):
    
    y_pred_sets = []

    for batch in range(n_batch):
        for cv in range(n_cv):

            # load data and model
            path_data = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/data_scaled_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"
            data = pd.read_csv(path_data, index_col=[0,1])
            model = joblib.load(CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/model_"+str(batch+1)+ "_cv_"+str(cv+1)+".sav")
            
            data_test = data.loc['test']
            y_test = data_test[target]
            X_test = data_test.drop([target],axis=1)

            # Predictions
            y_pred = model.predict(X_test)
            # Select shap values for class=2 and common features
            pos = np.where(y_test.values == 2)[0]
            y_pred_sets.append(pd.Series(y_pred[pos]))
            
    #y_pred_cv = pd.DataFrame(y_pred_sets[0])
    y_pred_cv = pd.concat(y_pred_sets,keys=[i for i in range(n_cv)])

    pos = np.where((df_mapper['cluster'] == cluster) & (df_mapper['y_pred'] == prediction) )[0]
    pos_id = np.random.choice(pos)

    if pos_id_n != -1:
        pos_id = pos_id_n
    #pos_id = pos[0]

    cv = y_pred_cv.index[pos_id][0]
    row = y_pred_cv.index[pos_id][1]

    batch = 0

    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}

    path_data = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/data_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"
    path_data_scaled = CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/data_scaled_batch_"+str(batch+1)+"_cv_"+str(cv+1)+".csv"
    
    data = pd.read_csv(path_data, index_col=[0,1])
    data_scaled = pd.read_csv(path_data_scaled, index_col=[0,1])

    model = joblib.load(CONFIG.path_results+"results/" + target+"/"+CONFIG.clf_name+"/model_"+str(batch+1)+ "_cv_"+str(cv+1)+".sav")

    data_test = data.loc['test']
    y_test = data_test[target]
    X_test = data_test.drop([target],axis=1)

    data_test_scaled = data_scaled.loc['test']
    y_test = data_test_scaled[target]
    X_test_scaled = data_test_scaled.drop([target],axis=1)

    y_prob = model.predict_proba(X_test_scaled)

    explainer = shap.TreeExplainer(model)
    shap_2 = explainer.shap_values(X_test_scaled,tree_limit = -1)[2]

    expected_value = explainer.expected_value

    pos_aux = np.where(y_test.values == prediction)[0]
    pos_id_p = pos_id
    pos_id = pos_aux[row]

    y1 = y_prob[pos_id,2]
    aux = softmax(np.array([expected_value[0],expected_value[1],expected_value[2]]))
    y0 = aux[2]

    raw_predt = model.predict(X_test_scaled.iloc[pos_id:pos_id+1, :], output_margin=True)
    raw_pred = raw_predt[0][2]

    shap_test_coef = (raw_pred-expected_value[2])/(y1-y0)
    shap_test = shap_2[pos_id]/shap_test_coef

    X_test_plot = X_test.copy()
    #X_test_plot.at[X_test_plot.index.to_list()[pos_id],'CHR_DEP'] = "yes"
    #X_test_plot.at[X_test_plot.index.to_list()[pos_id],'HEALTH_STATUS'] = "Regular"
    #X_test_plot.at[X_test_plot.index.to_list()[pos_id],'HEALTH_STATUS'] = "bad"
    #X_test_plot.at[X_test_plot.index.to_list()[pos_id],'CONFIN_BUY'] = "yes"
    X_test_plot.at[X_test_plot.index.to_list()[pos_id],'GEN'] = "female"
    X_test_plot.at[X_test_plot.index.to_list()[pos_id],'SLEEP_CHANGES'] = "less hours"
    #X_test_plot.at[X_test_plot.index.to_list()[pos_id],'PHYSACT_CHANGES'] = "A little more"
    X_test_plot.at[X_test_plot.index.to_list()[pos_id],'HEALTH_STATUS'] = "Very good"
    #X_test_plot.at[X_test_plot.index.to_list()[pos_id],'CONFIN_CONFLICTS'] = "Yes"





    sns.set_style("white")
    #print(X_test_plot.iloc[pos_id:pos_id+1, :])
    local_plot = shap.force_plot(y0, shap_test,X_test_plot.iloc[pos_id:pos_id+1, :], figsize=(20, 3), matplotlib=True,show=False,contribution_threshold = thr)

    local_plot.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/local_explanation_"+sufix[target]+".svg", format='svg', dpi=1200, bbox_inches="tight")

    plt.show()

    return pos_id_p

def compute_risk_factors(df_mapper, target, df_profiles, df_plot,label_n,raw):

    sufix = {"G_depressionscore": "dep", "G_anxietyscore": "anx", "G_totalscore": "tot"}

    df_labels = df_profiles.groupby('risk_factor').max()
    l = []
    for var in df_plot.columns:
        clusters = list(np.where(df_plot[var].values  > 0)[0])
        l_aux = []
        for c in clusters:
            df_aux_1 = df_mapper[df_mapper['cluster'] == c]
            df_aux_2 = df_aux_1[(df_aux_1[var]<= df_labels.loc[var,"max_thr"]) & (df_aux_1[var]>= df_labels.loc[var,"min_thr"])]
            l_aux.extend(df_aux_2["shap_"+var].values)
        l.append(l_aux)
    l.reverse()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    y0 = np.arange(len(l))+1
    y0 = y0[::-1]
    parts = ax.violinplot(l, points=40, widths=0.5,showmeans=True, showextrema=True, bw_method='silverman', vert=False)

    if raw == True:
        palette = sns.color_palette(cc.glasbey_light, n_colors=25)
        colors = list(map(mpl.colors.rgb2hex, palette))        
    else:
        colors = {'DUKE_IDX': '#FF8000','HEALTH_STATUS':'#61FF00','CHR_DEP': '#EC00FF','PHYSACT_CHANGES': '#89663C', 'CONFIN_BUY': '#00FFF3', 'N_SYMP_CUR':'#056161','HOURS_SLEEP':'#000000','SLEEP_CHANGES':'#C9CF00','FOLLOW_DEP_TREAT_0.0':'#CB6FFE','GEN':'#5100FF', 'CONFIN_CONFLICTS': '#B662BF','CONFIN_RULES': '#0400FF'}

    cont = 0   
    f_ordered = list(df_plot.columns)
    f_ordered.reverse()
    for pc in parts['bodies']:
        if raw:
            pc.set_facecolor(colors[cont])
        else:
            pc.set_facecolor(colors[f_ordered[cont]])
        #pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        cont = cont+1
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    ax.set_xlim([-1.3,1.8])
    ax.set_ylim([0.5,len(l)+0.5])
    ax.set_yticks(y0 ,label_n,fontweight ='bold', rotation=0, fontsize = 12)
    ax.set_xlabel("SHAP values",fontsize = 14)

    plt.plot([0,0],[0.5,len(l)+0.5],'--r',linewidth = 2)

    plt.xticks(fontsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    fig.savefig("C:/Users/gvillanueva/Desktop/Technician_2022_2023/Projects/Mental_health_COVID19/reports/figures/risk_factors_"+sufix[target]+".svg", format='svg', dpi=1200,bbox_inches="tight")

    plt.show()

    return l