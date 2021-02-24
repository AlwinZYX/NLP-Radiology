######################
## Import Libraries ##
######################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn import metrics

#################
## Import Data ##
#################
#KKH data used for training & validation from 2007-2013
os.chdir('D:\\Users\\srrzyx\\Desktop\\MRI Brain Text Classification\\MRI_text_classification_v2\\cleaned_data')
data_ACR = pd.read_csv('KKH_ACR_cleaned_v2.csv')
data_non_ACR = pd.read_csv('KKH_non_ACR_cleaned_v2.csv') 
len(data_ACR)/(len(data_ACR)+len(data_non_ACR)) #Blind guess probability of following ACR (baseline accuracy) 

KKH_data = data_ACR[['No.','Indication for MRI', 'ACR/No ACR']].append(data_non_ACR[['No.','Indication for MRI', 'ACR/No ACR']], ignore_index=True) #Combining Dataset

########################
## Cleaning the texts ##
########################
#--------------------------#
#-- Negated text removal --#
#--------------------------#
NegWords = [' NO ']
indication_cleaned = [] #To store cleaned data
removed_text = [] #To store sentence containing text that were removed

def shorten(text, NegWord): #Formula for removing texts after NegWord
    i = text.index(NegWord)
    return text[:i]
    
#Remove negation text   
for i in range(0,len(KKH_data)):
    SentenceList = re.split(r'[.?\-",;:]+',KKH_data['Indication for MRI'][i].upper()) #Split by punctuation
    NewSentenceList = re.split(r'[.?\-",;:]+',KKH_data['Indication for MRI'][i].upper())
    for j in NewSentenceList:
        #Removing negated terms
        for k in NegWords:
            try :
                shorten(j, k)
                NewSentenceList[NewSentenceList.index(j)] = shorten(j, k)
            except :
                pass
    removed_text.append({'removed_text' : ';'.join(list(set(SentenceList) - set(NewSentenceList)))})
    NewSentence = ''.join(NewSentenceList)
    indication_cleaned.append(NewSentence)

KKH_data['Indication for MRI cleaned'] = pd.DataFrame(indication_cleaned)
KKH_data['removed_text'] = pd.DataFrame(removed_text)

#-----------------------#
#-- Normalizing texts --#
#-----------------------#
import nltk
#nltk.download('stopwords') #no need to run if already downloaded
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(df_text_column, data):   
    corpus = []
    for i in range(0, len(data)):
        text = re.sub('[^a-zA-Z]', ' ', df_text_column[i])
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)
        corpus.append(text)
    return corpus

KKH_data['Indication for MRI cleaned'] = clean_text(KKH_data['Indication for MRI cleaned'], KKH_data)

#--------------------------------#
#-- Remove empty cell (if any) --#
#--------------------------------#
KKH_data['Indication for MRI cleaned'].replace('\s+', ' ',regex=True,inplace=True) #Replace cells with multiple whitespaces to single whitespace
KKH_data = KKH_data[KKH_data['Indication for MRI cleaned'] != ' ']
KKH_data.dropna(subset=['Indication for MRI cleaned'], how='any', inplace=True)
KKH_data = KKH_data.reset_index(drop=True)

######################
## Split Train-Test ##
######################
X = KKH_data[['No.','Indication for MRI cleaned']]
y = KKH_data['ACR/No ACR']
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train.value_counts()
y_test.value_counts()
X_test['y_test'] = y_test

# Creating the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer() # Can also try ngram_range=(1,2)
#from sklearn.feature_extraction.text import CountVectorizer
#vect = CountVectorizer() # Can also try ngram_range=(1,2)

####################
## Training Model ##
####################
#-------------------------#
#-- Logistic Regression --#
#-------------------------#
import statsmodels.api as sm

max_features = []
#Iterate to get the maximum number of features/words available
for i in range(1,51):
    try:   
        log_vect = TfidfVectorizer(max_features=i)
        X_logreg = log_vect.fit_transform(KKH_data['Indication for MRI cleaned']).toarray()
        features = log_vect.get_feature_names()
        X_logreg = pd.DataFrame(X_logreg, columns = features)
        X_logreg_train, X_logreg_test, y_logreg_train, y_logreg_test = train_test_split(X_logreg, y, test_size = 0.20, random_state = 0)
        logreg = sm.Logit(y_logreg_train, X_logreg_train)
        lresult = logreg.fit(max_iter = 1)
        max_features.append(i)
    except:
        print("Failed for max features = ", i)
max_features = max(max_features)

log_vect = TfidfVectorizer(max_features=max_features)
X_logreg = log_vect.fit_transform(KKH_data['Indication for MRI cleaned']).toarray()
features = log_vect.get_feature_names()
X_logreg = pd.DataFrame(X_logreg, columns = features)
X_logreg_train, X_logreg_test, y_logreg_train, y_logreg_test = train_test_split(X_logreg, y, test_size = 0.20, random_state = 0)
logreg = sm.Logit(y_logreg_train, X_logreg_train)
lresult = logreg.fit(max_iter = 1)

y_prob_logreg = lresult.predict(X_logreg_test)
y_pred_logreg = np.where(y_prob_logreg > 0.5, 1,0)
lresult.summary2() # Logistic Regression summary

#Performance Metrics
metrics.accuracy_score(y_test, y_pred_logreg) #Accuracy
metrics.roc_auc_score(y_test, y_prob_logreg) #ROC-AUC score
cm_logreg = metrics.confusion_matrix(y_test, y_pred_logreg); cm_logreg #Confusion Matrix
cm_logreg[0,0]/(cm_logreg[0,0]+cm_logreg[0,1]) #Specificity
cm_logreg[1,1]/(cm_logreg[0,1]+cm_logreg[1,1]) #Precision
cm_logreg[1,1]/(cm_logreg[1,0]+cm_logreg[1,1]) #Recall
metrics.f1_score(y_test, y_pred_logreg, average='binary') #F1 Score

#------------------------#
#-- K-Nearest Neigbour --#
#------------------------#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
from sklearn.pipeline import make_pipeline
np.random.seed(0)
knn_pipe = make_pipeline(vect, knn)
knn_pipe.steps
knn_pipe.fit(X_train['Indication for MRI cleaned'], y_train)

from sklearn.model_selection import GridSearchCV
parameters_knn = [{'kneighborsclassifier__n_neighbors': list(range(1,151))}]
grid_search = GridSearchCV(estimator = knn_pipe,
                           param_grid = parameters_knn,
                           scoring = 'roc_auc',
                           cv = 3,
                           verbose=2)
grid_search = grid_search.fit(X_train['Indication for MRI cleaned'], y_train)
grid_mean_scores = grid_search.cv_results_['mean_test_score']
plt.plot(list(range(1,151)), grid_mean_scores)
plt.xlabel('k value')
plt.ylabel('Cross-Validated Accuracy')
best_accuracy_knn = grid_search.best_score_ #Best cross-validation accuracy (not validation accuracy)
best_parameters_knn = grid_search.best_params_ #Best parameters
best_parameters_value_knn = list(best_parameters_knn.values())

np.random.seed(0)
knn = KNeighborsClassifier(n_neighbors = best_parameters_value_knn[0])
knn_pipe = make_pipeline(vect, knn)
knn_pipe.fit(X_train['Indication for MRI cleaned'], y_train)

y_pred_knn = knn_pipe.predict(X_test['Indication for MRI cleaned'])
y_prob_knn = knn_pipe.predict_proba(X_test['Indication for MRI cleaned'])
X_test['y_pred_knn'] = y_pred_knn

#Performance Metrics
metrics.accuracy_score(y_test, y_pred_knn) #Accuracy
metrics.roc_auc_score(y_test, y_prob_knn[:, 1]) #ROC-AUC score
cm_knn = metrics.confusion_matrix(y_test, y_pred_knn); cm_knn #Confusion Matrix
cm_knn[0,0]/(cm_knn[0,0]+cm_knn[0,1]) #Specificity
cm_knn[1,1]/(cm_knn[0,1]+cm_knn[1,1]) #Precision
cm_knn[1,1]/(cm_knn[1,0]+cm_knn[1,1]) #Recall
metrics.f1_score(y_test, y_pred_knn, average='binary') #F1 Score

#-------------------#
#-- Random Forest --#
#-------------------#
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
from sklearn.pipeline import make_pipeline
np.random.seed(0)
rf_pipe = make_pipeline(vect, rf)
rf_pipe.steps
rf_pipe.fit(X_train['Indication for MRI cleaned'], y_train)

#Grid Search
from sklearn.model_selection import GridSearchCV
parameters_rf = [{'randomforestclassifier__n_estimators': [10,50,100,300,500,1000,2000,3000,5000],
                  'randomforestclassifier__max_depth': [None,3,5,7,10,15,20],
                  'randomforestclassifier__criterion': ["gini", "entropy"],
                  'randomforestclassifier__max_features': ["auto", "log2", None]}]

grid_search = GridSearchCV(estimator = rf_pipe,
                           param_grid = parameters_rf,
                           scoring = 'roc_auc',
                           cv = 3,
                           verbose=2,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train['Indication for MRI cleaned'], y_train)
best_accuracy_rf = grid_search.best_score_ #Best cross-validation accuracy (not validation accuracy)
best_parameters_rf = grid_search.best_params_ #Best parameters
best_parameters_value_rf = list(best_parameters_rf.values())

np.random.seed(0)
rf = RandomForestClassifier(n_estimators = best_parameters_value_rf[3],
                            max_depth = best_parameters_value_rf[1],
                            criterion = best_parameters_value_rf[0],
                            max_features = best_parameters_value_rf[2])
rf_pipe = make_pipeline(vect, rf)
rf_pipe.fit(X_train['Indication for MRI cleaned'], y_train)

y_pred_rf = rf_pipe.predict(X_test['Indication for MRI cleaned'])
y_prob_rf = rf_pipe.predict_proba(X_test['Indication for MRI cleaned'])
X_test['y_pred_rf'] = y_pred_rf

#Performance Metrics
metrics.accuracy_score(y_test, y_pred_rf) #Accuracy
metrics.roc_auc_score(y_test, y_prob_rf[:, 1]) #ROC-AUC score
cm_rf = metrics.confusion_matrix(y_test, y_pred_rf); cm_rf #Confusion Matrix
cm_rf[0,0]/(cm_rf[0,0]+cm_rf[0,1]) #Specificity
cm_rf[1,1]/(cm_rf[0,1]+cm_rf[1,1]) #Precision
cm_rf[1,1]/(cm_rf[1,0]+cm_rf[1,1]) #Recall
metrics.f1_score(y_test, y_pred_rf, average='binary') #F1 Score

#Random Forest Feature Importance
max_num_features = 15 #Number of features to display on feature importance plot
rf_importance = pd.DataFrame({'importance_value':rf_pipe.steps[1][1].feature_importances_}) #Based on "gini importance" or "mean decrease impurity"
rf_importance['feature_name'] = vect.get_feature_names()
rf_importance = rf_importance.sort_values(by='importance_value', ascending=False)[0:max_num_features]

plt.figure()
plt.title('RandomForest Feature Importance Plot', fontweight='bold')
plt.barh(list(rf_importance['feature_name'][0:max_num_features]), list(rf_importance['importance_value'][0:max_num_features]), color='b', align='center')
plt.xlabel('Relative Importance', fontweight='bold')
plt.ylabel('Features', fontweight='bold')
plt.gca().invert_yaxis()

 #Identifying false negative cases
#cm_rf[1,0] #Number of false negative cases
#false_negative_list = X_test[(X_test['y_test'] == 1) & (X_test['y_pred_rf'] == 0)]['No.']
#false_negative_cases = data_ACR[data_ACR['No.'].isin(list(false_negative_list))]
#os.chdir('D:\\Users\\srrzyx\\Desktop')
#false_negative_cases.to_csv('false_negative_cases.csv',index=False)

#-------------#
#-- XGBoost --#
#-------------#
from xgboost import XGBClassifier
xgb = XGBClassifier()
from sklearn.pipeline import make_pipeline
np.random.seed(0)
xgb_pipe = make_pipeline(vect, xgb)
xgb_pipe.steps
xgb_pipe.fit(X_train['Indication for MRI cleaned'], y_train)

#Grid Search
from sklearn.model_selection import GridSearchCV
parameters_xgb = [{'xgbclassifier__colsample_bytree': [0.05,0.1,0.3,0.5,1],
                   'xgbclassifier__max_depth': [1,3,5,10,15,20],
                   'xgbclassifier__n_estimators': [50, 100, 150, 300, 500, 750, 1000],
                   'xgbclassifier__reg_alpha': [0.01,0.05,0.1,0.2],
                   'xgbclassifier__gamma': [0.3,0.5,0.7,1],               
                   'xgbclassifier__subsample': [0.5,0.7,1]},]

grid_search = GridSearchCV(estimator = xgb_pipe,
                           param_grid = parameters_xgb,
                           scoring = 'roc_auc',
                           cv = 3,
                           verbose=2,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train['Indication for MRI cleaned'], y_train)
best_accuracy_xgb = grid_search.best_score_ #Best cross-validation accuracy (not validation accuracy)
best_parameters_xgb = grid_search.best_params_ #Best parameters
best_parameters_value_xgb = list(best_parameters_xgb.values())

np.random.seed(0)
xgb = XGBClassifier(n_estimators = best_parameters_value_xgb[3], 
                    max_depth = best_parameters_value_xgb[2], 
                    colsample_bytree = best_parameters_value_xgb[0], 
                    subsample = best_parameters_value_xgb[5], 
                    reg_alpha = best_parameters_value_xgb[4],
                    gamma = best_parameters_value_xgb[1])
xgb_pipe = make_pipeline(vect, xgb)
xgb_pipe.fit(X_train['Indication for MRI cleaned'], y_train)

y_pred_xgb = xgb_pipe.predict(X_test['Indication for MRI cleaned'])
y_prob_xgb = xgb_pipe.predict_proba(X_test['Indication for MRI cleaned'])
X_test['y_pred_xgb'] = y_pred_xgb

#Performance Metrics
metrics.accuracy_score(y_test, y_pred_xgb) #Accuracy
metrics.roc_auc_score(y_test, y_prob_xgb[:, 1]) #ROC-AUC score
cm_xgb = metrics.confusion_matrix(y_test, y_pred_xgb); cm_xgb #Confusion Matrix
cm_xgb[0,0]/(cm_xgb[0,0]+cm_xgb[0,1]) #Specificity
cm_xgb[1,1]/(cm_xgb[0,1]+cm_xgb[1,1]) #Precision
cm_xgb[1,1]/(cm_xgb[1,0]+cm_xgb[1,1]) #Recall
metrics.f1_score(y_test, y_pred_xgb, average='binary') #F1 Score

# XGB Feature Importance
max_num_features = 15 #Number of features to display on feature importance plot
from xgboost import plot_importance
plot_importance(xgb_pipe.steps[1][1], max_num_features = max_num_features, importance_type='weight') #But no feature names

#Alternative feature importance plot
f_score = pd.DataFrame({'keys':list(xgb.get_booster().get_fscore().keys()), 'f_score': list(xgb.get_booster().get_fscore().values())})
f_score = f_score.sort_values(by=['f_score'], ascending=False).reset_index(drop=True) # Sort by f_score and reset index
f_score['keys'].replace('f', '', regex=True, inplace=True) #Remove letter 'f' from the keys column
f_score['feature'] = 0
for i in range(0,len(f_score)):
    f_score['feature'][i] = vect.get_feature_names()[int(f_score['keys'][i])]
    
#plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(list(f_score['feature'][0:max_num_features]), list(f_score['f_score'][0:max_num_features]), color='blue', align='center')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('F score', fontweight='bold')
ax.set_ylabel('Features', fontweight='bold')
ax.set_title('XGBoost Feature Importance Plot', fontweight='bold')
for i, v in enumerate(list(f_score['f_score'][0:max_num_features])):
    ax.text(v + 0.5, i + .20, str(v), color='gray', fontsize=9)
plt.show()

#Identifying false negative cases
#cm_xgb[1,0] #Number of false negative cases
#false_negative_list = X_test[(X_test['y_test'] == 1) & (X_test['y_pred_xgb'] == 0)]['No.']
#false_negative_cases = data_ACR[data_ACR['No.'].isin(list(false_negative_list))]
#os.chdir('D:\\Users\\srrzyx\\Desktop')
#false_negative_cases.to_csv('false_negative_cases.csv',index=False)

#---------------#
#-- ROC Curve --#
#---------------#
plt.figure(0).clf()

plt.title('ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_logreg) #Logistic Regression
plt.plot(fpr, tpr, 'r', label = 'LR AUC = %0.2f' % metrics.roc_auc_score(y_test, y_prob_logreg))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_knn[:, 1]) #k-Nearest Neighbor
plt.plot(fpr, tpr, 'y', label = 'KNN AUC = %0.2f' % metrics.roc_auc_score(y_test, y_prob_knn[:, 1]))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_knn[:, 1]) #Random Forest
plt.plot(fpr, tpr, 'b', label = 'RF AUC = %0.2f' % metrics.roc_auc_score(y_test, y_prob_rf[:, 1]))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_xgb[:, 1]) #XGboost
plt.plot(fpr, tpr, 'g', label = 'XGB AUC = %0.2f' % metrics.roc_auc_score(y_test, y_prob_xgb[:, 1]))
plt.plot([0, 1], [0, 1],'k--') #Baseline
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=0)

plt.show()

#######################
## LIME - Validation ##
#######################
#-------------------#
#-- Random Forest --#
#-------------------#
from lime.lime_text import LimeTextExplainer
patient_no = 659 #Input patient case number ('No.') that you want to review
index_no = X_test[X_test['No.'] == patient_no].index.tolist()[0]
class_names = ['Non_ACR', 'ACR']
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(X_test['Indication for MRI cleaned'][index_no], rf_pipe.predict_proba, num_features=6)
exp.as_list()
exp.as_pyplot_figure();

print('Patient ID: %d' % index_no)
print('Probability of following ACR:', rf_pipe.predict_proba([X_test['Indication for MRI cleaned'][index_no]])[0,1])
print('True class: %s' % class_names[y_test[index_no]])
print('Indication for MRI / History:', list(data_ACR[data_ACR['No.'] == X_test.loc[index_no,]['No.']]['Indication for MRI']))

#-------------#
#-- XGBoost --#
#-------------#
from lime.lime_text import LimeTextExplainer
patient_no = 659 #Input patient case number ('No.') that you want to review
index_no = X_test[X_test['No.'] == patient_no].index.tolist()[0]
class_names = ['Non_ACR', 'ACR']
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(X_test['Indication for MRI cleaned'][index_no], xgb_pipe.predict_proba, num_features=6)
exp.as_list()
exp.as_pyplot_figure();

print('Patient ID: %d' % index_no)
print('Probability of following ACR:', xgb_pipe.predict_proba([X_test['Indication for MRI cleaned'][index_no]])[0,1])
print('True class: %s' % class_names[y_test[index_no]])
print('Indication for MRI / History:', list(data_ACR[data_ACR['No.'] == X_test.loc[index_no,]['No.']]['Indication for MRI']))

##########################
## Predict New KKH data ##
##########################
#Load new KKH data (2014-2017)
os.chdir('D:\\Users\\srrzyx\\Desktop\\MRI Brain Text Classification\\MRI_text_classification_v2\\NewData\\KKH_newdata(2014-2017)\\Original')
new_data = pd.read_excel('KKH_201401-201709 annonymised.xls')
new_data.columns.tolist()

#Negated text removal
NegWords = [' NO ']
indication_cleaned = [] #To store cleaned data
removed_text = [] #To store sentence containing text that were removed

def shorten(text, NegWord): #Formula for removing texts after NegWord
    i = text.index(NegWord)
    return text[:i]

for i in range(0,len(new_data)):
    SentenceList = re.split(r'[.?\-",;:]+',new_data['Indication/History'][i].upper()) #Split by punctuation
    NewSentenceList = re.split(r'[.?\-",;:]+',new_data['Indication/History'][i].upper())
    for j in NewSentenceList:
        #Removing negated terms
        for k in NegWords:
            try :
                shorten(j, k)
                NewSentenceList[NewSentenceList.index(j)] = shorten(j, k)
            except :
                pass
    removed_text.append({'removed_text' : ';'.join(list(set(SentenceList) - set(NewSentenceList)))})
    NewSentence = ''.join(NewSentenceList)
    indication_cleaned.append(NewSentence)

new_data['Indication_History_cleaned'] = pd.DataFrame(indication_cleaned)
new_data['removed_text'] = pd.DataFrame(removed_text)

#Data Cleaning
new_data['Indication_History_cleaned'].replace('\s+', ' ',regex=True,inplace=True) #Replace cells with multiple whitespaces to single whitespace
new_data.dropna(subset=['Indication_History_cleaned'], how='any', inplace=True) #Drop rows if text is null
new_data = new_data[(new_data['Indication_History_cleaned'] != ' ') & (new_data['Indication_History_cleaned'] != '. ')]
new_data.dropna(subset=['Follow_ACR'], how='any', inplace=True)
new_data = new_data.reset_index(drop=True)
new_data['Indication_History_cleaned'] = clean_text(new_data['Indication_History_cleaned'], new_data) #Clean data using created function 'clean_text'

#Preparing data for prediction
X_newtest = new_data[['Annonymised ID','Indication_History_cleaned']]
y_newtest = new_data['Follow_ACR']
X_newtest['y_newtest'] = y_newtest
y_newtest.value_counts()[1]/(y_newtest.value_counts()[1]+y_newtest.value_counts()[0]) #Blind guess probability of following ACR (baseline accuracy) 

#-------------------------#
#-- Logistic Regression --#
#-------------------------#
log_vect_new = TfidfVectorizer(max_features=max_features) 
X_logreg_new = log_vect_new.fit_transform(X_newtest['Indication_History_cleaned']).toarray()
features_new = log_vect_new.get_feature_names()
X_logreg_new = pd.DataFrame(X_logreg_new, columns = features)

#Predicting the new data
y_newprob_logreg = lresult.predict(X_logreg_new)
y_newpred_logreg = np.where(y_newprob_logreg > 0.5, 1,0)
X_newtest['y_newpred_logreg'] = y_newpred_logreg

#Performance Metrics
metrics.accuracy_score(y_newtest, y_newpred_logreg) #Accuracy
metrics.roc_auc_score(y_newtest, y_newprob_logreg) #ROC-AUC score
cm_newdata_logreg = metrics.confusion_matrix(y_newtest, y_newpred_logreg); cm_newdata_logreg #Confusion Matrix
cm_newdata_logreg[0,0]/(cm_newdata_logreg[0,0]+cm_newdata_logreg[0,1]) #Specificity
cm_newdata_logreg[1,1]/(cm_newdata_logreg[0,1]+cm_newdata_logreg[1,1]) #Precision
cm_newdata_logreg[1,1]/(cm_newdata_logreg[1,0]+cm_newdata_logreg[1,1]) #Recall
metrics.f1_score(y_newtest, y_newpred_logreg, average='binary') #F1 Score

#------------------------#
#-- K-Nearest Neigbour --#
#------------------------#
#Predicting the new data
y_newpred_knn = knn_pipe.predict(X_newtest['Indication_History_cleaned'])
y_newprob_knn = knn_pipe.predict_proba(X_newtest['Indication_History_cleaned'])
X_newtest['y_newpred_knn'] = y_newpred_knn

#Performance Metrics
metrics.accuracy_score(y_newtest, y_newpred_knn) #Accuracy
metrics.roc_auc_score(y_newtest, y_newprob_knn[:, 1]) #ROC-AUC score
cm_newdata_knn = metrics.confusion_matrix(y_newtest, y_newpred_knn); cm_newdata_knn #Confusion Matrix
cm_newdata_knn[0,0]/(cm_newdata_knn[0,0]+cm_newdata_knn[0,1]) #Specificity
cm_newdata_knn[1,1]/(cm_newdata_knn[0,1]+cm_newdata_knn[1,1]) #Precision
cm_newdata_knn[1,1]/(cm_newdata_knn[1,0]+cm_newdata_knn[1,1]) #Recall
metrics.f1_score(y_newtest, y_newpred_knn, average='binary') #F1 Score

#Identifying false negative cases
#cm_newdata_knn[1,0] #Number of false negative cases
#false_negative_new_list = X_newtest[(X_newtest['y_newtest'] == 1) & (X_newtest['y_newpred_knn'] == 0)]['Annonymised ID']
#false_negative_new_cases = new_data[new_data['Annonymised ID'].isin(list(false_negative_new_list))]
#os.chdir('D:\\Users\\srrzyx\\Desktop')
#false_negative_new_cases.to_csv('false_negative_cases.csv',index=False)

#-------------------#
#-- Random Forest --#
#-------------------#
#Predicting the new data
y_newpred_rf = rf_pipe.predict(X_newtest['Indication_History_cleaned'])
y_newprob_rf = rf_pipe.predict_proba(X_newtest['Indication_History_cleaned'])
X_newtest['y_newpred_rf'] = y_newpred_rf

#Performance Metrics
metrics.accuracy_score(y_newtest, y_newpred_rf) #Accuracy
metrics.roc_auc_score(y_newtest, y_newprob_rf[:, 1]) #ROC-AUC score
cm_newdata_rf = metrics.confusion_matrix(y_newtest, y_newpred_rf); cm_newdata_rf #Confusion Matrix
cm_newdata_rf[0,0]/(cm_newdata_rf[0,0]+cm_newdata_rf[0,1]) #Specificity
cm_newdata_rf[1,1]/(cm_newdata_rf[0,1]+cm_newdata_rf[1,1]) #Precision
cm_newdata_rf[1,1]/(cm_newdata_rf[1,0]+cm_newdata_rf[1,1]) #Recall
metrics.f1_score(y_newtest, y_newpred_rf, average='binary') #F1 Score

#Identifying false negative cases
#cm_newdata[1,0] #Number of false negative cases
#false_negative_new_list = X_newtest[(X_newtest['y_newtest'] == 1) & (X_newtest['y_newpred_rf'] == 0)]['Annonymised ID']
#false_negative_new_cases = new_data[new_data['Annonymised ID'].isin(list(false_negative_new_list))]
#os.chdir('D:\\Users\\srrzyx\\Desktop')
#false_negative_new_cases.to_csv('false_negative_cases.csv',index=False)

#-------------#
#-- XGBoost --#
#-------------#
#Predicting the new data
y_newpred_xgb = xgb_pipe.predict(X_newtest['Indication_History_cleaned'])
y_newprob_xgb = xgb_pipe.predict_proba(X_newtest['Indication_History_cleaned'])
X_newtest['y_newpred_xgb'] = y_newpred_xgb

#Performance Metrics
metrics.accuracy_score(y_newtest, y_newpred_xgb) #Accuracy
metrics.roc_auc_score(y_newtest, y_newprob_xgb[:, 1]) #ROC-AUC score
cm_newdata_xgb = metrics.confusion_matrix(y_newtest, y_newpred_xgb); cm_newdata_xgb #Confusion Matrix
cm_newdata_xgb[0,0]/(cm_newdata_xgb[0,0]+cm_newdata_xgb[0,1]) #Specificity
cm_newdata_xgb[1,1]/(cm_newdata_xgb[0,1]+cm_newdata_xgb[1,1]) #Precision
cm_newdata_xgb[1,1]/(cm_newdata_xgb[1,0]+cm_newdata_xgb[1,1]) #Recall
metrics.f1_score(y_newtest, y_newpred_xgb, average='binary') #F1 Score

#Identifying false negative cases
#cm_newdata[1,0] #Number of false negative cases
#false_negative_new_list = X_newtest[(X_newtest['y_newtest'] == 1) & (X_newtest['y_newpred_xgb'] == 0)]['Annonymised ID']
#false_negative_new_cases = new_data[new_data['Annonymised ID'].isin(list(false_negative_new_list))]
#os.chdir('D:\\Users\\srrzyx\\Desktop')
#false_negative_new_cases.to_csv('false_negative_cases.csv',index=False)

#---------------#
#-- ROC Curve --#
#---------------#
plt.figure(0).clf()

plt.title('ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(y_newtest, y_newprob_logreg) #Logistic Regression
plt.plot(fpr, tpr, 'r', label = 'LR AUC = %0.2f' % metrics.roc_auc_score(y_newtest, y_newprob_logreg))
fpr, tpr, thresholds = metrics.roc_curve(y_newtest, y_newprob_knn[:, 1]) #k-Nearest Neighbor
plt.plot(fpr, tpr, 'y', label = 'KNN AUC = %0.2f' % metrics.roc_auc_score(y_newtest, y_newprob_knn[:, 1]))
fpr, tpr, thresholds = metrics.roc_curve(y_newtest, y_newprob_rf[:, 1]) #Random Forest
plt.plot(fpr, tpr, 'b', label = 'RF AUC = %0.2f' % metrics.roc_auc_score(y_newtest, y_newprob_rf[:, 1]))
fpr, tpr, thresholds = metrics.roc_curve(y_newtest, y_newprob_xgb[:, 1]) #XGboost
plt.plot(fpr, tpr, 'g', label = 'XGB AUC = %0.2f' % metrics.roc_auc_score(y_newtest, y_newprob_xgb[:, 1]))
plt.plot([0, 1], [0, 1],'k--') #Baseline
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=0)

plt.show()

#################
## LIME - Test ##
#################
#-------------------#
#-- Random Forest --#
#-------------------#
#Predicting the output using RF & explaining using LIME
index_no = 1473 #Input index number
ID_no = X_newtest['Annonymised ID'][index_no]
class_names = ['Non_ACR', 'ACR']
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(X_newtest['Indication_History_cleaned'][index_no], rf_pipe.predict_proba, num_features=6)
exp.as_list()
exp.as_pyplot_figure();

print('Patient ID:', ID_no)
print('Probability of following ACR:', rf_pipe.predict_proba([X_newtest['Indication_History_cleaned'][index_no]])[0,1])
print('True class: %s' % class_names[int(y_newtest[index_no])])
print('Indication for MRI / History:', list(new_data[new_data['Annonymised ID'] == ID_no]['Indication/History']))

#-------------#
#-- XGBoost --#
#-------------#
#Predicting the output using RF & explaining using LIME
index_no = 1473 #Input index number
ID_no = X_newtest['Annonymised ID'][index_no]
class_names = ['Non_ACR', 'ACR']
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(X_newtest['Indication_History_cleaned'][index_no], rf_pipe.predict_proba, num_features=6)
exp.as_list()
exp.as_pyplot_figure();

print('Patient ID:', ID_no)
print('Probability of following ACR:', xgb_pipe.predict_proba([X_newtest['Indication_History_cleaned'][index_no]])[0,1])
print('True class: %s' % class_names[int(y_newtest[index_no])])
print('Indication for MRI / History:', list(new_data[new_data['Annonymised ID'] == ID_no]['Indication/History']))
