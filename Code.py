#importing working libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score , classification_report , accuracy_score
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split ,KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler ,  StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegressionCV      
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from scipy . stats import zscore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from math import pi
import warnings
warnings.simplefilter("ignore")




#importing the data and making a modifiable copy of it
original_data = pd.read_csv("oil_datas.csv") #importing the data
data_set = original_data.copy() #making a copy of the data

#exammining the missing data in the data_set
missing_data = data_set.isnull().sum().sum()
missing_values_in_features = data_set.isna().sum()
#storing the features names with missing data into groups
missing_nt = ["xtw","FL_WV","PHV","THV","THSKV","THSGIV","THSGIHV","GIHV","SSSVStatus"]
missing_n=[" Natural Flowing","Communication","Conductor Vibration","TxA"]
missing_num=["B Annulus","H2S PPM"," BOPD Impact","G.Gas  MMSCFD"]
replace_num=["B Annulus","H2S PPM"," BOPD Impact","G.Gas  MMSCFD"]
num_data=["H2S PPM","B Annulus"," BOPD Impact","G.Gas  MMSCFD"]
outlier_data=["B Annulus","H2S PPM"," BOPD Impact","G.Gas  MMSCFD"]
cat_data=["SV","UMV","LMV","xtw","FL_WV","PHV","THV","THSKV","THSGIV","THSGIHV","GIHV","TxA","SSSVStatus"," Natural Flowing","Communication","Conductor Vibration","Complex vs Satellite"," Ship Lane Hazard"," Structure Problem","Resort Proximity","RISK Category"]
cat_data2=["SV","UMV","LMV","xtw","FL_WV","PHV","THV","THSKV","THSGIV","THSGIHV","GIHV","TxA","SSSVStatus"," Natural Flowing","Communication","Conductor Vibration","Complex vs Satellite"," Ship Lane Hazard"," Structure Problem","Resort Proximity"]

colomns_names = []
for col in data_set.columns: 
    colomns_names.append(col)
colomns_inputs = colomns_names.copy()
colomns_inputs.remove("RISK Category")
colomns_names.remove("X/T_WV")
colomns_names.insert(0,"xtw")
data_set.columns = colomns_names
#Filling missing data in each feature individaully according to the instructions of dr/mostafa
data_no_missing = data_set.copy()
data_no_missing[missing_nt]=data_no_missing[missing_nt].fillna("NT")
data_no_missing[missing_n]=data_no_missing[missing_n].fillna("N")
data_no_missing[missing_num]=data_no_missing[missing_num].fillna(0)
data_no_missing["Complex vs Satellite"]=data_no_missing["Complex vs Satellite"].fillna("S")
data_no_missing[" Ship Lane Hazard"]=data_no_missing[" Ship Lane Hazard"].fillna("N")
data_no_missing[" Structure Problem"]=data_no_missing[" Structure Problem"].fillna("N")
data_no_missing["Resort Proximity"]=data_no_missing["Resort Proximity"].fillna("N")
data_no_missing[cat_data]=data_no_missing[cat_data].fillna("N")


#Replacing not tested (NT) in numerical features
data_num = data_no_missing.copy()
data_num[replace_num] =data_num[replace_num].replace("NT",0)


#Exammining the type of values in each feature
data_type=data_num.dtypes.copy()
#Changing the type of numerical data into numeric
data_num[num_data]=data_num[num_data].astype("float")
data_num[cat_data]=data_num[cat_data].astype("str")
data_final=data_num.copy()

#Removing outliers
for i in outlier_data :
    data_final["z"] = zscore(data_final[i])
    data_final["z"] = data_final["z"].apply(lambda x: x <= -3 or x >= 3)
    data_final[data_final["z"]]
    data_final = data_final[data_final["z"]==False]
    data_final=data_final.drop("z",axis=1)
#Statestical analysis for the data
general_stat = data_final.describe()
general_stat.to_csv(r'general_stat .csv')
median_stat = data_final.median()
median_stat.to_csv(r'median_stat .csv')

variance = data_final.var()
variance.to_csv(r'variance .csv')

print("Genral statistics :" ,general_stat)
print("median : ", median_stat)
print ("variance : " , variance)
#correlations
correlation = data_final.corr() 
print ("correlation", correlation)
correlation.to_csv(r'correlation .csv')


#labelling catigorical features
le = preprocessing.LabelEncoder()
data_final[cat_data] = data_final[cat_data].apply(lambda x: le.fit_transform(x))

data_final[cat_data]=data_final[cat_data].astype("category")


#ANOVA
nova =pd.DataFrame(columns =["F-value","p-value","Feature"])
for i in num_data : 
    nova_test = np.array(f_oneway(data_final[[i]],data_final[["RISK Category"]]))
    F_value_test=nova_test[0][0]
    p_value_test=nova_test[1][0]
    nova=nova.append(pd.Series([F_value_test,p_value_test,i], index=nova.columns ), ignore_index=True) 
    
print ("Anova test : ",nova )



for i in colomns_names:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,8))
    plt.xlabel(i)
    sns.distplot(data_final[[i]])
    plt.savefig('%s sdistplot.png'%i)#scatter graphs
for i in colomns_names:
    plt.scatter(data_final[[i]],data_final[["RISK Category"]])
    plt.xlabel(i)
    plt.ylabel("RISK Category")
    plt.savefig('%s scatter.png'%i)
    plt.show()
#jitterong with stribplot
for i in colomns_names :
        fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
        sns.stripplot(x=data_final[[i]], hue=data_final[["RISK Category"]],jitter=0.25, size=8, ax=ax, linewidth=.5)
        plt.xlabel(i)
        plt.savefig('%s stripplot.png'%i)
        plt.show()
#box plot for nummerical data
for i in num_data:
    sns.boxplot(data_final[[i]])
    plt.xlabel(i)
    plt.savefig('%s boxplot.png'%i)
    plt.show()
#graph box plot in relation to the category
for i in num_data:
    plt.figure(figsize=(13,10), dpi= 80)
    sns.boxplot(x=i, y="RISK Category", data=data_final, notch=False)
    plt.xlabel(i)
    plt.show()
#bar plot for the data
for i in num_data:
    sns.set(style="whitegrid")
    sns.barplot(x="RISK Category", y=i, data=data_final)
    plt.xlabel("RISK Category")
    plt.ylabel(i)
    plt.savefig('%s barplot.png'%i)
    plt.show()
   #countplot chart
    for i in cat_data:
        sns.set(style='darkgrid')
        plt.figure(figsize=(10,10))
        plt.xlabel(i)
        sns.countplot(x=i, data=data_final)
        plt.savefig('%s countplot.png'%i)
        plt.show()
for i in num_data:        
    sns.jointplot(x=data_final["RISK Category"], y=data_final[i], kind='scatter')
    plt.xlabel("RISK Category")
    plt.ylabel(i)
    plt.ylim(-250,750)
    plt.savefig('%s jointplot scatter .png'%i)
    plt.show()
    sns.jointplot(x=data_final["RISK Category"], y=data_final[i], kind='hex')
    plt.xlabel("RISK Category")
    plt.ylabel(i)
    plt.ylim(-250,750)
    plt.savefig('%s jointplot hex .png'%i)
    plt.show()
    sns.jointplot(x=data_final["RISK Category"], y=data_final[i], kind='kde')
    plt.xlabel("RISK Category")
    plt.ylabel(i)
    plt.ylim(-250,750)
    plt.savefig('%s jointplot kde .png'%i)
    plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(data_final['RISK Category'], data_final['B Annulus'], data_final['H2S PPM'], cmap=plt.cm.viridis, linewidth=0.2)
# to Add a color bar which maps values to colors.
surf=ax.plot_trisurf(data_final['RISK Category'], data_final['B Annulus'], data_final['H2S PPM'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()



# Rotate it
ax.view_init(30, 45)
plt.show()
 
# Other palette
ax.plot_trisurf(data_final['RISK Category'], data_final['B Annulus'], data_final['H2S PPM'], cmap=plt.cm.jet, linewidth=0.01)
plt.savefig(' plot_trisurf 3d.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_final['RISK Category'], data_final['B Annulus'], data_final['H2S PPM'], c='skyblue', s=60)
ax.view_init(30, 185)
plt.savefig('scatter 3d .png')

plt.show()


scaler=MinMaxScaler()
data_final[num_data]=scaler.fit_transform(data_final[num_data])

x = data_final.drop("RISK Category",axis=1)
y=data_final[["RISK Category"]]

ranks = []
for i in range(1,25):
    ranks.append(i)
forest_c=RandomForestClassifier().fit(x,y)
forest_importances = pd.DataFrame(forest_c.feature_importances_ , columns=["Score"])
forest_importances["Gen"] = colomns_inputs
forest_importances.to_csv(r'random forest ranking.csv')
imp_graph = forest_c.feature_importances_
indices = np.argsort(imp_graph)[::-1]
std = np.std([tree.feature_importances_ for tree in forest_c.estimators_],
             axis=0)
plt.figure(figsize=(200,100))
plt.title("Feature importances forest")
plt.xlabel(colomns_inputs)
plt.bar(range(x.shape[1]), imp_graph[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.savefig('Feature importances forest')
plt.show()

x_ranking = 
from sklearn.linear_model import LogisticRegression
formula_reg = LogisticRegression().fit(x,y)
decision =formula_reg.coef_
colls = x.columns
decision = pd.DataFrame(decision , columns = colls , index = ["cat1", "cat2" , "cat3" , "cat4" , "cat 5"])


plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

forest10 = forest_importances.head(20)
 
# number of variable
categories=list(forest10.iloc[:,1])
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=list((forest10.iloc[:,0])*200)
values += values[:1]

 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20,50,50,100],  color="grey", size=7)
plt.ylim(0,40)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)
plt.title("forest spyder ranking")
plt.savefig('forest spyder')
plt.savefig(' importance spider .png')

plt.show()


encoder = OneHotEncoder()
encoder.fit(data_final[cat_data2])
dummy = encoder.transform(data_final[cat_data2]).toarray()
dummy_n = encoder.get_feature_names(cat_data2)
dummy = pd.DataFrame(dummy , columns = dummy_n)
data_final = data_final.drop(cat_data2 , axis=1)
data_final = data_final.join(dummy)

data_final= data_final.dropna( axis=0, how='any', thresh=None, subset=None, inplace=False)
#histogram 

#spliting data into inputs and outputs
x = data_final.drop("RISK Category",axis=1)
y=data_final[["RISK Category"]]


#splitting the data into train and test with test data size of 20% 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,stratify=y,random_state=3)

#Using svm model 

def svm_builder():
    pip_svm = Pipeline([("selector",SelectKBest(chi2)),("svm_clf",SVC())])
    parameters_svm ={ 'selector__k':[20],
                "svm_clf__C":[10],"svm_clf__kernel":[ 'poly', 'rbf', 'sigmoid' ],
                "svm_clf__degree":[7],"svm_clf__random_state":[7]}
    scorer_svm = make_scorer(accuracy_score)
    searcher_svm = GridSearchCV(pip_svm, parameters_svm, scoring=scorer_svm)
    return searcher_svm
svm_clf=svm_builder().fit(x_train,y_train)
score_svm = svm_clf.score(x_test,y_test)
svm_pred = svm_clf.predict(x_test)
svm_cm = confusion_matrix(y_test,svm_pred)
svm_recall = recall_score(y_test,svm_pred , average="micro")
svm_precision = precision_score(y_test,svm_pred , average="micro")
svm_recall_none = recall_score(y_test,svm_pred , average=None)
svm_precision_none = precision_score(y_test,svm_pred , average=None)
svm_f1 = f1_score(y_test,svm_pred , average="micro")
svm_f1_none = f1_score(y_test,svm_pred , average=None)
svm_clf.score(x_train,y_train)

print ("SVM Recal details : ", svm_recall_none)
print ("SVM precition details : ", svm_precision_none)
print ( "SVM F1 Score Details : " , svm_f1_none)


#Using decision tree model 
def tree_builder():
    pip_tree = Pipeline([("selector",SelectKBest(chi2)),("tree_clf",tree.DecisionTreeClassifier())])
    parameters_tree ={ 'selector__k':[20],
                     "tree_clf__random_state":[7],"tree_clf__min_samples_split":[13],
                     "tree_clf__max_depth":[12]}
    scorer_tree = make_scorer(accuracy_score)
    searcher_tree = GridSearchCV(pip_tree, parameters_tree, scoring=scorer_tree)
    return searcher_tree
tree_clf=tree_builder().fit(x_train,y_train)
score_tree = tree_clf.score(x_test,y_test)
tree_pred = tree_clf.predict(x_test)
tree_cm = confusion_matrix(y_test,tree_pred)
tree_recall = recall_score(y_test,tree_pred , average="micro")
tree_precision = precision_score(y_test,tree_pred , average="micro")
tree_recall_none = recall_score(y_test,tree_pred , average=None)
tree_precision_none = precision_score(y_test,tree_pred , average=None)
tree_f1 = f1_score(y_test,tree_pred , average="micro")
tree_f1_none = f1_score(y_test,tree_pred , average=None)
tree_clf.score(x_train,y_train)

print ("decision tree Recal details : ", tree_recall_none)
print ("decision tree precition details : ", tree_precision_none)
print ( "decision tree F1 Score Details : " , tree_f1_none)
#Using logistic regression model 
def log_builder():
    pip_log = Pipeline([("selector",SelectKBest(chi2)),("log_clf",LogisticRegressionCV())])
    parameters_tree ={ 'selector__k':[20],
                "log_clf__cv":[5],"log_clf__random_state":[7]}
    scorer_log = make_scorer(accuracy_score)
    searcher_log = GridSearchCV(pip_log, parameters_tree, scoring=scorer_log)
    return searcher_log
log_clf=log_builder().fit(x_train,y_train)
score_log = log_clf.score(x_test,y_test)
log_pred = log_clf.predict(x_test)
log_cm = confusion_matrix(y_test,log_pred)
log_recall = recall_score(y_test,log_pred , average="micro")
log_precision = precision_score(y_test,log_pred , average="micro")
log_recall_none = recall_score(y_test,log_pred , average=None)
log_precision_none = precision_score(y_test,log_pred , average=None)
log_f1 = f1_score(y_test,log_pred , average="micro")
log_f1_none = f1_score(y_test,log_pred , average=None)
log_clf.score(x_train,y_train)

print ("logistic regression Recal details : ", log_recall_none)
print ("logistic regression precition details : ", log_precision_none)
print ( "logistic regression F1 Score Details : " , log_f1_none)

#Using random forest model 
def forest_builder():
    pip_forest = Pipeline([("selector",SelectKBest(chi2)),("forest_clf",RandomForestClassifier())])
    parameters_forest ={'selector__k':[20],"forest_clf__random_state":[5],
                "forest_clf__n_estimators":[200]}
    scorer_forest = make_scorer(accuracy_score)
    searcher_forest = GridSearchCV(pip_forest, parameters_forest, scoring=scorer_forest)
    return searcher_forest
forest_clf=forest_builder().fit(x_train,y_train)
score_forest = forest_clf.score(x_test,y_test)
forest_pred = forest_clf.predict(x_test)
forest_cm = confusion_matrix(y_test,forest_pred)
forest_recall = recall_score(y_test,forest_pred , average="micro")
forest_precision = precision_score(y_test,forest_pred , average="micro")
forest_recall_none = recall_score(y_test,forest_pred , average=None)
forest_precision_none = precision_score(y_test,forest_pred , average=None)
forest_f1 = f1_score(y_test,forest_pred , average="micro")
forest_f1_none = f1_score(y_test,forest_pred , average=None)
forest_clf.score(x_train,y_train)

print ("Random forest Recal details : ", forest_recall_none)
print ("Random forest precition details : ", forest_precision_none)
print ( "Random forest F1 Score Details : " , forest_f1_none)

def sgd_builder():
    pip_sgd = Pipeline([("selector",SelectKBest(chi2)),("sgd_clf",SGDClassifier())])
    parameters_sgd ={ 'selector__k':[20],
                "sgd_clf__loss":["hinge","modified_huber","log"],
                "sgd_clf__max_iter":[200]}
    scorer_sgd = make_scorer(accuracy_score)
    searcher_sgd = GridSearchCV(pip_sgd, parameters_sgd, scoring=scorer_sgd)
    return searcher_sgd
sgd_clf=sgd_builder().fit(x_train,y_train)
score_sgd = sgd_clf.score(x_test,y_test)
sgd_pred = sgd_clf.predict(x_test)
sgd_cm = confusion_matrix(y_test,sgd_pred)
sgd_recall = recall_score(y_test,sgd_pred , average="micro")
sgd_precision = precision_score(y_test,sgd_pred , average="micro")
sgd_recall_none = recall_score(y_test,sgd_pred , average=None)
sgd_precision_none = precision_score(y_test,sgd_pred , average=None)
sgd_f1 = f1_score(y_test,sgd_pred , average="micro")
sgd_f1_none = f1_score(y_test,sgd_pred , average=None)
sgd_clf.score(x_train,y_train)

print ("Stochastic gradient classifier Recal details : ", sgd_recall_none)
print ("Stochastic gradient classifier precition details : ", sgd_precision_none)
print ( "Stochastic gradient classifier F1 Score Details : " , sgd_f1_none)


def knn_builder():
    pip_knn = Pipeline([("selector",SelectKBest(chi2)),("knn_clf",KNeighborsClassifier())])
    parameters_knn ={'selector__k':[20],
                "knn_clf__n_neighbors":[1]}
    scorer_knn = make_scorer(accuracy_score)
    searcher_knn = GridSearchCV(pip_knn, parameters_knn, scoring=scorer_knn)
    return searcher_knn
knn_clf=knn_builder().fit(x_train,y_train)
score_knn = knn_clf.score(x_test,y_test)
knn_pred = knn_clf.predict(x_test)
knn_cm = confusion_matrix(y_test,knn_pred)
knn_recall = recall_score(y_test,knn_pred , average="micro")
knn_precision = precision_score(y_test,knn_pred , average="micro")
knn_recall_none = recall_score(y_test,knn_pred , average=None)
knn_precision_none = precision_score(y_test,knn_pred , average=None)
knn_f1 = f1_score(y_test,knn_pred , average="micro")
knn_f1_none = f1_score(y_test,knn_pred , average=None)
knn_clf.score(x_train,y_train)


print ("K Neighbors Classifier Recal details : ", knn_recall_none)
print ("K Neighbors Classifier precition details : ", knn_precision_none)
print ( "K Neighbors Classifier F1 Score Details : " , knn_f1_none)


def ada_builder():
    pip_ada = Pipeline([("selector",SelectKBest(chi2)),("ada_clf",AdaBoostClassifier())])
    parameters_ada ={ 'selector__k':[20]}
    scorer_ada = make_scorer(accuracy_score)
    searcher_ada = GridSearchCV(pip_ada, parameters_ada, scoring=scorer_ada)
    return searcher_ada
ada_clf=ada_builder().fit(x_train,y_train)
score_ada = ada_clf.score(x_test,y_test)
ada_pred = ada_clf.predict(x_test)
ada_cm = confusion_matrix(y_test,ada_pred)
ada_recall = recall_score(y_test,ada_pred , average="micro")
ada_precision = precision_score(y_test,ada_pred , average="micro")
ada_recall_none = recall_score(y_test,ada_pred , average=None)
ada_precision_none = precision_score(y_test,ada_pred , average=None)
ada_f1 = f1_score(y_test,ada_pred , average="micro")
ada_f1_none = f1_score(y_test,ada_pred , average=None)
ada_clf.score(x_train,y_train)


print ("AdaBoostClassifier Recal details : ", ada_recall_none)
print ("AdaBoostClassifier precition details : ", ada_precision_none)
print ( "AdaBoostClassifier F1 Score Details : " , ada_f1_none)


def gua_builder():
    pip_gua = Pipeline([("selector",SelectKBest(chi2)),("gau_clf",GaussianNB())])
    parameters_gua ={ 'selector__k':[20]
                }
    scorer_gua = make_scorer(accuracy_score)
    searcher_gua = GridSearchCV(pip_gua, parameters_gua, scoring=scorer_gua)
    return searcher_gua
gua_clf=gua_builder().fit(x_train,y_train)
score_gua = gua_clf.score(x_test,y_test)
gua_pred = gua_clf.predict(x_test)
gua_cm = confusion_matrix(y_test,gua_pred)
gua_recall = recall_score(y_test,gua_pred , average="micro")
gua_precision = precision_score(y_test,gua_pred , average="micro")
gua_recall_none = recall_score(y_test,gua_pred , average=None)
gua_precision_none = precision_score(y_test,gua_pred , average=None)
gua_f1 = f1_score(y_test,gua_pred , average="micro")
gua_f1_none = f1_score(y_test,gua_pred , average=None)
gua_clf.score(x_train,y_train)


print ("GaussianNBClassifier Recal details : ", gua_recall_none)
print ("GaussianNBClassifier precition details : ", gua_precision_none)
print ( "GaussianNBClassifier F1 Score Details : " , gua_f1_none)


def qua_builder():
    pip_qua = Pipeline([("selector",SelectKBest(chi2)),("qua_clf",QuadraticDiscriminantAnalysis())])
    parameters_qua ={ 'selector__k':[20]
                }
    scorer_qua = make_scorer(accuracy_score)
    searcher_qua = GridSearchCV(pip_qua, parameters_qua, scoring=scorer_qua)
    return searcher_qua
qua_clf=gua_builder().fit(x_train,y_train)
score_qua = qua_clf.score(x_test,y_test)
qua_pred = qua_clf.predict(x_test)
qua_cm = confusion_matrix(y_test,qua_pred)
qua_recall = recall_score(y_test,qua_pred , average="micro")
qua_precision = precision_score(y_test,qua_pred , average="micro")
qua_recall_none = recall_score(y_test,qua_pred , average=None)
qua_precision_none = precision_score(y_test,qua_pred , average=None)
qua_f1 = f1_score(y_test,qua_pred , average="micro")
qua_f1_none = f1_score(y_test,qua_pred , average=None)
qua_clf.score(x_train,y_train)


print ("QuadraticDiscriminantAnalysis Recal details : ", qua_recall_none)
print ("QuadraticDiscriminantAnalysis precition details : ", qua_precision_none)
print ( "QuadraticDiscriminantAnalysis F1 Score Details : " , qua_f1_none)

# unsupervised learning k means clustring
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=5, 
                              max_iter=1000, tol=0.001, precompute_distances='auto', 
                              verbose=1, random_state=None,
                              copy_x=True, n_jobs=None, algorithm='auto').fit(x_train)
output_unspervised_k_means = kmeans.labels_
y_pred_kmeans = kmeans.predict(x_test)
from sklearn import metrics
y_array=list(np.array(y_test).reshape(477))
kmeansscore =metrics.adjusted_rand_score(y_array, y_pred_kmeans)
metrics.adjusted_mutual_info_score(y_array, y_pred_kmeans)
metrics.homogeneity_score(y_array, y_pred_kmeans)

print ("k means score : " ,kmeansscore )
# classification neural network
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.activations import sigmoid

y_binary_tr = to_categorical(y)
x_train_,x_test_,y_train_,y_test_ = train_test_split(x,y_binary_tr,test_size=.2,stratify=y_binary_tr,random_state=3)

(x_train_, x_valid) = x_train_[150:], x_train_[:150]
(y_train_, y_valid) = y_train_[150:], y_train_[:150]
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=25))
model.add(Dense(100, activation=sigmoid))
model.add(Dense(5, activation='softmax'))
model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                                save_best_only=True)


hist = model.fit(x_train_, y_train_, batch_size=40, epochs=300,
          validation_data=(x_valid, y_valid),
          verbose=2, shuffle=True)  

score = model.evaluate(x_test_, y_test_, verbose=0)
print("Accuracy: ", score[1])
'''

# models comparison
comparison = {"SVM": [score_svm,svm_cm,svm_recall,svm_recall_none,svm_precision,svm_precision_none,svm_f1,svm_f1_none] , 
              "Decision Tree":[score_tree,tree_cm,tree_recall,tree_recall_none,tree_precision,tree_precision_none,tree_f1,tree_f1_none] , 
              "Logistic regression":[score_log,log_cm,log_recall,log_recall_none,log_precision,log_precision_none,log_f1,log_f1_none] , 
              "Random forest":[score_forest,forest_cm,forest_recall,forest_recall_none,forest_precision,forest_precision_none,forest_f1,forest_f1_none] , 
              "SGDClassifier":[score_sgd,sgd_cm,sgd_recall,sgd_recall_none,sgd_precision,sgd_precision_none,sgd_f1,sgd_f1_none] , 
              "KNeighborsClassifier":[score_knn,knn_cm,knn_recall,knn_recall_none,knn_precision,knn_precision_none,knn_f1,knn_f1_none] ,
              "AdaBoostClassifier":[score_ada,ada_cm,ada_recall,ada_recall_none,ada_precision,ada_precision_none,ada_f1,ada_f1_none] , 
              "GaussianNB":[score_gua,gua_cm,gua_recall,gua_recall_none,gua_precision,gua_precision_none,gua_f1,gua_f1_none] , 
              "QuadraticDiscriminantAnalysis": [score_qua,qua_cm,qua_recall,qua_recall_none,qua_precision,qua_precision_none,qua_f1,qua_f1_none]}
comaprison_frame = pd.DataFrame(comparison)

comparison_simple = {"SVM": [score_svm,svm_recall,svm_precision,svm_f1] , 
              "Decision Tree":[score_tree,tree_recall,tree_precision,tree_f1] , 
              "Logistic regression":[score_log,log_recall,log_precision,log_f1] , 
              "Random forest":[score_forest,forest_recall,forest_precision,forest_f1] , 
              "SGDClassifier":[score_sgd,sgd_recall,sgd_precision,sgd_f1] , 
              "KNeighborsClassifier":[score_knn,knn_recall,knn_precision,knn_f1] ,
              "AdaBoostClassifier":[score_ada,ada_recall,ada_precision,ada_f1] , 
              "GaussianNB":[score_gua,gua_recall,gua_precision,gua_f1] , 
              "QuadraticDiscriminantAnalysis": [score_qua,qua_recall,qua_precision,qua_f1]}
comaprison_frame_simple = pd.DataFrame(comparison_simple , index = ["Accuracy","Recall score","Precision score","F1-score"])

print(comaprison_frame_simple)
comaprison_frame_simple.to_csv(r'comparison_simple.csv')




clfs =[svm_clf,tree_clf,log_clf,forest_clf,sgd_clf,knn_clf,ada_clf,gua_clf,qua_clf]
def getList(clf_dict): 
    return clf_dict.keys() 
clfs_names = getList(comparison)


scoring = 'accuracy'
n_folds = 7

results, names  = [], [] 
validation_results = []
for name, clf_model  in zip(clfs_names,clfs):
    kfold = KFold(n_splits=n_folds, random_state=3)
    cv_results = cross_val_score(clf_model, x, y, cv= 5, scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    validation_results.append(msg)
    print(msg)

print("validation result for classifiers : ")
print(validation_results)
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.savefig('%s models comparison .png'%i)

plt.show()

from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[
         ('Decision Tree', tree_clf)
        ,('Random forest', forest_clf),('KNeighborsClassifier', knn_clf , "kmeans",kmeans),
        ])
eclf1.fit(x_train, y_train)
eclf1_score = eclf1.score(x_test,y_test)
print ("compining the strongest classifiers score : ",eclf1_score)



from sklearn.linear_model import LogisticRegression
formula_reg = LogisticRegression().fit(x,y)
decision =formula_reg.coef_
colls = x.columns
decision = pd.DataFrame(decision , columns = colls , index = ["cat1", "cat2" , "cat3" , "cat4" , "cat 5"])

##### with pca

de = PCA()
de.fit(x,y=y)
de_x=de.transform(x)
#splitting the data into train and test with test data size of 20% 
x_train,x_test,y_train,y_test = train_test_split(de_x,y,test_size=.2,stratify=y,random_state=3)

#Using svm model 

def svm_builder():
    pip_svm = Pipeline([("selector",SelectKBest(chi2)),("svm_clf",SVC())])
    parameters_svm ={ 'selector__k':[20],
                "svm_clf__C":[10],"svm_clf__kernel":[ 'poly', 'rbf', 'sigmoid' ],
                "svm_clf__degree":[7],"svm_clf__random_state":[7]}
    scorer_svm = make_scorer(accuracy_score)
    searcher_svm = GridSearchCV(pip_svm, parameters_svm, scoring=scorer_svm)
    return searcher_svm
svm_clf=svm_builder().fit(x_train,y_train)
score_svm = svm_clf.score(x_test,y_test)
svm_pred = svm_clf.predict(x_test)
svm_cm = confusion_matrix(y_test,svm_pred)
svm_recall = recall_score(y_test,svm_pred , average="micro")
svm_precision = precision_score(y_test,svm_pred , average="micro")
svm_recall_none = recall_score(y_test,svm_pred , average=None)
svm_precision_none = precision_score(y_test,svm_pred , average=None)
svm_f1 = f1_score(y_test,svm_pred , average="micro")
svm_f1_none = f1_score(y_test,svm_pred , average=None)
svm_clf.score(x_train,y_train)

print ("SVM Recal details : ", svm_recall_none)
print ("SVM precition details : ", svm_precision_none)
print ( "SVM F1 Score Details : " , svm_f1_none)


#Using decision tree model 
def tree_builder():
    pip_tree = Pipeline([("selector",SelectKBest(chi2)),("tree_clf",tree.DecisionTreeClassifier())])
    parameters_tree ={ 'selector__k':[20],
                     "tree_clf__random_state":[7],"tree_clf__min_samples_split":[13],
                     "tree_clf__max_depth":[12]}
    scorer_tree = make_scorer(accuracy_score)
    searcher_tree = GridSearchCV(pip_tree, parameters_tree, scoring=scorer_tree)
    return searcher_tree
tree_clf=tree_builder().fit(x_train,y_train)
score_tree = tree_clf.score(x_test,y_test)
tree_pred = tree_clf.predict(x_test)
tree_cm = confusion_matrix(y_test,tree_pred)
tree_recall = recall_score(y_test,tree_pred , average="micro")
tree_precision = precision_score(y_test,tree_pred , average="micro")
tree_recall_none = recall_score(y_test,tree_pred , average=None)
tree_precision_none = precision_score(y_test,tree_pred , average=None)
tree_f1 = f1_score(y_test,tree_pred , average="micro")
tree_f1_none = f1_score(y_test,tree_pred , average=None)
tree_clf.score(x_train,y_train)

print ("decision tree Recal details : ", tree_recall_none)
print ("decision tree precition details : ", tree_precision_none)
print ( "decision tree F1 Score Details : " , tree_f1_none)
#Using logistic regression model 
def log_builder():
    pip_log = Pipeline([("selector",SelectKBest(chi2)),("log_clf",LogisticRegressionCV())])
    parameters_tree ={ 'selector__k':[20],
                "log_clf__cv":[5],"log_clf__random_state":[7]}
    scorer_log = make_scorer(accuracy_score)
    searcher_log = GridSearchCV(pip_log, parameters_tree, scoring=scorer_log)
    return searcher_log
log_clf=log_builder().fit(x_train,y_train)
score_log = log_clf.score(x_test,y_test)
log_pred = log_clf.predict(x_test)
log_cm = confusion_matrix(y_test,log_pred)
log_recall = recall_score(y_test,log_pred , average="micro")
log_precision = precision_score(y_test,log_pred , average="micro")
log_recall_none = recall_score(y_test,log_pred , average=None)
log_precision_none = precision_score(y_test,log_pred , average=None)
log_f1 = f1_score(y_test,log_pred , average="micro")
log_f1_none = f1_score(y_test,log_pred , average=None)
log_clf.score(x_train,y_train)

print ("logistic regression Recal details : ", log_recall_none)
print ("logistic regression precition details : ", log_precision_none)
print ( "logistic regression F1 Score Details : " , log_f1_none)

#Using random forest model 
def forest_builder():
    pip_forest = Pipeline([("selector",SelectKBest(chi2)),("forest_clf",RandomForestClassifier())])
    parameters_forest ={'selector__k':[20],"forest_clf__random_state":[5],
                "forest_clf__n_estimators":[200]}
    scorer_forest = make_scorer(accuracy_score)
    searcher_forest = GridSearchCV(pip_forest, parameters_forest, scoring=scorer_forest)
    return searcher_forest
forest_clf=forest_builder().fit(x_train,y_train)
score_forest = forest_clf.score(x_test,y_test)
forest_pred = forest_clf.predict(x_test)
forest_cm = confusion_matrix(y_test,forest_pred)
forest_recall = recall_score(y_test,forest_pred , average="micro")
forest_precision = precision_score(y_test,forest_pred , average="micro")
forest_recall_none = recall_score(y_test,forest_pred , average=None)
forest_precision_none = precision_score(y_test,forest_pred , average=None)
forest_f1 = f1_score(y_test,forest_pred , average="micro")
forest_f1_none = f1_score(y_test,forest_pred , average=None)
forest_clf.score(x_train,y_train)

print ("Random forest Recal details : ", forest_recall_none)
print ("Random forest precition details : ", forest_precision_none)
print ( "Random forest F1 Score Details : " , forest_f1_none)

def sgd_builder():
    pip_sgd = Pipeline([("selector",SelectKBest(chi2)),("sgd_clf",SGDClassifier())])
    parameters_sgd ={ 'selector__k':[20],
                "sgd_clf__loss":["hinge","modified_huber","log"],
                "sgd_clf__max_iter":[200]}
    scorer_sgd = make_scorer(accuracy_score)
    searcher_sgd = GridSearchCV(pip_sgd, parameters_sgd, scoring=scorer_sgd)
    return searcher_sgd
sgd_clf=sgd_builder().fit(x_train,y_train)
score_sgd = sgd_clf.score(x_test,y_test)
sgd_pred = sgd_clf.predict(x_test)
sgd_cm = confusion_matrix(y_test,sgd_pred)
sgd_recall = recall_score(y_test,sgd_pred , average="micro")
sgd_precision = precision_score(y_test,sgd_pred , average="micro")
sgd_recall_none = recall_score(y_test,sgd_pred , average=None)
sgd_precision_none = precision_score(y_test,sgd_pred , average=None)
sgd_f1 = f1_score(y_test,sgd_pred , average="micro")
sgd_f1_none = f1_score(y_test,sgd_pred , average=None)
sgd_clf.score(x_train,y_train)

print ("Stochastic gradient classifier Recal details : ", sgd_recall_none)
print ("Stochastic gradient classifier precition details : ", sgd_precision_none)
print ( "Stochastic gradient classifier F1 Score Details : " , sgd_f1_none)


def knn_builder():
    pip_knn = Pipeline([("selector",SelectKBest(chi2)),("knn_clf",KNeighborsClassifier())])
    parameters_knn ={'selector__k':[20],
                "knn_clf__n_neighbors":[1]}
    scorer_knn = make_scorer(accuracy_score)
    searcher_knn = GridSearchCV(pip_knn, parameters_knn, scoring=scorer_knn)
    return searcher_knn
knn_clf=knn_builder().fit(x_train,y_train)
score_knn = knn_clf.score(x_test,y_test)
knn_pred = knn_clf.predict(x_test)
knn_cm = confusion_matrix(y_test,knn_pred)
knn_recall = recall_score(y_test,knn_pred , average="micro")
knn_precision = precision_score(y_test,knn_pred , average="micro")
knn_recall_none = recall_score(y_test,knn_pred , average=None)
knn_precision_none = precision_score(y_test,knn_pred , average=None)
knn_f1 = f1_score(y_test,knn_pred , average="micro")
knn_f1_none = f1_score(y_test,knn_pred , average=None)
knn_clf.score(x_train,y_train)


print ("K Neighbors Classifier Recal details : ", knn_recall_none)
print ("K Neighbors Classifier precition details : ", knn_precision_none)
print ( "K Neighbors Classifier F1 Score Details : " , knn_f1_none)


def ada_builder():
    pip_ada = Pipeline([("selector",SelectKBest(chi2)),("ada_clf",AdaBoostClassifier())])
    parameters_ada ={ 'selector__k':[20]}
    scorer_ada = make_scorer(accuracy_score)
    searcher_ada = GridSearchCV(pip_ada, parameters_ada, scoring=scorer_ada)
    return searcher_ada
ada_clf=ada_builder().fit(x_train,y_train)
score_ada = ada_clf.score(x_test,y_test)
ada_pred = ada_clf.predict(x_test)
ada_cm = confusion_matrix(y_test,ada_pred)
ada_recall = recall_score(y_test,ada_pred , average="micro")
ada_precision = precision_score(y_test,ada_pred , average="micro")
ada_recall_none = recall_score(y_test,ada_pred , average=None)
ada_precision_none = precision_score(y_test,ada_pred , average=None)
ada_f1 = f1_score(y_test,ada_pred , average="micro")
ada_f1_none = f1_score(y_test,ada_pred , average=None)
ada_clf.score(x_train,y_train)


print ("AdaBoostClassifier Recal details : ", ada_recall_none)
print ("AdaBoostClassifier precition details : ", ada_precision_none)
print ( "AdaBoostClassifier F1 Score Details : " , ada_f1_none)


def gua_builder():
    pip_gua = Pipeline([("selector",SelectKBest(chi2)),("gau_clf",GaussianNB())])
    parameters_gua ={ 'selector__k':[20]
                }
    scorer_gua = make_scorer(accuracy_score)
    searcher_gua = GridSearchCV(pip_gua, parameters_gua, scoring=scorer_gua)
    return searcher_gua
gua_clf=gua_builder().fit(x_train,y_train)
score_gua = gua_clf.score(x_test,y_test)
gua_pred = gua_clf.predict(x_test)
gua_cm = confusion_matrix(y_test,gua_pred)
gua_recall = recall_score(y_test,gua_pred , average="micro")
gua_precision = precision_score(y_test,gua_pred , average="micro")
gua_recall_none = recall_score(y_test,gua_pred , average=None)
gua_precision_none = precision_score(y_test,gua_pred , average=None)
gua_f1 = f1_score(y_test,gua_pred , average="micro")
gua_f1_none = f1_score(y_test,gua_pred , average=None)
gua_clf.score(x_train,y_train)


print ("GaussianNBClassifier Recal details : ", gua_recall_none)
print ("GaussianNBClassifier precition details : ", gua_precision_none)
print ( "GaussianNBClassifier F1 Score Details : " , gua_f1_none)


def qua_builder():
    pip_qua = Pipeline([("selector",SelectKBest(chi2)),("qua_clf",QuadraticDiscriminantAnalysis())])
    parameters_qua ={ 'selector__k':[20]
                }
    scorer_qua = make_scorer(accuracy_score)
    searcher_qua = GridSearchCV(pip_qua, parameters_qua, scoring=scorer_qua)
    return searcher_qua
qua_clf=gua_builder().fit(x_train,y_train)
score_qua = qua_clf.score(x_test,y_test)
qua_pred = qua_clf.predict(x_test)
qua_cm = confusion_matrix(y_test,qua_pred)
qua_recall = recall_score(y_test,qua_pred , average="micro")
qua_precision = precision_score(y_test,qua_pred , average="micro")
qua_recall_none = recall_score(y_test,qua_pred , average=None)
qua_precision_none = precision_score(y_test,qua_pred , average=None)
qua_f1 = f1_score(y_test,qua_pred , average="micro")
qua_f1_none = f1_score(y_test,qua_pred , average=None)
qua_clf.score(x_train,y_train)


print ("QuadraticDiscriminantAnalysis Recal details : ", qua_recall_none)
print ("QuadraticDiscriminantAnalysis precition details : ", qua_precision_none)
print ( "QuadraticDiscriminantAnalysis F1 Score Details : " , qua_f1_none)

# unsupervised learning k means clustring
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=5, 
                              max_iter=1000, tol=0.001, precompute_distances='auto', 
                              verbose=1, random_state=None,
                              copy_x=True, n_jobs=None, algorithm='auto').fit(x_train)
output_unspervised_k_means = kmeans.labels_
y_pred_kmeans = kmeans.predict(x_test)
from sklearn import metrics
y_array=list(np.array(y_test).reshape(477))
kmeansscore =metrics.adjusted_rand_score(y_array, y_pred_kmeans)
metrics.adjusted_mutual_info_score(y_array, y_pred_kmeans)
metrics.homogeneity_score(y_array, y_pred_kmeans)

print ("k means score : " ,kmeansscore )
# classification neural network
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.activations import sigmoid

y_binary_tr = to_categorical(y)
x_train_,x_test_,y_train_,y_test_ = train_test_split(x,y_binary_tr,test_size=.2,stratify=y_binary_tr,random_state=3)

(x_train_, x_valid) = x_train_[150:], x_train_[:150]
(y_train_, y_valid) = y_train_[150:], y_train_[:150]
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=25))
model.add(Dense(100, activation=sigmoid))
model.add(Dense(5, activation='softmax'))
model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                                save_best_only=True)


hist = model.fit(x_train_, y_train_, batch_size=40, epochs=300,
          validation_data=(x_valid, y_valid),
          verbose=2, shuffle=True)  

score = model.evaluate(x_test_, y_test_, verbose=0)
print("Accuracy: ", score[1])
'''
# models comparison
comparison = {"SVM": [score_svm,svm_cm,svm_recall,svm_recall_none,svm_precision,svm_precision_none,svm_f1,svm_f1_none] , 
              "Decision Tree":[score_tree,tree_cm,tree_recall,tree_recall_none,tree_precision,tree_precision_none,tree_f1,tree_f1_none] , 
              "Logistic regression":[score_log,log_cm,log_recall,log_recall_none,log_precision,log_precision_none,log_f1,log_f1_none] , 
              "Random forest":[score_forest,forest_cm,forest_recall,forest_recall_none,forest_precision,forest_precision_none,forest_f1,forest_f1_none] , 
              "SGDClassifier":[score_sgd,sgd_cm,sgd_recall,sgd_recall_none,sgd_precision,sgd_precision_none,sgd_f1,sgd_f1_none] , 
              "KNeighborsClassifier":[score_knn,knn_cm,knn_recall,knn_recall_none,knn_precision,knn_precision_none,knn_f1,knn_f1_none] ,
              "AdaBoostClassifier":[score_ada,ada_cm,ada_recall,ada_recall_none,ada_precision,ada_precision_none,ada_f1,ada_f1_none] , 
              "GaussianNB":[score_gua,gua_cm,gua_recall,gua_recall_none,gua_precision,gua_precision_none,gua_f1,gua_f1_none] , 
              "QuadraticDiscriminantAnalysis": [score_qua,qua_cm,qua_recall,qua_recall_none,qua_precision,qua_precision_none,qua_f1,qua_f1_none]}
comaprison_frame = pd.DataFrame(comparison)

comparison_simple = {"SVM": [score_svm,svm_recall,svm_precision,svm_f1] , 
              "Decision Tree":[score_tree,tree_recall,tree_precision,tree_f1] , 
              "Logistic regression":[score_log,log_recall,log_precision,log_f1] , 
              "Random forest":[score_forest,forest_recall,forest_precision,forest_f1] , 
              "SGDClassifier":[score_sgd,sgd_recall,sgd_precision,sgd_f1] , 
              "KNeighborsClassifier":[score_knn,knn_recall,knn_precision,knn_f1] ,
              "AdaBoostClassifier":[score_ada,ada_recall,ada_precision,ada_f1] , 
              "GaussianNB":[score_gua,gua_recall,gua_precision,gua_f1] , 
              "QuadraticDiscriminantAnalysis": [score_qua,qua_recall,qua_precision,qua_f1]}
comaprison_frame_simple = pd.DataFrame(comparison_simple , index = ["Accuracy","Recall score","Precision score","F1-score"])

print(comaprison_frame_simple)
comaprison_frame_simple.to_csv(r'pca comparison_simple.csv')



clfs =[svm_clf,tree_clf,log_clf,forest_clf,sgd_clf,knn_clf,ada_clf,gua_clf,qua_clf]
def getList(clf_dict): 
    return clf_dict.keys() 
clfs_names = getList(comparison)


scoring = 'accuracy'
n_folds = 7

results, names  = [], [] 
validation_results = []
for name, clf_model  in zip(clfs_names,clfs):
    kfold = KFold(n_splits=n_folds, random_state=3)
    cv_results = cross_val_score(clf_model, x, y, cv= 5, scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    validation_results.append(msg)
    print(msg)

print("validation result for classifiers : ")
print(validation_results)
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()
plt.savefig('%s pca models comparison .png'%i)


from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[
         ('Decision Tree', tree_clf)
        ,('Random forest', forest_clf),('KNeighborsClassifier', knn_clf , "kmeans",kmeans),
        ])
eclf1.fit(x_train, y_train)
eclf1_score = eclf1.score(x_test,y_test)
print ("compining the strongest classifiers score : ",eclf1_score)



        
        


        
