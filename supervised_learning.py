__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Johnson, Will",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"


# Modular function to apply decision tree classifier
def DT_classifier(X, Y, numFold):
    
    # Intilization of the figure
    myFig = plt.figure(figsize=[12,10])
    
    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=numFold,random_state=None, shuffle=False)
    
    # Initialization of the decision tree classifier
    classifier = tree.DecisionTreeClassifier()
    
    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 1
    for train, test in cv.split(X, Y):
        
        # Spliting the dataset
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        
        # Fitting the classifier into training set
        classifier = classifier.fit(X_train, Y_train)
        
        # Breakdown of statistical measure based on classes
        Y_pred = classifier.predict(X_test)
        print(classification_report(Y_test, Y_pred, digits=4))
        
        # Compute the model's performance
        acc_scores.append(accuracy_score(Y_test, Y_pred))
        if(len(np.unique(Y)) > 2):
            f1_scores_temp = []
            f1_scores_temp.append(f1_score(Y_test, Y_pred, average=None))
            f1_scores.append(np.mean(f1_scores_temp))
            del f1_scores_temp
            
            precision_scores_temp = []
            precision_scores_temp.append(precision_score(Y_test, Y_pred, average=None))
            precision_scores.append(np.mean(precision_scores_temp))
            del precision_scores_temp
            
            recall_scores_temp = []
            recall_scores_temp.append(recall_score(Y_test, Y_pred, average=None))
            recall_scores.append(np.mean(recall_scores_temp))
            del recall_scores_temp
        else:
            f1_scores.append(f1_score(Y_test, Y_pred, average='binary'))
            precision_scores.append(precision_score(Y_test, Y_pred, average='binary'))
            recall_scores.append(recall_score(Y_test, Y_pred, average='binary'))
        
        if(len(np.unique(Y)) == 2):
            probas_ = classifier.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
        
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, color='black', alpha=0.5,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
            print("Iteration ongoing inside DT method - KFold step: ", i)
            i += 1
        
    if(len(np.unique(Y)) == 2):
        plt.plot([0,1],[0,1],linestyle = '--',lw = 1, alpha=0.5, color = 'black')
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='black',
                 label=r'Mean ROC (AUC = %0.3f)' % (mean_auc),
                 lw=2, alpha=0.8)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nDecision Tree', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        
        fileName = 'Decision_Tree_ROC_' + str(numFold) + '_Fold.eps'
        # Saving the figure
        myFig.savefig(fileName, format='eps', dpi=1200)
    
    # Statistical measurement of the model
    print("Accuracy: ", np.mean(acc_scores))
    print("Precision: ", np.mean(precision_scores))
    print("Recall: ", np.mean(recall_scores))
    print("F1: ", np.mean(f1_scores))
    if(len(np.unique(Y)) == 2):
        print(acc_scores)
        print(precision_scores)
        print(recall_scores)
        print(f1_scores)

        
# Modular function to apply artificial neural network 
def ANN_classifier(X, Y, batchSize, epochCount):
    
    myFig = plt.figure(figsize=[12,10])

    # Spliting the dataset into the Training and Test Set
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)
    
    # Initializing the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = round(X.shape[1]/2), init =  'uniform', activation = 'relu', input_dim = X.shape[1]))
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = round(X.shape[1]/2), init =  'uniform', activation = 'relu'))
    
    # Add a dropout layer
    #classifier.add(Dropout(0.4))
    
    if(len(np.unique(Y)) > 2): # Multi-classification task
        # Adding the output layer
        classifier.add(Dense(output_dim = len(np.unique(Y)), init =  'uniform', activation = 'softmax'))
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else: # Binary classification task
        # Adding the output layer
        classifier.add(Dense(output_dim = 1, init =  'uniform', activation = 'sigmoid'))
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Callback to stop if validation loss does not decrease
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    # Fitting the ANN to the Training set
    history = classifier.fit(X_train,
                   Y_train,
                   callbacks=callbacks,
                   validation_split=0.1,
                   batch_size = batchSize,
                   epochs = epochCount,
                   shuffle=True)
    
    print(history.history)
    print(classifier.summary())
    
    # ------ Evaluation -------

    print("Artificial Neural Network")
        
    # Predicting the Test set results
    Y_pred = classifier.predict_classes(X_test)
    Y_pred = (Y_pred > 0.5)
    
    # Breakdown of statistical measure based on classes
    print(classification_report(Y_test, Y_pred, digits=4))
    
    # Compute the model's performance

    # Making the cufusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:\n", cm)
    print("Accuracy: ", accuracy_score(Y_test, Y_pred))
    
    if(len(np.unique(Y))) == 2:
        print("F1: ", f1_score(Y_test, Y_pred, average='binary'))
        print("Precison: ", precision_score(Y_test, Y_pred, average='binary'))
        print("Recall: ", recall_score(Y_test, Y_pred, average='binary'))
    else:
        f1_scores = f1_score(Y_test, Y_pred, average=None)
        print("F1: ", np.mean(f1_scores))
        print("F1: ", np.mean(f1_scores))
        precision_scores = precision_score(Y_test, Y_pred, average=None)
        print("Precison: ", np.mean(precision_scores))
        recall_scores = recall_score(Y_test, Y_pred, average=None)
        print("Recall: ", np.mean(recall_scores))
    
    # ------------ Print Accuracy over Epoch --------------------

    plt.plot(history.history['accuracy'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
    plt.plot(history.history['val_accuracy'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
    plt.title('Accuracy over Epoch\nArtificial Neural Network', fontsize=20, weight='bold')
    plt.ylabel('Accuracy', fontsize=18, weight='bold')
    plt.xlabel('Epoch', fontsize=18, weight='bold')
    plt.legend(['Train', 'Validation'], loc='lower right', fontsize=14)
    plt.xticks(ticks=range(0, len(history.history['accuracy'])))
    
    plt.yticks(fontsize=16)
    plt.show()
        
    if(len(np.unique(Y))) == 2:
        fileName = 'ANN_Accuracy_over_Epoch_Binary_Classification.eps'
    else:
        fileName = 'ANN_Accuracy_over_Epoch_Multiclass_Classification.eps'
    
    # Saving the figure
    myFig.savefig(fileName, format='eps', dpi=1200)
    
    # ------------ Print Loss over Epoch --------------------

    # Clear figure
    plt.clf()
    myFig = plt.figure(figsize=[12,10])
    
    plt.plot(history.history['loss'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
    plt.plot(history.history['val_loss'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
    plt.title('Loss over Epoch\nArtificial Neural Network', fontsize=20, weight='bold')
    plt.ylabel('Loss', fontsize=18, weight='bold')
    plt.xlabel('Epoch', fontsize=18, weight='bold')
    plt.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
    plt.xticks(ticks=range(0, len(history.history['loss'])))
    
    plt.yticks(fontsize=16)
    plt.show()
        
    if(len(np.unique(Y))) == 2:
        fileName = 'ANN_Loss_over_Epoch_Binary_Classification.eps'
    else:
        fileName = 'ANN_Loss_over_Epoch_Multiclass_Classification.eps'
    
    # Saving the figure
    myFig.savefig(fileName, format='eps', dpi=1200)
    
    
    # ------------ ROC Curve --------------------

    # Clear figure
    plt.clf()
    myFig = plt.figure(figsize=[12,10])
    
    if len(np.unique(Y)) == 2:
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        plt.plot(fpr, tpr, color='black',
                label=r'ROC (AUC = %0.3f)' % (auc(fpr, tpr)),
                lw=2, alpha=0.8)
            
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nArtificial Neural Network', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
            
        fileName = 'ANN_Binary_Classification_ROC.eps'
        # Saving the figure
        myFig.savefig(fileName, format='eps', dpi=1200)

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Libraries relevant to performance metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.preprocessing import MinMaxScaler

# Libraries relevant to supervised learning 
from sklearn import tree

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.callbacks import EarlyStopping

#importing the data set
from scipy.io import arff
data = arff.loadarff('../TRAbID/probe_known_attacks.arff')
dataset = pd.DataFrame(data[0])
print(dataset.head())

# Some manual processing on the dataframe
dataset = dataset.dropna()
dataset = dataset.drop(['Flow_ID', '_Source_IP', '_Destination_IP', '_Timestamp'], axis = 1)
dataset['Flow_Bytes/s'] = dataset['Flow_Bytes/s'].astype(float)
dataset['_Flow_Packets/s'] = dataset['_Flow_Packets/s'].astype(float)

dataset_sample = dataset
#dataset_sample = dataset_sample.truncate(before=0, after=0)

#dataset_sample.loc[dataset.index[1]] = dataset.iloc[5475]

max_count = 1501
iBenign = 20000
iDDoS = 0
iBot = 0
iDoS_slowloris = 0
iDoS_Slowhttptest = 0
iDoS_Hulk = 0
iDoS_GoldenEye = 0
iHeartbleed = 0
iFTP_Patator = 0
iSSH_Patator = 0
iWeb_Attack_BF = 0
iWeb_Attack_XSS = 0
iWeb_Attack_SQL = 0
iInfiltration = 0
iIndex = 0

for i in range(0,948995):
    try:
        if(dataset.iloc[[i]]['Label'].values == 'BENIGN'):
            iBenign = iBenign - 1
            if(iBenign >= 0):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'DDoS'):
            iDDoS = iDDoS + 1
            if(iDDoS < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'Bot'):
            iBot = iBot + 1
            if(iBot < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'DoS slowloris'):
            iDoS_slowloris = iDoS_slowloris + 1
            if(iDoS_slowloris < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'DoS Slowhttptest'):
            iDoS_Slowhttptest = iDoS_Slowhttptest + 1
            if(iDoS_Slowhttptest < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'DoS Hulk'):
            iDoS_Hulk = iDoS_Hulk + 1
            if(iDoS_Hulk < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'DoS GoldenEye'):
            iDoS_GoldenEye = iDoS_GoldenEye + 1
            if(iDoS_GoldenEye < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'Heartbleed'):
            iHeartbleed = iHeartbleed + 1
            if(iHeartbleed < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'FTP-Patator'):
            iFTP_Patator = iFTP_Patator + 1
            if(iFTP_Patator < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'SSH-Patator'):
            iSSH_Patator = iSSH_Patator + 1
            if(iSSH_Patator < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1  
                
        elif(dataset.iloc[[i]]['Label'].values == 'Web Attack ñ Brute Force'):
            iWeb_Attack_BF = iWeb_Attack_BF + 1
            if(iWeb_Attack_BF < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'Web Attack ñ XSS'):
            iWeb_Attack_XSS = iWeb_Attack_XSS + 1
            if(iWeb_Attack_XSS < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'Web Attack ñ Sql Injection'):
            iWeb_Attack_SQL = iWeb_Attack_SQL + 1
            if(iWeb_Attack_SQL < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        elif(dataset.iloc[[i]]['Label'].values == 'Infiltration'):
            iInfiltration = iInfiltration + 1
            if(iInfiltration < max_count):
                #dataset = dataset.drop([i], axis=0)
                dataset_sample.loc[dataset.index[iIndex]] = dataset.iloc[i]
                iIndex = iIndex + 1
                
        else:
            continue
        
        print(i)

    except:
        print("Exception")
        continue

dataset_sample.to_csv('sample_dataset.csv', index = None, header=True)

# Creating X and Y from the dataset
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y_class)
Y_class = le.transform(Y_class)
print(list(le.classes_))
print(np.unique(Y_attack))
Y_class = dataset.iloc[:,-1].values
Y_class = Y_class.astype(str)
dataset['class'] = dataset['class'].astype(str)
X = dataset.iloc[:,0:43].values
X = X.astype(int)

# Performing scale data
scaler = MinMaxScaler ().fit(X)
X_scaled = np.array(scaler.transform(X))
    
# 5-fold cross validation
DT_classifier(X_scaled, Y_class, 5)