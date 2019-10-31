__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Johnson, Will",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"


# Generate a multilayer perceptron  model or ANN
def mlp_model(X, Y):
    
    # Initializing the ANN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(output_dim = round(X.shape[1]/2), init =  'uniform', activation = 'relu', input_dim = X.shape[1]))
    
    # Adding the second hidden layer
    model.add(Dense(output_dim = round(X.shape[1]/2), init =  'uniform', activation = 'relu'))

    
    if(len(np.unique(Y)) > 2): # Multi-classification task
        # Adding the output layer
        model.add(Dense(output_dim = len(np.unique(Y)), init =  'uniform', activation = 'softmax'))
        # Compiling the ANN
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else: # Binary classification task
        # Adding the output layer
        model.add(Dense(output_dim = 1, init =  'uniform', activation = 'sigmoid'))
        # Compiling the ANN
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    print(model.summary())
    
    return model


# Train the multilayer perceptron  model or ANN
def mlp_model_train(X, Y, val_split, batch_size, epochs_count):
    # Callback to stop if validation loss does not decrease
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    # Fitting the ANN to the Training set
    history = model.fit(X, Y,
                   callbacks=callbacks,
                   validation_split=val_split,
                   batch_size = batch_size,
                   epochs = epochs_count,
                   shuffle=True)

    print(history.history)
    print(model.summary())
    return history


# Evaluate the multilayer perceptron  model or ANN during test time
def mlp_model_eval(X, Y, history, flag):
    # Predicting the results given instances X
    Y_pred = model.predict_classes(X)
    Y_pred = (Y_pred > 0.5)

    # Breakdown of statistical measure based on classes
    print(classification_report(Y, Y_pred, digits=4))

    # Making the cufusion Matrix
    cm = confusion_matrix(Y, Y_pred)
    print("Confusion Matrix:\n", cm)
    print("Accuracy: ", accuracy_score(Y, Y_pred))

    if(len(np.unique(Y))) == 2:
        print("F1: ", f1_score(Y, Y_pred, average='binary'))
        print("Precison: ", precision_score(Y, Y_pred, average='binary'))
        print("Recall: ", recall_score(Y, Y_pred, average='binary'))
    else:
        f1_scores = f1_score(Y, Y_pred, average=None)
        print("F1: ", np.mean(f1_scores))
        precision_scores = precision_score(Y, Y_pred, average=None)
        print("Precison: ", np.mean(precision_scores))
        recall_scores = recall_score(Y, Y_pred, average=None)
        print("Recall: ", np.mean(recall_scores))

    # ------------ Print Accuracy over Epoch --------------------

    # Intilization of the figure
    myFig = plt.figure(figsize=[12,10])

    plt.plot(history.history['acc'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
    plt.plot(history.history['val_acc'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
    plt.title('Accuracy over Epoch', fontsize=20, weight='bold')
    plt.ylabel('Accuracy', fontsize=18, weight='bold')
    plt.xlabel('Epoch', fontsize=18, weight='bold')
    plt.legend(['Train', 'Validation'], loc='lower right', fontsize=14)
    plt.xticks(ticks=range(0, len(history.history['acc'])))
    
    plt.yticks(fontsize=16)
    plt.show()
        
    if(len(np.unique(Y))) == 2:
        if(flag == 1): #Regular
            fileName = 'ANN_Accuracy_over_Epoch_Binary_Classification_TRAbIDRegular.eps'
        else: #Adversarial
            fileName = 'ANN_Accuracy_over_Epoch_Binary_Classification_TRAbID_Adversarial.eps'
    else:
        if(flag == 1): #Regular
            fileName = 'ANN_Accuracy_over_Epoch_Multiclass_Classification_TRAbID_Regular.eps'
        else: #Adversarial
            fileName = 'ANN_Accuracy_over_Epoch_Multiclass_Classification_TRAbID_Adversarial.eps'
    
    # Saving the figure
    myFig.savefig(fileName, format='eps', dpi=1200)
    
    # ------------ Print Loss over Epoch --------------------

    # Clear figure
    plt.clf()
    myFig = plt.figure(figsize=[12,10])
    
    plt.plot(history.history['loss'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
    plt.plot(history.history['val_loss'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
    plt.title('Loss over Epoch', fontsize=20, weight='bold')
    plt.ylabel('Loss', fontsize=18, weight='bold')
    plt.xlabel('Epoch', fontsize=18, weight='bold')
    plt.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
    plt.xticks(ticks=range(0, len(history.history['loss'])))
    
    plt.yticks(fontsize=16)
    plt.show()
        
    if(len(np.unique(Y))) == 2:
        if(flag == 1): #Regular
            fileName = 'ANN_Loss_over_Epoch_Binary_Classification_TRAbID_Regular.eps'
        else: #Adversarial 
            fileName = 'ANN_Loss_over_Epoch_Binary_Classification_TRAbID_Adversarial.eps'
    else:
        if(flag == 1): #Regular
            fileName = 'ANN_Loss_over_Epoch_Multiclass_Classification_TRAbID_Regular.eps'
        else: #Adversarial
            fileName = 'ANN_Loss_over_Epoch_Multiclass_Classification_TRAbID_Adversarial.eps'
    
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
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
        
        if(flag == 1): #Regular
            fileName = 'ANN_Binary_Classification_ROC_TRAbID_Regular.eps'
        else: #Adversarial
            fileName = 'ANN_Binary_Classification_ROC_TRAbID_Adversarial.eps'

        # Saving the figure
        myFig.savefig(fileName, format='eps', dpi=1200)


# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# importing cleverhans - an adversarial example library
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks_tf import jacobian_graph
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.utils_tf import model_train, model_eval, batch_eval

# Libraries relevant to performance metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

#importing the data set
# ==== Data processing for CICIDS 2017 ====
'''dataset = pd.read_csv('../CICIDS2017/master.csv')
print(dataset.head())
print(dataset.shape)

# Some manual processing on the dataframe
dataset = dataset.dropna()
dataset = dataset.drop(['Flow_ID', '_Source_IP', '_Destination_IP', '_Timestamp'], axis = 1)
dataset['Flow_Bytes/s'] = dataset['Flow_Bytes/s'].astype(float)
dataset['_Flow_Packets/s'] = dataset['_Flow_Packets/s'].astype(float)

# Creating X and Y from the dataset
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(dataset['Label'])
Y_attack = le.transform(dataset['Label'])
print(list(le.classes_))
print(np.unique(Y_attack))
Y_class = dataset.iloc[:,-1].values
X = dataset.iloc[:,0:80].values
X = X.astype(int)'''

# ==== Data processing for TRAbID 2017 ====
from scipy.io import arff
data = arff.loadarff('TRAbID2017_dataset.arff')
dataset = pd.DataFrame(data[0])
print(dataset.head())
print(dataset.shape)

# Creating X and Y from the dataset
X = dataset.iloc[:,0:43].values
Y_class = pd.read_csv('TRAbID2017_dataset_Y_class.csv')
Y_class = Y_class.iloc[:,:].values

# Performing scale data
scaler = MinMaxScaler().fit(X)
X_scaled = np.array(scaler.transform(X))

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_class, test_size = 0.2, random_state = 42, stratify=Y_class)

print("Data Processing has been performed.")

# Tensorflow  placeholder  variables
X_placeholder = tf.placeholder(tf.float32 , shape=(None , X_train.shape[1]))
Y_placeholder = tf.placeholder(tf.float32 , shape=(None))

tf.set_random_seed(42)
model = mlp_model(X_train, Y_train)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

predictions = model(X_placeholder)
print('Prediction: ', predictions)

# ============== Training the model ==============
history = mlp_model_train(X_train, Y_train,
                0.1, # Validation Split
                64, # Batch Size
                100 # Epoch Count
                )

# ============== Evaluation of the model with actual instances ==============

print("Performance when using actual testing instances")
mlp_model_eval(X_test, Y_test, history, 1)


# ============== Generate adversarial samples for all test datapoints ==============

source_samples = X_test.shape[0]

# Jacobian-based Saliency Map
results = np.zeros((1, source_samples), dtype=float)
perturbations = np.zeros((1, source_samples), dtype=float)
grads = jacobian_graph(predictions , X_placeholder, 1)

X_adv = np.zeros((source_samples, X_test.shape[1]))

for sample_ind in range(0, source_samples):
    # We want to find an  adversarial  example  for  each  possible  target  class
    # (i.e. all  classes  that  differ  from  the  label  given  in the  dataset)
    current_class = int(np.argmax(Y_test[sample_ind]))
    
    # Target the benign class
    for target in [0]:
        if (current_class == 0):
            break
        
        # This call runs the Jacobian-based saliency map approac
        adv_x , res , percent_perturb = SaliencyMapMethod(sess, X_placeholder, predictions , grads,
                                             X_test[sample_ind: (sample_ind+1)],
                                             target , theta=1, gamma =0.1,
                                             increase=True ,
                                             clip_min=0, clip_max=1)
        
        X_adv[sample_ind] = adv_x
        results[target , sample_ind] = res
        perturbations[target , sample_ind] = percent_perturb


# ============== Evaluation of the model with adversarial instances ==============

print("Performance when using adversarial testing instances")
mlp_model_eval(X_adv, Y_test, history, 2)