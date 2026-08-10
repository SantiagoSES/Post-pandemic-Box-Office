#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import mlflow
import mlflow.sklearn

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[44]:


#Training Data.
#Skip the header as it is not a number.
#Each row corresponds to a movie and displays its runtime, IMDB rating, IMDB number of votes,...
X = np.genfromtxt('/Data/Training_X.csv', delimiter=',', skip_header=1)  
m, n = X.shape
#Each row of y says if 2.5*movie_budget was greater than worldwide_gross. This is the metric we are using to 
#determine if a movie was successful.
y = np.genfromtxt('/Data/Training_Y.csv', delimiter=',', skip_header=1)


# In[45]:


#Splitting data to have a part of it for testing and cross-validation
#I am saving 30% of the total data for that purpose.
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.3, random_state=1)

#Then split into test and cross-validation that 30%. Half goes to cv half to test.
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)

#Normalize the data so that features are on the same range.
#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train_norm=scaler.transform(X_train)

#Transform test and cross-validation data as well
#X_cv_norm=scaler.transform(X_cv)
#X_test_norm=scaler.transform(X_test)


# In[46]:


#Normalize the data so that features are on the same range.
#Tried also standard scaler.
scaler= StandardScaler()
scaler.fit(X_train)
X_train_norm=scaler.transform(X_train)

#Transform test and cross-validation data as well
X_cv_norm=scaler.transform(X_cv)
X_test_norm=scaler.transform(X_test)


# In[47]:


#Multilayer perceptron to be trained on the data.
#Input size is the number of features of each example.
#, kernel_regularizer=regularizers.l2(0.01)
model = Sequential(                      
    [                                   
        tf.keras.Input(shape=(n,)),  
        Dense(40, activation='relu'),
        Dense(1,  activation='linear')
    ], name = "my_model"                                    
) 
#I made the last layer linear and apply the sigmoid manually.

#Define a loss function and use adaptative alpha (Adam) to perform gradient descent. 
#It is important to specify from_logits=True as the last layer is not a sigmoid. To make it numerically stable
#the sigmoid is in that way applied when computing the cost. 
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

#I use early_stopping to allow a fast evaluation of any one model.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',      
    patience=5,              # epochs to wait before stopping
    restore_best_weights=True
)

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_cv_norm, y_cv),
    epochs=1000,
    callbacks=[early_stopping]
)
'''
model.fit(
    X_train_norm, y_train,
    validation_data=(X_cv_norm, y_cv),
    epochs=170
)
'''


# In[48]:


val_loss = model.evaluate(X_cv_norm, y_cv)
print("Validation loss:", val_loss)


# In[49]:


#Routine to make predictions for X_input
def model_predict(X_input):
    '''
     X_input    : (ndarray )  Array of features to predict
     returns yhat a numpy array of 0/1 values corresponding to predictions.
    '''    
    #Making predictions.
    prediction = model.predict(X_input)
    #Applying the sigmoid at the end manually.
    probs = tf.nn.sigmoid(prediction).numpy()

    #Convert to 0/1 predictions
    yhat=np.zeros(len(probs))
    for i in range(len(probs)):    
        #Threshold to convert the predictions into TRUE/FALSE outputs.
        if probs[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    return yhat



# In[50]:


# Classification Error.
def eval_cat_err(y, yhat):
    """ 
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example           
    """
    m = len(y)
    incorrect = 0
    for i in range(m):

        if(y[i]!=yhat[i]):
            incorrect+=1

    cerr=incorrect/m


    return(cerr)

#Error on training set and cross-validation set.
training_cerr = eval_cat_err(y_train, model_predict(X_train_norm))
cv_cerr = eval_cat_err(y_cv, model_predict(X_cv_norm))


print("Training Error")
print(training_cerr)
print("Cross-validation Error")
print(cv_cerr)
print("With weights")
print(model.get_weights())


# In[51]:


#Error for reporting model accuracy
accuracy = eval_cat_err(y_test, model_predict(X_test_norm))
print(accuracy)

#Saving the experiments to keep a record and then easily determine the best
mlflow.set_experiment("BOff_Success_Prediction")

with mlflow.start_run():
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("Training Error", training_cerr)
    mlflow.log_metric("Cross-validation Error", cv_cerr)
    mlflow.tensorflow.log_model(
        model,
        name="model"
    )


# In[30]:


'''from mlflow import MlflowClient

client = MlflowClient()

runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.val_loss ASC"]
)

for run in runs:
    print(run.info.run_id)
    print(run.data.metrics)
    print(run.data.params)
    model = mlflow.tensorflow.load_model(
    f"runs:/{run.info.run_id}/model"
    )
    model.summary()
    '''


# In[52]:


"Serialize the model"
import joblib
joblib.dump(model, "model.pkl")


# In[53]:


#Now we test data from 2023
#Training Data.
#Skip the header as it is not a number.
#Each row corresponds to a movie and displays its runtime, IMDB rating, IMDB number of votes,...
X_1 = np.genfromtxt('/Data/Hypothesis_X.csv', delimiter=',', skip_header=1)  
m_1, n_1 = X_1.shape
#Each row of y says if 2.5*movie_budget was greater than worldwide_gross. This is the metric we are using to 
#determine if a movie was successful.
y_1 = np.genfromtxt('/Data/Hypothesis_Y.csv', delimiter=',', skip_header=1)

#We have to scale before predicting
X_1_test=scaler.transform(X_1)

data2023_cerr = eval_cat_err(y_1, model_predict(X_1_test))


print("2023 Error")
print(data2023_cerr)


# In[73]:


#Finally, I generate a confusion matrix for 2023 data.
y_pred_probs = model.predict(X_1_test)
y_pred = (y_pred_probs >= 0.5).astype(int).ravel()

true_positives=[i for i in range(len(y_1)) if y_pred[i]==1 and y_1[i]==0 ]
positives=[i for i in range(len(y_1)) if y_1[i]==0 ]

print(f"True positives {len(true_positives)/len(positives)}")

cm = confusion_matrix(y_1, y_pred, normalize="true")

plt.figure(figsize=(8, 6))

class_names=['Flop', 'BOffice Success']

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("../plots/confusion_matrix_2013.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:




