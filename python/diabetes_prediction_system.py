import numpy as np  #array
import pandas as pd  #creating data plan
from sklearn.preprocessing import StandardScaler # standardize the data
from sklearn.model_selection import train_test_split #use to split the data
from sklearn import svm #
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv('C:\Users\AAYUSH\OneDrive\Desktop\python\diabetes (1).csv')
pd.read_csv?
#printing the first 5 rows of the dataset
diabetes_dataset.head()
#number of rows and columns in this dataset
diabetes_dataset.shape
#getting the statistical measures of the data
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
#label 0 represents non diabetic patient
# label 1 represents diabetic patient
diabetes_dataset.groupby('Outcome').mean()
#separating the data and labels
X= diabetes_dataset.drop(columns = 'Outcome',axis=1)
Y= diabetes_dataset['Outcome']
print(X)
print(Y)
#standardization of data
scaler = StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X) #fit and transform the data in same range
print(standardized_data)
X = standardized_data
Y= diabetes_dataset['Outcome']
print(X)
print(Y)
#train test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
#training the model
classifier = svm.SVC(kernel='linear')
#traing the support vector machine Classifier
classifier.fit(X_train, Y_train)
#evaluate the model
#accuracy Score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:',training_data_accuracy)
#accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:',test_data_accuracy)
#making a predictive SYstem
input_data = (10,168,74,0,0,38,0.537,34)
#we need to change the given data in to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standardize the input_data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
  print('the person is not diabetic')
else:
  print('the person is diabetic')  