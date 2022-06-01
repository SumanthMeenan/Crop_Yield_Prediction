import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection  import train_test_split

filename = 'lrmodel.pkl'
filename1 = 'svrmodel.pkl'

df = pd.read_excel('DATASET.xlsx',engine='openpyxl')
df.info()

df1 = df.drop(labels='yield(kg/hect)', axis = 1)

df1.corr()

sns.heatmap(df1.corr(), annot = True)

labelencoder = LabelEncoder() 
df1['State']= labelencoder.fit_transform(df1['State'])

onehotencoder = OneHotEncoder() 
encoded_df = pd.DataFrame(onehotencoder.fit_transform(df1[['State']]).toarray())

df2 = encoded_df.join(df1)
df3 = df2.drop(labels = [0, 'State'], axis =1)

sns.pairplot(df)
sns.distplot(df['Production(tons)']) 

X = df3.iloc[:, :29]
Y = df3['Production(tons)'] 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,  random_state = 20)

lm = LinearRegression()
lm.fit(x_train, y_train)
prediction = lm.predict(x_test) 
plt.scatter(y_test, prediction) 
sns.distplot((y_test - prediction))

metrics.mean_absolute_error(y_test, prediction)
np.sqrt(metrics.mean_squared_error(y_test, prediction))
metrics.explained_variance_score(y_test, prediction) 
pickle.dump(lm, open(filename, 'wb'))

lr_saved_model = pickle.load(open(filename, 'rb'))
lr_saved_model.score(x_test, y_test)

svr_model = SVR()
svr_model.fit(x_train, y_train) 

print(" SVM Model Accuracy {}".format(svr_model.score(x_train, y_train)))
print(" SVM Model Accuracy {}".format(svr_model.score(x_test, y_test)))

#Save Model
pickle.dump(svr_model, open(filename1, 'wb'))

#Load Model
svr_saved_model = pickle.load(open(filename1, 'rb'))
svr_saved_model.score(x_test, y_test)