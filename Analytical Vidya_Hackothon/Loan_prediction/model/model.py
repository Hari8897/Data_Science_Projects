#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[21]:


# Read the data
df=pd.read_csv('../Data/House_loan_df.csv')
df.head(3)


# In[22]:


Loan_ID=df['Loan_ID']
df.drop('Loan_ID',axis=1,inplace=True)


# In[23]:


df.shape


# In[24]:


df.info()


# In[25]:


df.describe()


# In[26]:


# Ceck the missing values
df.isnull().sum()


# In[27]:


# Fill Numerical colums with mean And Caregorical columns with Mode
cat_cols=df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['number','float']).columns
print(f'cat_cols : {cat_cols}')
print(f'num_cols : {num_cols}')


# In[33]:


# fill missing values in categorical columns with mode
missing_value_cols = df.columns[df.isnull().any()].tolist()
# Fill missing values in numeric columns with mean
for col in missing_value_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)



# In[34]:


df.isnull().sum()


# In[35]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

for col in df.columns:
    if df[col].dtype=='object':
        df[col]=label_encoder.fit_transform(df[col])
df.head()


# In[38]:


# Divide the data into test and train data
X = df.drop('Loan_Status', axis=1)
y =df['Loan_Status']

X.shape,y.shape


# In[42]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=42, test_size=0.2)

print('X train:',X_train.shape)
print('X test:',X_test.shape)
print('y train:',y_train.shape)
print('y test:',y_test.shape)


# In[44]:


# Load the model
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[45]:


# predictions 
prediction = lr.predict(X_test)
prediction



# In[66]:


from sklearn.metrics import accuracy_score,\
                            precision_score,\
                            recall_score,\
                            f1_score,\
                            classification_report,\
                            roc_auc_score,roc_curve,auc
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# ============================ Step-6: Metrics==================================================================
acc_log= round(accuracy_score(y_test,prediction)*100,2)
f1_log=round(f1_score(y_test,prediction),2)
precision_log=round(precision_score(y_test,prediction),2)
recall_log=round(recall_score(y_test,prediction),2)
print("accuray is:",acc_log)
print("F1 is:",f1_log)
print("Precision is:",precision_log)
print("Recall is:",recall_log)

# ================================Step-7:Confusion matrix=========================================================================
cmt=confusion_matrix(y_test,prediction)

disp=ConfusionMatrixDisplay(confusion_matrix=cmt,
                            display_labels = [False, True])
disp.plot()
plt.grid(False)
plt.show()


tn, fp, fn, tp = confusion_matrix(y_test,prediction).ravel()
print("True negative:",tn)
print("False postive:",fp)
print("False negative:",fn)
print("True postive:",tp)

#=======================================Step-8: ROC-AUC curve================================================================

y_log_pred_prob=lr.predict_proba(X_test)[:,1]   # Class-1 probabilities
fpr,tpr,threshold=roc_curve(y_test,y_log_pred_prob)
plt.plot(fpr,tpr)
plt.show()


# In[67]:


import joblib

joblib.dump(lr,"model.pkl")


# In[69]:


# Load test data
test_df = pd.read_csv("../Data/test.csv")

# Save Loan_ID for submission
loan_ids = test_df['Loan_ID']

# Drop Loan_ID from test data for prediction
X_test = test_df.drop('Loan_ID', axis=1)


# In[70]:


# Fill missing values
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
    else:
        X_test[col] = X_test[col].fillna(X_test[col].mean())


# In[72]:


# Load the trained model
model = joblib.load("model.pkl")  # replace with actual file path

# Encode categorical columns 
cat_cols = X_test.select_dtypes(include='object').columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])
    label_encoders[col] = le  # save if needed again


# In[73]:


# Make predictions
predictions = model.predict(X_test)
predictions


# In[76]:


# Decode predictions 
# Example: 1 → 'Y', 0 → 'N'
decoded_preds = ['Y' if p == 1 else 'N' for p in predictions]
decoded_preds[:10]


# In[77]:


# Create submission DataFrame
submission_df = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Loan_Status': decoded_preds
})

# Save to CSV
submission_df.to_csv("Predction.csv", index=False)



