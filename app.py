import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

#load all the pickle files and models
model=load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_encoding_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('std_scalar.pkl','rb') as file:
    scalar=pickle.load(file)

#streamlit app
st.title("Customer Churn Prediction")

#user Input
geography=st.selectbox("Geograpgy",onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
Credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Products",1,4)
has_credit_card=st.selectbox("Has credit card",[0,1])
is_active_member=st.selectbox("Is active member",[0,1])

Input={'CreditScore': [Credit_score],
         'Gender':[label_encoder_gender.transform([gender])[0]],
         'Age':[age]
 , 'Tenure':[tenure],
   'Balance':[balance],
   'NumOfProducts': [num_of_products],
  'HasCrCard':[has_credit_card]
 ,'IsActiveMember':[is_active_member]
 , 'EstimatedSalary':[estimated_salary]
}

Input_data=pd.DataFrame(Input)
# One-hot encode "Geography"
geo_encoded=onehot_encoder_geo.transform([[geography]])
data_geo=pd.DataFrame(geo_encoded.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

StopAsyncIterationInput_data=pd.concat([Input_data.reset_index(drop=True),data_geo],axis=1)

#standard scaling
Input_data=scalar.transform(Input_data)

prediction=model.predict(Input_data)
pred_prob= prediction[0][0]

if pred_prob>0.5:
    st.write("customer is likely to churn")
else:
    st.write("customer is unlikely to churn")