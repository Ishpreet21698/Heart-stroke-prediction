#from fastapi import FastAPI
#from pydantic import BaseModel
import pickle
import streamlit as st
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

'''
app=FastAPI()
class request_body(BaseModel):
    Age:float
    Hypertension:str
    Heart_disease:str
    Average_glucose: float
    BMI: float
    Marital_status: str
    Gender: str
    Work_type: str
    Residence:str
    Smoking_status: str
'''
df=pd.read_csv("stroke_data_Cleaned3.csv",header=0)
print(df.head())

#Creating X and y Variables for Training Testing datasets
X=df.drop(["stroke"],axis=1)
y=df["stroke"] #Target Variable

#Creating Training Testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=5)
#I have tried using K-best(taking 6 Variables) and the Classification Report doesnot show much Variation thus we are not using K-best

#Creating Classification Models
#I am using Naive Bayes Classification here, thus Scaling and One Hot Encoding is not Required
model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Testing effeciency of Naive Bayes Classifier
from sklearn.metrics import confusion_matrix, classification_report,recall_score,f1_score
cm=confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
cr=classification_report(y_test, y_pred)
print("Classification Report")
print(cr)
#Since this dataset is not balanced accuracy is not the measure we will be looking for
# Here we want to reduce the number of False negatives so we will look at Recall and F1 Score
rs=recall_score(y_test,y_pred, average="weighted")
fs=f1_score(y_test,y_pred,average="weighted")
print("Recall Value: ",rs)
print("F1 Score: ",fs)

#Pickle is used for saving .pkl file
#We are using .pkl file because we are going to deploy the model on Streamlit
pickle.dump(model,open('stroke.pkl','wb'))
loaded_model=pickle.load(open('stroke.pkl','rb'))

#For deploying on the Website
def predict_input_page():
    loaded_model = pickle.load(open('stroke.pkl', 'rb'))
    st.title("Stroke Prediction Model")
    Age=st.slider("Age: ", min_value=0, max_value=90)
    Hypertension = st.radio("Do you suffer from Hypertension: ",("Yes","No"))
    Heart_disease=st.radio("Do you suffer from Heart Disease: ",("Yes","No"))
    Average_glucose=st.slider("Average Glucose Levels: ", min_value=50, max_value=280)
    BMI=st.slider("BMI: ",min_value=10, max_value=70)
    Marrital_status=st.radio("Are you married: ",("Yes","No"))
    Gender=st.radio("What is your Gender: ",("Male","Female"))
    Work_type=st.radio("What is your Work type?",("Private","Self-employed","children","Govt_job","Never_worked"))
    Residence=st.radio("What is your area of Residence",("Urban","Rural"))
    Smoking_status=st.radio("Enter your Smoking Status:",("never smoked","Unknown","formerly smoked","smokes"))
    ok=st.button("Predict")

    #Since we are taking the input as a string and the model needs the values in numbers we convert the String to int
    if Hypertension=="Yes":
        Hypertension= 1
    elif Hypertension=="No":
        Hypertension=0

    if Heart_disease=="Yes":
        Heart_disease= 1
    elif Heart_disease=="No":
        Heart_disease=0

    if Marrital_status=="Yes":
        Marrital_status=1
    elif Marrital_status=="No":
        Marrital_status=0

    if Gender=="Male":
        Gender=1
    elif Gender=="Female":
        Gender=0

    if Work_type=="Govt_job":
        Work_type=0
    elif Work_type=="Never_worked":
        Work_type=1
    elif Work_type=="Private":
        Work_type=2
    elif Work_type=="Self-employed":
        Work_type=3
    elif Work_type=="children":
        Work_type=4

    if Residence=="Rural":
        Residence=0
    elif Residence=="Urban":
        Residence=1

    if Smoking_status=="Unknown":
        Smoking_status=0
    elif Smoking_status=="formerly smoked":
        Smoking_status=1
    elif Smoking_status=="never smoked":
        Smoking_status=2
    elif Smoking_status=="smokes":
        Smoking_status=3

    testdata=np.array([[Age, Hypertension, Heart_disease, Average_glucose, BMI, Marrital_status,Gender,Work_type,Residence,Smoking_status]])
    classi=loaded_model.predict(testdata)[0]

    try:
        if ok==True:
            if classi == 0:
                st.success("Awesome! You are on low risk of getting Stroke")
            elif classi == 1:
                st.error("Cautious! You are on high risk of getting Stroke")
    except:
        st.info("Enter some Data")




