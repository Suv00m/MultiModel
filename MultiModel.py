import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def build_model(df):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    
    # splitting  the data
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=(100-split_size)/100)

    st.markdown('**1.2. Data Splits:**')
    st.write('training set')
    st.info(x_train.shape)
    st.write('Test Set')
    st.info(y_train.shape)

    # features and label
    st.markdown('**1.3. Variable details:**')
    st.write('X Variable')
    st.info(list(x_train.columns))
    st.write('Y Variable')
    st.info(y_train.name)

    # model
    model  = RandomForestRegressor()
    model.fit(x_train,y_train)

    #prediction

    #training set
    st.subheader('2. Model Performance')
    st.markdown('**2.1. Training Set:**')
    st.write('Prediction of Training dataset')
    y_pre = model.predict(x_train)
    st.dataframe(y_pre)
    #download--------------------
    y_pre_df= pd.DataFrame(y_pre)
    csv = convert_df(y_pre_df)
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',)
    #----------------------------
    st.write('coeff of determination($r^2$):')
    st.write('= actual - prediction')
    st.info(r2_score(y_train,y_pre))
    m= x_train.shape[0]
    st.write('Cost:')
    err1 = (1/(2*m))*(mean_squared_error(y_train,y_pre))
    st.info(err1)

    #test set
    st.markdown('**2.2. Test Set**')
    st.write('Prediction of Training dataset')
    y_pre1 = model.predict(x_test)
    st.dataframe(y_pre1)
    #download-------------------
    y_pre_df_1= pd.DataFrame(y_pre1)
    csv = convert_df(y_pre_df_1)
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',)
    #---------------------------
    st.write('coeff of determination($r^2$):')
    st.write('= actual - prediction')
    st.info(r2_score(y_test,y_pre1))
    st.write('Cost:')
    err2 = (1/(2*m))*(mean_squared_error(y_test,y_pre1))
    st.info(err2)

def build_model_1_classifier(df):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    
    # splitting  the data
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=(100-split_size)/100)
    st.markdown('**1.2. Data Splits:**')
    st.write('training set')
    st.info(x_train.shape)
    st.write('Test Set')
    st.info(y_train.shape)

    # features and label
    st.markdown('**1.3. Variable details:**')
    st.write('X Variable')
    st.info(list(x_train.columns))
    st.write('Y Variable')
    st.info(y_train.name)

    #model
    model = XGBClassifier()
    model.fit(x_train,y_train)
    
    #prediction

    #training set
    st.subheader('2. Model Performance')
    st.markdown('**2.1. Training Set:**')
    st.write('Prediction of Training dataset')
    y_pre = model.predict(x_train)
    st.dataframe(y_pre)
    #download-------------------
    y_pre_df= pd.DataFrame(y_pre)
    csv = convert_df(y_pre_df)
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',)
    #---------------------------
    st.write('coeff of determination($r^2$):')
    st.write('= actual - prediction')
    st.info(r2_score(y_train,y_pre))
    m= x_train.shape[0]
    st.write('Cost:')
    err1 = (1/(2*m))*(mean_squared_error(y_train,y_pre))
    st.info(err1)

    #test set
    st.markdown('**2.2. Test Set**')
    st.write('Prediction Test dataset')
    y_pre1 = model.predict(x_test)
    st.dataframe(y_pre1)
    #download-------------------
    y_pre_df_1= pd.DataFrame(y_pre1)
    csv = convert_df(y_pre_df_1)
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',)
    #---------------------------
    st.write('coeff of determination($r^2$):')
    st.write('= actual - prediction')
    st.info(r2_score(y_test,y_pre1))
    st.write('Cost:')
    err2 = (1/(2*m))*(mean_squared_error(y_test,y_pre1))
    st.info(err2)

def build_model_1_regressor(df):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    
    # splitting  the data
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=(100-split_size)/100)
    st.markdown('**1.2. Data Splits:**')
    st.write('training set')
    st.info(x_train.shape)
    st.write('Test Set')
    st.info(y_train.shape)

    # features and label
    st.markdown('**1.3. Variable details:**')
    st.write('X Variable')
    st.info(list(x_train.columns))
    st.write('Y Variable')
    st.info(y_train.name)

    #model
    model = XGBRegressor()
    model.fit(x_train,y_train)
    
    #prediction

    #training set
    st.subheader('2. Model Performance')
    st.markdown('**2.1. Training Set:**')
    st.write('Prediction of Training dataset')
    y_pre = model.predict(x_train)
    st.dataframe(y_pre)
    #download-------------------
    y_pre_df= pd.DataFrame(y_pre)
    csv = convert_df(y_pre_df)
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',)
    #---------------------------
    st.write('coeff of determination($r^2$):')
    st.write('= actual - prediction')
    st.info(r2_score(y_train,y_pre))
    m= x_train.shape[0]
    st.write('Cost:')
    err1 = (1/(2*m))*(mean_squared_error(y_train,y_pre))
    st.info(err1)

    #test set
    st.markdown('**2.2. Test Set**')
    st.write('Prediction Test dataset')
    y_pre1 = model.predict(x_test)
    st.dataframe(y_pre1)
    #download-------------------
    y_pre_df_1= pd.DataFrame(y_pre1)
    csv = convert_df(y_pre_df_1)
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',)
    #---------------------------
    st.write('coeff of determination($r^2$):')
    st.write('= actual - prediction')
    st.info(r2_score(y_test,y_pre1))
    st.write('Cost:')
    err2 = (1/(2*m))*(mean_squared_error(y_test,y_pre1))
    st.info(err2)
# heading
image = Image.open('C:\pyhton\multimodel(2)\Purple Modern Best Portfolio LinkedIn Banner .png')
st.image(image, use_column_width=True)

#sidebar
with st.sidebar.header('1. Upload your CSV data'):
    upld_file= st.sidebar.file_uploader('Upload your input CSV file',type=["csv"])

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

#for random forest (build_model)
def upld_df(upld_file):
    #dataset
    st.subheader('1.Data Set')
    if upld_file is not None:
        df = pd.read_csv(upld_file,axis=1)
        st.markdown('**1.1. Glimpse of Dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Diabetes dataset
            diabetes = load_diabetes()
            X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            Y = pd.Series(diabetes.target, name='response')
            df = pd.concat( [X,Y], axis=1 )

            st.markdown('The Diabetes dataset is used as the example.')
            st.write(df.head(5))
            build_model(df)



#for xgboost (build_model_1_classifier) 
def upld_df_1_classifier(upld_file):
    #dataset
    st.subheader('1.Data Set')
    if upld_file is not None:
        df = pd.read_csv(upld_file, axis=1)
        st.markdown('**1.1. Glimpse of Dataset**')
        st.write(df)
        build_model_1_classifier(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Diabetes dataset
            diabetes = load_diabetes()
            X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            Y = pd.Series(diabetes.target, name='response')
            df = pd.concat( [X,Y], axis=1 )

            st.warning('this might not be optimal dataset but for eg. you can consider it')
            st.markdown('The Diabetes dataset is used as the example.')
            st.write(df.head(5))
            build_model_1_classifier(df)

#for xgboost (build_model_1_regressor) 
def upld_df_1_regressor(upld_file):
    #dataset
    st.subheader('1.Data Set')
    if upld_file is not None:
        df = pd.read_csv(upld_file, axis=1)
        st.markdown('**1.1. Glimpse of Dataset**')
        st.write(df)
        build_model_1_regressor(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Diabetes dataset
            diabetes = load_diabetes()
            X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            Y = pd.Series(diabetes.target, name='response')
            df = pd.concat( [X,Y], axis=1 )

            st.warning('this might not be optimal dataset but for eg. you can consider it')
            st.markdown('The Diabetes dataset is used as the example.')
            st.write(df.head(5))
            build_model_1_regressor(df)

#model selection
select_model = st.selectbox('Pick one', ['Random Forrest Regressor', 'XGBoost(classifier)','XGBoost(regressor)'])
if select_model == 'Random Forrest Regressor':
    upld_df(upld_file)
elif select_model == 'XGBoost(classifier)':
    upld_df_1_classifier(upld_file)
elif select_model == 'XGBoost(regressor)':
    upld_df_1_regressor(upld_file)
else :
    st.warning('Input Please')

#================================================================================

#contact form , my link for linkedin

st.subheader('2.Keep in Touch')


contact_form = """
<form action="https://formsubmit.co/storagebro62@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="name" required>
     <input type="email" name="email" placeholder="email" required>
     <textarea name="message" placeholder="message"></textarea>
     <button type="submit">Send</button>
</form>
"""
st.markdown(contact_form,unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("C:\pyhton\multimodel(2)\style\style.css")