import streamlit as st
import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import *
from pycaret.regression import *

def read_data(file_path , file_format):

    try:

        if file_format == '.csv':
            return pd.read_csv(file_path)
        elif file_format == '.xlsx':
            return pd.read_excel(file_path)
        elif file_format == '.json':
            return pd.read_json(file_path)
        else :
            raise ValueError()
    except ValueError:
        return ValueError        


def handle_missing_values(data, strategy = 'mean'):

      categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
      numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

      try :
        if strategy == 'mean':
          for col in numeric_cols :
              data[col].fillna(data[col].mean(), inplace=True)
          for col in categorical_cols :
               data[col].fillna(data[col].mode().iloc[0], inplace=True)
          return data
        elif strategy == 'median' :
            for col in numeric_cols :
              data[col].fillna(data[col].median(), inplace=True)
            for col in categorical_cols :
              data[col].fillna(data[col].mode().iloc[0], inplace=True)
            return data
        elif strategy == 'mode' :
            for col in data.columns :
              data[col].fillna(data[col].mode().iloc[0], inplace=True)
            return data
        elif strategy == 'drop' :
            return data.dropna()
        else :
            raise ValueError()
      except ValueError :
          print("Invalid missing value strategy. Supported strategies: 'mean', 'median', 'mode', 'drop'")

def encode_categorical(data,categorical_cols, strategy):
      try :
        if strategy == 'OneHot':
           data =  pd.get_dummies(data , columns = categorical_cols)
          
        elif strategy == 'label' :
          data[data.columns[-1]] = LabelEncoder().fit_transform(data[data.columns[-1]])
        else :
            raise ValueError()
      except Exception as e:
        print(f"Error reading data: {str(e)}")
      return data

def drop_columns(data, columns_to_drop):
    return data.drop(columns_to_drop ,axis=1)


st.header("LOAD YOUR DATASET")
file = st.file_uploader("Browse")
if file: 
    try:
        df = read_data(file,Path(file.name).suffix)
        st.dataframe(df)
        st.header("DATA PREPROCESSING")

        st.subheader("handle missing values")
        strategy = st.selectbox("choose your strategy to handle missing values " , ["mean" , "mode" , "median" , "drop"])
        Handle = st.button("Handle")
        handled_data = handle_missing_values(df , strategy)

        if Handle :
            st.dataframe(handled_data)
            st.success("missing values are handled successfully")
       

        st.subheader("drop columns")
        columns_to_drop = st.multiselect("choose columns to drop " , handled_data.columns)
        Drop = st.button("Drop")
        data_after_drop = drop_columns(handled_data , columns_to_drop)
        
        if Drop :
            
            st.dataframe(data_after_drop)
            st.success("columns have been dropped successfully")

        st.subheader("choose your target")
        target = st.selectbox("choose your target label  " , data_after_drop.columns)
        Select = st.button("Select")
        x, y = data_after_drop.drop(target,axis=1) , pd.DataFrame(data_after_drop[target])

        if Select :
        
            col1 , col2  = st.columns([3 , 1])
            with col1 :
                st.text("data features")
                col1.dataframe(x)

            with col2 :
                st.text("label")
                col2.dataframe(data_after_drop[target])


        st.subheader("encodig categorical columns")
        categorical_cols = st.multiselect("choose categorical columns to encoden (note* if y label is categorical please select it also) " , data_after_drop.columns)
        categorical_cols2 = categorical_cols.copy()
        Encode = st.button("Encode")


        
        if y.columns[0] in categorical_cols:

            encoded_x =encode_categorical(x, categorical_cols.remove(y.columns[0]),'OneHot' )
            encoded_y = encode_categorical(y, y.columns[0],'label' )
            
            st.text("you chose the label as categorical column then it's a classfication problem")
        else:
            encoded_x = encode_categorical(x, categorical_cols , 'OneHot')
            encoded_y = y 
            st.text("you didin't choose the label as categorical column then it's a regression problem")

        if Encode :
            
            col1 , col2  = st.columns([3 , 1])
            with col1 :
                st.text("data features")
                col1.dataframe(encoded_x)

            with col2 :
                st.text("label")
                col2.dataframe(encoded_y)

        st.header("TRAIN MODEL BY PYCARET")
        pycaret = st.toggle("pycaret")

        cls_pycaret = ClassificationExperiment()
        reg_pycaret = RegressionExperiment()
        if pycaret:
            x[y.columns[-1]] = y
            
            if x.columns[-1] in categorical_cols2:
                cls_pycaret.setup(x, target= x.columns[-1],  categorical_features= categorical_cols2.remove(x.columns[-1]))
                with st.spinner("trying to determine best model ..."):
                    best_model = cls_pycaret.compare_models()
                results = cls_pycaret.pull()
                st.text("best model is ")
                st.text(best_model)
                st.dataframe(results)
                st.subheader("train model")
                st.text("holdout predictions")
                pred_holdout = cls_pycaret.predict_model(best_model)
                st.dataframe(pred_holdout)
                st.text("all predictions")
                predicitions = cls_pycaret.predict_model(best_model , x.drop(x.columns[-1] , axis = 1))
                st.dataframe(predicitions)
                
            else:
                reg_pycaret.setup(x, target= x.columns[-1],  categorical_features= categorical_cols2)
                with st.spinner("trying to determine best model ..."):
                    best_model = reg_pycaret.compare_models()
                
                results = reg_pycaret.pull()
                st.text("best model is ")
                st.text(best_model)
                st.dataframe(results)
                st.subheader("train model")
                st.text("holdout predictions")
                pred_holdout = reg_pycaret.predict_model(best_model)
                st.dataframe(pred_holdout)
                st.text("all predictions")
                predicitions = reg_pycaret.predict_model(best_model , x.drop(x.columns[-1] , axis = 1))
                st.dataframe(predicitions)

            
        
             
            
                        
        


        
    except Exception as e:
        st.error(e)




        
    