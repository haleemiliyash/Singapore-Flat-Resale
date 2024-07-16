import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import datetime as dt
from PIL import Image

##-----------------------------------------------------------------Pickle--------------------------------------------------------------------------------------------#
with open("D:/project/.venv/sigapore_flat/sell.pkl",'rb') as file:
    price_model = pickle.load(file)
with open("D:/project/.venv/sigapore_flat/Cat_Columns_Encoded.json",'rb') as file:
    encode_file = json.load(file)

st.set_page_config(page_title='Flat Resell Price model', layout="wide")
st.title(':violet[*Singapore Resale Flat Prices Predicting Model By Abdul Haleem*]')

def home():
    col1,col2=st.columns(2)
    with col1:
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :orange[*Overview*] : Build regression model to predict the flat resale price. Dataset which used from 1990 -till now date")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :blue[*Technologies used*] : Python, Pandas, Numpy, Pickel, Matplotlib, Seaborn, Scikit-learn, Streamlit.")
    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        st.image(Image.open(r'D:/project/.venv/sigapore_flat/flat.jpg'),width=400)
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        st.image(Image.open(r'D:/project/.venv/sigapore_flat/hdb-flats.jpg'),width=400)

##----------------------------------------------------------Price prediction---------------------------------------------------------------------------------------------------------------------------#
def sell_price():
    st.markdown("# :orange[Predicting price based on Trained data and model]")
    with st.form("Regression"):
        col1,col2,col3=st.columns([0.5,0.2,0.5])
        with col1:
            start=1
            end=12
            month=st.number_input("select the**Transaction month**",min_value=1,max_value=12,value=start,step=1)
            town=st.selectbox("Select the **town**",encode_file['town_initial'])
            block=st.selectbox('Select the **Block**',encode_file['block_initial'])
            street=st.selectbox('Select the **street**',encode_file['street_name_initial'])
            flat_type=st.selectbox('Select the **Flat model**',encode_file['flat_type_initial'])
        
        with col3:
            year = st.number_input("Select the transaction Year", min_value=1990, max_value=2024, value=dt.datetime.now().year)
            floor_area=st.number_input('Select the **Floor area**',value=28.0,min_value=28.0,max_value=307.0,step=1.0)
            flat_model=st.selectbox('Select the**Flat model**',encode_file['flat_model_initial'])
            lease_year=st.number_input('Enter the **Lease Commence Year**', min_value=1966, max_value=2022, value=2017)
            storey_range = st.number_input('Select the **storey Range**', value=0, min_value=0, max_value=100)
            

        with col2:
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            col2.markdown("# ")
            st.markdown('Click below button to predict')
            button=st.form_submit_button(label='Predict')
    
    if button:
        town_encode = encode_file['town_initial'].index(town)
        flat_type_encode = encode_file['flat_type_initial'].index(flat_type)
        block_encode = encode_file['block_initial'].index(block)
        street_name_encode = encode_file['street_name_initial'].index(street)
        flat_model_encode = encode_file['flat_model_initial'].index(flat_model)

        input_ar = np.array([[town_encode,flat_type_encode,block_encode,street_name_encode,storey_range,floor_area, flat_model_encode,lease_year,year,month,]],dtype=np.float32)
        Y_pred=price_model.predict(input_ar)
        sell_price=round(Y_pred[0],2)
        st.header(f'Predicted Resell Price is: {sell_price}')

##-----------------------------------------------------------------Streamlit Part--------------------------------------------------------------------------------------------------------------##
with st.sidebar:
    option = option_menu("Main menu",['Home','Resell Price Prediction'],
                       icons=["house","cloud-upload","list-task","pencil-square"],
                       menu_icon="cast",
                       styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "green"},
                                   "nav-link-selected": {"background-color": "green"}},
                       default_index=0)
if option=='Home':
    home()
elif option=='Resell Price Prediction':
    sell_price()