import streamlit as st
from PIL import Image
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import base64


st.set_page_config(page_title="INDUSTRIAL COPPER MODELING",layout="wide")
st.header(":red[INDUSTRIAL COPPER MODELING]")

tab1,tab2,tab3=st.tabs(["Introduction","Selling Price Prediction","Status Prediction"])
default_option="Introduction"
    



with tab1:
    col1,col2,col3 = st.columns([6,0.1,6])
    with col1:
        st.write("")
        st.image(Image.open("C:\\Users\\prave\\Downloads\\copper.jpeg"), width=450)
        
    with col3:
        st.write("#### :red[**Overview :**] This project aims to develop a machine learning model and deploy it as a user-friendly online application to accurately predict the selling price and status of copper transactions based on historical data.")
        st.markdown("#### :red[**Technologies Used :**] Python, Pandas,NumPy, Visualization, Streamlit, Scikit-learn,Pickle")
    
with tab2: 
    st.header("Fill the below following details to predict the Selling Price")
    st.write("NOTE: Min and Max values are provided for reference, You can enter your desired value.")
    country_option=['28.0', '25.0', '30.0', '32.0', '38.0', '78.0', '27.0', '77.0','113.0', '79.0', '26.0', '39.0', '40.0', '84.0', '80.0', '107.0','89.0']
    item_type_option=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    status_option=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised', 'Offered', 'Offerable']
    application_option=[10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20.,66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69.,70., 65., 58., 68.]
    produ_ref_option=['1670798778', '1668701718', '628377', '640665', '611993','1668701376', '164141591', '1671863738', '1332077137', '640405',
       '1693867550', '1665572374', '1282007633', '1668701698', '628117','1690738206', '628112', '640400', '1671876026', '164336407',
       '164337175', '1668701725', '1665572032', '611728', '1721130331','1693867563', '611733', '1690738219', '1722207579', '929423819',
       '1665584320', '1665584662', '1665584642']
    
    with st.form(key='my_form'):
        
        
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.subheader("Select Your Options")
            Country=st.selectbox("Country",sorted(country_option),key=1)
            Item_type=st.selectbox("Item Type",item_type_option,key=2)
            Product_ref=st.selectbox("Product Reference",produ_ref_option,key=3)
            Application=st.selectbox("Application",application_option,key=4)
            status=st.selectbox("Status",status_option,key=5)
        with col3:
            st.subheader("Enter your values")
            customer=st.text_input("Customer ID(Min:12458.0 & Max:2147483647.0)")
            quantity_tons=st.text_input("Quantity_tons(Min:1e-05 & Max:1000000000.0 )")
            thickness=st.text_input("Thickness(Min:0.18 & Max:2500.0)")
            width=st.text_input("width (Min:1.0 & Max:2990.0)")

            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")

        if submit_button:

            with open('model.pkl', 'rb') as file:
                training=pickle.load(file)

            with open('c.pkl', 'rb') as f:
                country_loaded=pickle.load(f)
            with open('i.pkl', 'rb') as f:
                item_loaded=pickle.load(f)
            with open('s.pkl', 'rb') as f:
                status_loaded=pickle.load(f)
            
            with open('p.pkl', 'rb') as f:
                product_loaded=pickle.load(f)

            with open('scalar.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)


            #'quantity_tons_log', 'thickness_log', 'width', 'application', 'customer',country, item_type,status, product_ref
            new_sample = np.array([[np.log(float(quantity_tons)), np.log(float(thickness)), float(width), Application, float(customer), Country,Item_type, status, Product_ref]])

            # Transform new sample categorical features using already fitted encoders
            country = country_loaded.transform(new_sample[:, [5]]).toarray()
            item = item_loaded.transform(new_sample[:, [6]]).toarray()
            status = status_loaded.transform(new_sample[:, [7]]).toarray()
            prod = product_loaded.transform(new_sample[:, [8]]).toarray()

            new_samples=np.concatenate((new_sample[:, [0, 1, 2, 3, 4]].astype(float),country,  item, status,prod), axis=1)
            new_samples1= scaler_loaded.transform(new_samples)

            new_pred = training.predict(new_samples1)
            predicted_selling_price = np.exp(new_pred)
            st.write(f'## :green[Predicted selling price:]:{predicted_selling_price}')

with tab3:
    st.header("Fill the below following details to predict the Status")
    st.write("NOTE: Min and Max values are provided for reference, You can enter your desired value.")
    with st.form(key='my_form1'):
        
        
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.subheader("Select Your Options")
            S_Country=st.selectbox("Country",sorted(country_option),key=11)
            S_Item_type=st.selectbox("Item Type",item_type_option,key=12)
            S_Product_ref=st.selectbox("Product Reference",produ_ref_option,key=13)
            S_Application=st.selectbox("Application",application_option,key=14)
        
        with col3:
            st.subheader("Enter your values")
            S_customer=st.text_input("Customer ID(Min:12458.0 & Max:2147483647.0)")
            S_quantity_tons=st.text_input("Quantity_tons(Min:1e-05 & Max:1000000000.0 )")
            S_thickness=st.text_input("Thickness(Min:0.18 & Max:2500.0)")
            S_width=st.text_input("width (Min:1.0 & Max:2990.0)")
            S_selling=st.text_input("Selling price(Min:0.1 & Max:100001015.0 )")

            S_submit_button = st.form_submit_button(label="PREDICT STATUS")

            
        if S_submit_button:
            
            with open('classifier_model.pkl', 'rb') as file:
                ml=pickle.load(file)

            with open('country.pkl','rb') as f:
                country_pkl=pickle.load(f)

            with open('item.pkl','rb') as f:
                item_pkl=pickle.load(f)

            with open('product.pkl','rb') as f:
                product_pkl=pickle.load(f)

            with open('scaler.pkl','rb') as f:
                scalar_pkl=pickle.load(f)

        #'quantity_tons_log', 'thickness_log', 'width', 'application', 'customer','selling_price_log',country, item_type, product_ref

            new_sample = np.array([[np.log(float( S_quantity_tons)), np.log(float(S_thickness)), float(S_width), S_Application, float(S_customer),np.log(float(S_selling)), S_Country, S_Item_type,   S_Product_ref]])

            country_new = country_pkl.transform(new_sample[:, [6]]).toarray()
            item_new = item_pkl.transform(new_sample[:, [7]]).toarray()
            prod_new = product_pkl.transform(new_sample[:, [8]]).toarray()

            new_samples = np.concatenate((new_sample[:, [0,1,2,3,4,5]].astype(float), country_new, item_new, prod_new), axis=1)
            new_samples1 = scalar_pkl.transform(new_samples)
            # Predict with the model
            prediction = ml.predict(new_samples1)
            

            if prediction==1:
                st.success('Won')
            else:
                st.warning('Lost')

            


