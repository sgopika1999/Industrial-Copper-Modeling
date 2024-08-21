# Industrial-Copper-Modeling

**#Introduction:**
The "Industrial Copper Modeling" project is focused on addressing challenges in the copper manufacturing industry related to sales and pricing data. By leveraging machine learning, this project aims to improve the accuracy of predictions for selling prices and lead classification, which are critical for making informed business decisions.

__#Domain: Manufacturing__


__#Technologies Used:Python, Pandas, Numpy, Visualization, Streamlit, Scikit-learn, pickle__

__#Problem Statement :__

The copper industry's sales and pricing data are prone to issues such as skewness and noise, making manual predictions time-consuming and potentially inaccurate. This project seeks to build two machine learning models: a regression model to predict the 'Selling_Price' and a classification model to determine the 'Status' of a lead (WON/LOST). 

**#Approach:**

- Data Understanding: Analyzing variable types, distributions, and identifying rubbish values.

- Data Preprocessing: Handling missing values, treating outliers, and addressing skewness through transformations like log transformation.

- Feature Engineering: Encoding categorical variables, and creating new features where applicable.

- Model Building: Developing regression and classification models, evaluating performance using metrics like accuracy, F1 score, and AUC curve.

- Streamlit Integration: Building an interactive page for predictions by taking user inputs and applying the trained models to new data.
