import streamlit as st
import pandas as pd
import pickle

st.title("What's Your House Worth?")

st.header("A house sale price estimator for Ames, Iowa.")

st.subheader("First, we need some data on the house:")

lot_area = st.slider("What's the total lot area, in square feet?", 
                     min_value=1300, max_value=40000)

sqft_1st_flr = st.slider("What's the total area of the first floor, in square feet?", 
                         min_value=350, max_value=2250)

gr_liv_area = st.slider("What's the total ground-level living area, in square feet?", 
                         min_value=350, max_value=3000)


if st.button("Push here to get started!"):
    st.subheader("Calculating estimated price for the following home details:")
    st.write(f"Lot Area: {lot_area} sqft")
    st.write(f"First Floor Area: {sqft_1st_flr} sqft")
    st.write(f"Ground-Level Living Area: {gr_liv_area} sqft")

    model = pickle.load(open('model.sav', 'rb'))

    data = pd.DataFrame(columns = ['LotArea', '1stFlrSF', 'GrLivArea'])
    data = data.append({'LotArea': lot_area, 
                        '1stFlrSF' : sqft_1st_flr, 
                        'GrLivArea': gr_liv_area}, 
                       ignore_index=True)
    pred = model.predict(data)

    st.subheader(f"Predicted Sale Price: ${pred[0]:,.2f}")