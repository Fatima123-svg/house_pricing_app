import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

st.title("House Price Prediction")

# The Excel file must be in the same GitHub folder as this app.py
df = pd.read_excel("houseprice.xlsx")

st.subheader("Data preview")
st.dataframe(df.head())

# Train the model on Area + No of bedroms
X = df[["Area", "No of bedroms"]]
y = df["Price"]
model = LinearRegression()
model.fit(X, y)

# Charts
st.subheader("Area vs Price")
fig1, ax1 = plt.subplots()
ax1.scatter(df["Area"], df["Price"])
ax1.set_xlabel("Area")
ax1.set_ylabel("Price")
st.pyplot(fig1)

st.subheader("Bedrooms vs Price")
fig2, ax2 = plt.subplots()
ax2.scatter(df["No of bedroms"], df["Price"])
ax2.set_xlabel("No of bedrooms")
ax2.set_ylabel("Price")
st.pyplot(fig2)

# Prediction
st.subheader("Predict a price")
area = st.number_input("Area", value=float(df["Area"].median()))
bedrooms = st.number_input("Number of bedrooms", value=int(df["No of bedroms"].median()), step=1)

if st.button("Predict"):
    new_data = pd.DataFrame({"Area": [area], "No of bedroms": [bedrooms]})
    st.success(f"Predicted Price: {round(model.predict(new_data)[0], 2)}")
