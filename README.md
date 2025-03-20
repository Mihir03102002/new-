import streamlit as st
import joblib
import numpy as np

# Load the trained machine learning model
model = joblib.load('delivery_time_model.joblib')  # Replace with your model file

# Define the Streamlit app
def main():
    st.title("Order to Delivery Time Prediction")
    st.write("Enter the order details to get the predicted delivery time.")

    # Input fields for order details
    product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home & Kitchen", "Books", "Other"])
    customer_location = st.text_input("Customer Location")
    shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Same Day"])

    # Button to make prediction
    if st.button("Predict Delivery Time"):
        # Preprocess inputs and make prediction
        input_data = np.array([[product_category, customer_location, shipping_method]])
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"The predicted delivery time is {prediction[0]} days.")

if __name__ == "__main__":
    main()
