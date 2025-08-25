import streamlit as st
import requests 

# # URL du serveur FastAPI
# SERVER_URL = "http://server:8000"  # En local ; remplacez par http://server:8000 en Docker Compose

# # Titre de l'application Streamlit
# st.title("Fruit Adder App")

# # Champ de saisie pour le nom du fruit
# fruit = st.text_input("Enter a fruit name:", "")

# # Bouton pour ajouter un fruit
# if st.button("Add Fruit"):
#     if fruit:
#         # Requête au serveur FastAPI
#         response = requests.get(f"{SERVER_URL}/add/{fruit}")
#         if response.status_code == 200:
#             data = response.json()  # Récupérer la réponse JSON
#             st.success(f"Successfully added {fruit} to the database!")
#         else:
#             st.error("Failed to add the fruit. Please check the server.")
#     else:
#         st.warning("Please enter a fruit name.")

# # Bouton pour lister tous les fruits
# if st.button("List All Fruits"):
#     # Requête au serveur FastAPI
#    req = requests.get("http://localhost:8000/list").json()["results"]
#    st.write(req)


# Define the FastAPI URL
FASTAPI_URL = "http://server:8000/predict/"

# Streamlit app title
st.title("Iris Flower Class Prediction")

# Create input fields for the user to enter features
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Prepare the data in the correct format for the API
data = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}

# When the "Predict" button is pressed, make the request to the FastAPI endpoint
if st.button("Predict"):
    # Send the data to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=data)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.write(f"Predicted class: {prediction}")
    else:
        st.error("Error in prediction request")

