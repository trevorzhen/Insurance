from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model('deployment_02082021')

def predict(model,input_df):
    predictions_df = predict_model(estimator=model, data = input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    st.title("Insurance Claim Prediction")

    from PIL import Image
    image = Image.open('medical.jpg')
    image_medical = Image.open('profile.jpg')

    st.image(image)

    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online","Batch"))

    st.sidebar.info("This app predicts patient medical expenses using Gradient Boosting Regressor.")
    st.sidebar.success("http://trevorzhen.com/")

    st.sidebar.image(image_medical)

    if add_selectbox == "Online":

        st.header("Please provide basic information of the patient:")

        sex = st.selectbox("Sex", ['male','female'])
        age = st.number_input("Age", min_value= 0 , max_value = 120, value = 30)

        if st.checkbox("Smoker"):
            smoker = "yes"
        else:
            smoker = "no"

        bmi = st.select_slider('BMI', options = range(10,51), value = 21)
        
        children = st.select_slider("Children", options = range(11), value = 3)
        
        region = st.selectbox("Region", ['southwest', 'northwest', 'northeast', 'southeast'])
        #region = st.select_slider("Region", ['southwest', 'northwest', 'northeast', 'southeast'])

        output = ""

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df= input_df)
            output = '$' + str(round(output,2))
        
        st.success('The expected amount of the claim is {}.'.format(output))

    if add_selectbox == "Batch":
        file_upload = st.file_uploader("Please upload the .csv file for predictions", type = ["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data = data)
            st.write(predictions)

if __name__ == '__main__':
    run()