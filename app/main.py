import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

def get_the_data():
    df = pd.read_csv('data/cancer_data.csv')
    df.drop(columns=['Unnamed: 32'], inplace=True)
    df.replace(to_replace={'diagnosis': {'M': 0, 'B': 1}}, inplace=True)
    
    # X = df.drop(columns=['diagnosis'])
    # y = df['diagnosis']

    return df

def get_scaled_value(input_data):
    data = get_the_data()

    X = data.drop(columns=['diagnosis'])

    scaled_dict = {}

    for key, value in input_data.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_value(input_data)

    categories = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dimension"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'],
        input_data['texture_mean'],
        input_data['perimeter_mean'],
        input_data['area_mean'],
        input_data['smoothness_mean'],
        input_data['compactness_mean'],
        input_data['concavity_mean'],
        input_data['concave points_mean'],
        input_data['symmetry_mean'],
        input_data['fractal_dimension_mean']],

        theta=categories,
        fill='toself',
        name='mean values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius_se'],
        input_data['texture_se'],
        input_data['perimeter_se'],
        input_data['area_se'],
        input_data['smoothness_se'],
        input_data['compactness_se'],
        input_data['concavity_se'],
        input_data['concave points_se'],
        input_data['symmetry_se'],
        input_data['fractal_dimension_se']
    ],
        theta=categories,
        fill='toself',
        name='standard error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius_worst'],
        input_data['texture_worst'],
        input_data['perimeter_worst'],
        input_data['area_worst'],
        input_data['smoothness_worst'],
        input_data['compactness_worst'],
        input_data['concavity_worst'],
        input_data['concave points_worst'],
        input_data['symmetry_worst'],
        input_data['fractal_dimension_worst']
    ],
        theta=categories,
        fill='toself',
        name='worst value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=False,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_predictions(input_data):
    # Load the model and the scaler
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    st.subheader('Cell Cluster Prediction')
    
    # Check the features in the input data
    input_features = list(input_data.keys())
    
    # Check the number of features scaler expects
    expected_features = scaler.n_features_in_
    
    # Prepare and scale the input data
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    if input_array.shape[1] != expected_features:
        st.write(f"Input array has {input_array.shape[1]} features, but the scaler expects {expected_features} features.")
        return
    
    input_array_scaled = scaler.transform(input_array)

    # Make predictions
    prediction = model.predict(input_array_scaled)

    # Display results
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

    # Display probabilities
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

    # Display disclaimer
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for professional medical advice.")

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_the_data()

    slider_label = [
    ["Mean Radius", "radius_mean"],
    ["Mean Texture", "texture_mean"],
    ["Mean Perimeter", "perimeter_mean"],
    ["Mean Area", "area_mean"],
    ["Mean Smoothness", "smoothness_mean"],
    ["Mean Compactness", "compactness_mean"],
    ["Mean Concavity", "concavity_mean"],
    ["Mean Concave Points", "concave points_mean"],
    ["Mean Symmetry", "symmetry_mean"],
    ["Mean Fractal Dimension", "fractal_dimension_mean"],
    ["SE Radius", "radius_se"],
    ["SE Texture", "texture_se"],
    ["SE Perimeter", "perimeter_se"],
    ["SE Area", "area_se"],
    ["SE Smoothness", "smoothness_se"],
    ["SE Compactness", "compactness_se"],
    ["SE Concavity", "concavity_se"],
    ["SE Concave Points", "concave points_se"],
    ["SE Symmetry", "symmetry_se"],
    ["SE Fractal Dimension", "fractal_dimension_se"],
    ["Worst Radius", "radius_worst"],
    ["Worst Texture", "texture_worst"],
    ["Worst Perimeter", "perimeter_worst"],
    ["Worst Area", "area_worst"],
    ["Worst Smoothness", "smoothness_worst"],
    ["Worst Compactness", "compactness_worst"],
    ["Worst Concavity", "concavity_worst"],
    ["Worst Concave Points", "concave points_worst"],
    ["Worst Symmetry", "symmetry_worst"],
    ["Worst Fractal Dimension", "fractal_dimension_worst"]
]

    input_dict = {'id': 1}

    for label, key in slider_label:
        input_dict[key]= st.sidebar.slider(
            label= label,
            min_value= float(0),
            max_value= float(data[key].max()),
            value= float(data[key].mean())
        )

    return input_dict

def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('This app uses a machine learning model to determine whether a breast cancer tumor is malignant or benign.')

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data=input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()
