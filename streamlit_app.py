## Import libraries

import streamlit as st
import pandas as pd
import joblib
import base64


## Page configuration

st.set_page_config(
    page_title="Dr CirrhoRisk",
    page_icon=":material/health_and_safety:",
    layout="wide"
)


## Page styling

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F6F8FB;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    h1 {
        color: #003F6B;
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 0px;
    }

    p {
        color: #374A5E;
        font-size: 17px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #C5CFDA;
        border-left: 5px solid #004A78;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    label {
        color: #263A4D !important;
        font-weight: 650 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #F0F3F6;
        border-radius: 5px;
    }

    div[data-baseweb="select"] > div {
        background-color: #F0F3F6;
        border-radius: 5px;
    }

    .stButton > button {
        width: 100%;
        background-color: #004A78;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-size: 17px;
        font-weight: 700;
    }

    div.stButton > button {
        background-color: #005B8F;
        border: none;
        border-radius: 10px;
        padding: 12px 22px;
        width: auto;
        padding: 10px 24px;
    }

    div.stButton > button p {
        color: white !important;
        font-size: 17px;
        font-weight: 700;
    }

    .sample-heading {
        color: #52667A;
        font-size: 14px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    div.stButton > button[kind="secondary"] {
        width: 90%;
        min-height: 30px;
        background-color: #005B8F;
        color: white;
        border: 1px solid #005B8F;
        border-radius: 8px;
        padding: 7px 12px;
        font-size: 14px;
        font-weight: 650;
        transition: 0.15s ease;
    }

    div.stButton > button[kind="secondary"]:hover {
        background-color: #004A78;
        border-color: #004A78;
        color: white;
        transform: translateY(-1px);
    }

    div.stButton > button[kind="secondary"] p {
        color: white !important;
        font-size: 14px;
        font-weight: 650;
    }

    div.stButton > button[kind="primary"] {
        width: 100%;
        min-height: 48px;
        background-color: #004A78;
        border: none;
        border-radius: 8px;
        font-size: 17px;
        font-weight: 700;

    [data-testid="stSidebar"] {
        background-color: #EAF0F6;
        border-right: 1px solid #C5CFDA;
    }

    /* Sidebar spacing */
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }

    /* Sidebar headings */
    [data-testid="stSidebar"] h3 {
        color: #003F6B;
        font-size: 19px;
    }

    /* Sidebar normal text */
    [data-testid="stSidebar"] p {
        color: #374A5E;
        font-size: 14px;
        line-height: 1.5;
    }
    }

    </style>
    """,
    unsafe_allow_html=True
)

with open("dna_background.jpg", "rb") as image_file:
    background_image = base64.b64encode(
        image_file.read()
    ).decode()

st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(
                rgba(246, 248, 251, 0.88),
                rgba(246, 248, 251, 0.88)
            ),
            url("data:image/jpg;base64,%s");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    </style>
    """ % background_image,
    unsafe_allow_html=True
)


with st.sidebar:

    st.markdown(
        """
        <h2 style="color:#003F6B; margin-bottom:5px;">
            Dr CirrhoRisk
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "This is a machine learning tool that estimates whether "
        "a cirrhosis patient matches a higher or lower death risk. " 
        "The data provided is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984."
        
    )

    st.divider()

    st.markdown("### How to use")

    st.write(
        """
        1. Enter the patient's information.
        2. Key in clinical and laboratory results.
        3. Click **Predict Death**.
        4. Review the predicted class and model confidence.
        """
    )

    st.divider()

    st.markdown("### Prediction outcome")

    st.write(
        """
        **Class 0:** No death recorded during the research period.

        **Class 1:** Death recorded during the research period.
        """
    )

    st.divider()

    st.warning(
        "This prototype is designed to support hospital staff in monitoring patient data and identifying patients who " \
        "may require closer medical attention. It should not replace professional clinical judgement."
    )

    st.caption(
        "Model: Random Forest Classifier"
    )

model = joblib.load("cirrhosis_final_rf_model.pkl")

st.title("Cirrhosis Patient Death Risk Prediction")

st.write(
    "This is a training model that acts as an initial screening tool to identify whether the patient is at higher death risk can help the" 
    "hospital to gather resources, provide necessary measures and also to ensure that no patients at risks are being overlooked"
)


def load_sample(sample):

    for key, value in sample.items():
        st.session_state[key] = value

default_sample = {
    "age_selected": 54.2,
    "sex_selected": "Female",
    "drug_selected": "D-penicillamine",
    "stage_selected": 3,
    "ascites_selected": "No",
    "hepatomegaly_selected": "No",
    "spiders_selected": "No",
    "edema_selected": "No Edema",
    "bilirubin_selected": 1.5,
    "cholesterol_selected": 261,
    "albumin_selected": 3.50,
    "copper_selected": 156,
    "alk_phos_selected": 1718.0,
    "sgot_selected": 137.9,
    "tryglicerides_selected": 172,
    "platelets_selected": 304,
    "prothrombin_selected": 10.8
}


lower_risk_sample = {
    "age_selected": 45.2,
    "sex_selected": "Female",
    "drug_selected": "D-penicillamine",
    "stage_selected": 2,
    "ascites_selected": "No",
    "hepatomegaly_selected": "No",
    "spiders_selected": "No",
    "edema_selected": "No Edema",
    "bilirubin_selected": 0.8,
    "cholesterol_selected": 250,
    "albumin_selected": 4.10,
    "copper_selected": 40,
    "alk_phos_selected": 900.0,
    "sgot_selected": 70.0,
    "tryglicerides_selected": 90,
    "platelets_selected": 350,
    "prothrombin_selected": 10.0
}


higher_risk_sample = {
    "age_selected": 68.0,
    "sex_selected": "Male",
    "drug_selected": "Placebo",
    "stage_selected": 4,
    "ascites_selected": "Yes",
    "hepatomegaly_selected": "Yes",
    "spiders_selected": "Yes",
    "edema_selected": "Edema Present",
    "bilirubin_selected": 12.0,
    "cholesterol_selected": 600,
    "albumin_selected": 2.40,
    "copper_selected": 300,
    "alk_phos_selected": 5000.0,
    "sgot_selected": 250.0,
    "tryglicerides_selected": 250,
    "platelets_selected": 110,
    "prothrombin_selected": 14.0
}

button_col1, button_col2, button_col3, empty_col = st.columns(
    [1.35, 1.20, 0.55, 5],
    gap="xxsmall"
)

with button_col1:
    st.button(
        "No-Death Sample",
        on_click=load_sample,
        args=(lower_risk_sample,),
        use_container_width=True,
        type="secondary"
    )

with button_col2:
    st.button(
        "Death Sample",
        on_click=load_sample,
        args=(higher_risk_sample,),
        use_container_width=True,
        type="secondary"
    )

with button_col3:
    st.button(
        "Reset",
        on_click=load_sample,
        args=(default_sample,),
        use_container_width=True,
        type="secondary"
    )

## Demographics

with st.container(border=True):

    st.markdown(
        "<h3 style='color:#003F6B;'>Patient Profile</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age_selected = st.number_input(
            "Age (years)",
            min_value=26.3,
            max_value=78.5,
            value=54.2,
            step=0.1,
            key="age_selected"
        )

    with col2:

        sex_selected = st.selectbox(
            "Sex",
            ["Female", "Male"],
            key="sex_selected"
        )

    with col3:

        stage_selected = st.number_input(
            "Stage",
            min_value=1,
            max_value=4,
            value=3,
            step=1,
            key="stage_selected"
        )


## Clinical observations

with st.container(border=True):

    st.markdown(
        "<h3 style='color:#003F6B;'>Clinical Findings & Treatments</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        ascites_selected = st.selectbox(
            "Ascites",
            ["No", "Yes"],
            key="ascites_selected"
        )

    with col2:

        hepatomegaly_selected = st.selectbox(
            "Hepatomegaly",
            ["No", "Yes"],
            key="hepatomegaly_selected"

        )

    with col3:

        spiders_selected = st.selectbox(
            "Spiders",
            ["No", "Yes"],
            key="spiders_selected"

        )

    with col4:

        edema_selected = st.selectbox(
            "Edema",
            [
                "No Edema",
                "Edema Controlled",
                "Edema Present"
            ],
            key="edema_selected"

        )

    with col5:

        drug_selected = st.selectbox(
            "Drug",
            [
                "D-penicillamine",
                "Placebo"
            ],
            key="drug_selected"
        )


## Laboratory measurements

with st.container(border=True):

    st.markdown(
        "<h3 style='color:#003F6B;'>Laboratory Test Results</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        bilirubin_selected = st.number_input(
            "Bilirubin (mg/dL)",
            min_value=0.3,
            max_value=28.0,
            value=1.5,
            step=0.1,
            key="bilirubin_selected"

        )

        copper_selected = st.number_input(
            "Copper (µg/day)",
            min_value=4,
            max_value=588,
            value=156,
            step=1,
            key="copper_selected"


        )

        tryglicerides_selected = st.number_input(
            "Triglycerides (mg/dL)",
            min_value=33,
            max_value=598,
            value=172,
            step=1,
            key="tryglicerides_selected"

        )

    with col2:

        cholesterol_selected = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=120,
            max_value=1775,
            value=261,
            step=1,
            key="cholesterol_selected"

        )

        alk_phos_selected = st.number_input(
            "Alkaline Phosphatase (U/L)",
            min_value=289.0,
            max_value=13862.4,
            value=1718.0,
            step=1.0,
            key="alk_phos_selected"

        )

        platelets_selected = st.number_input(
            "Platelets (k/mm³)",
            min_value=62,
            max_value=563,
            value=304,
            step=1,
            key="platelets_selected"

        )

    with col3:

        albumin_selected = st.number_input(
            "Albumin (g/dL)",
            min_value=1.96,
            max_value=4.64,
            value=3.50,
            step=0.01,
            key="albumin_selected"

        )

        sgot_selected = st.number_input(
            "SGOT (U/mL)",
            min_value=26.3,
            max_value=457.3,
            value=137.9,
            step=0.1,
            key="sgot_selected"

        )

        prothrombin_selected = st.number_input(
            "Prothrombin (seconds)",
            min_value=9.0,
            max_value=17.1,
            value=10.8,
            step=0.1,
            key="prothrombin_selected"
        )


## Predict button


if st.button(
    "Predict Death",
    type="primary",
    use_container_width=True
):

    ## Apply the same one-hot encoding used during training

    if drug_selected == "Placebo":
        drug_placebo = 1
    else:
        drug_placebo = 0

    if sex_selected == "Male":
        sex_m = 1
    else:
        sex_m = 0

    if ascites_selected == "Yes":
        ascites_y = 1
    else:
        ascites_y = 0

    if hepatomegaly_selected == "Yes":
        hepatomegaly_y = 1
    else:
        hepatomegaly_y = 0

    if spiders_selected == "Yes":
        spiders_y = 1
    else:
        spiders_y = 0

    if edema_selected == "Edema Controlled":
        edema_s = 1
    else:
        edema_s = 0

    if edema_selected == "Edema Present":
        edema_y = 1
    else:
        edema_y = 0


    ## Convert input data to a DataFrame

    df_input = pd.DataFrame({

        "Age": [age_selected],
        "Bilirubin": [bilirubin_selected],
        "Cholesterol": [cholesterol_selected],
        "Albumin": [albumin_selected],
        "Copper": [copper_selected],
        "Alk_Phos": [alk_phos_selected],
        "SGOT": [sgot_selected],
        "Tryglicerides": [tryglicerides_selected],
        "Platelets": [platelets_selected],
        "Prothrombin": [prothrombin_selected],
        "Stage": [stage_selected],
        "Drug_Placebo": [drug_placebo],
        "Sex_M": [sex_m],
        "Ascites_Y": [ascites_y],
        "Hepatomegaly_Y": [hepatomegaly_y],
        "Spiders_Y": [spiders_y],
        "Edema_S": [edema_s],
        "Edema_Y": [edema_y]
    })


    ## Predict mortality risk

    ## Predict mortality risk

    y_unseen_pred = model.predict(df_input)[0]


    ## Obtain prediction probabilities

    prediction_probability = model.predict_proba(df_input)[0]


    ## Find the probability for the predicted class

    predicted_class_position = list(model.classes_).index(y_unseen_pred)

    model_confidence = (
        prediction_probability[predicted_class_position] * 100
    )

    ## Display prediction

    if y_unseen_pred == 0:

        st.success(
            "No death recorded during the period of research."
        )

        st.write(
            f"Model confidence: {model_confidence:.1f}%"
        )

        st.write(
            "The patient's measurements were classified as Class 0."
        )

    else:

        st.error(
            "Death recorded during the time of research."
        )

        st.write(
            f"Model confidence: {model_confidence:.1f}%"
        )

        st.write(
            "The patient's measurements were classified as Class 1."
        )