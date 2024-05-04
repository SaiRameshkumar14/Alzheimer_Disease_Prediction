from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib
import tensorflow
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow_addons.metrics import F1Score
import numpy as np
import io
import keras
import keras.utils
from keras import utils as np_utils

app = Flask(__name__)

#########  HOME  #########################################################################################

@app.route('/')
def home():
    return render_template('home.html')

######### END ############################################################################################



#########  MRI  #########################################################################################

# Load your custom Inception model

CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
custom_inception_model = keras.models.load_model("alzheimer_cnn_model", custom_objects={'F1Score': F1Score})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded image
        img = request.files['image']

        # Read image data as bytes
        img_bytes = img.read()

        # Convert bytes to image
        img = image.load_img(io.BytesIO(img_bytes), target_size=(176, 176))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = custom_inception_model.predict(img_array)
        
        # Get the top-4 predicted classes
        top4_classes = np.argsort(predictions[0])[-4:][::-1]

        # Prepare the prediction result string
        result_str = "<h2>Prediction:</h2>"
        for i, class_index in enumerate(top4_classes):
            class_name = CLASSES[class_index]
            score = predictions[0][class_index]
            result_str += f"<p>{i + 1}: {class_name} ({score:.2f})</p>"

        return result_str

# Define route for rendering the HTML page
@app.route('/mri')
def mri():
    return render_template('mri.html')




# CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
# custom_inception_model = keras.models.load_model("alzheimer_cnn_model", custom_objects={'F1Score': F1Score})

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get uploaded image
#         img = request.files['image']

#         # Read image data as bytes
#         img_bytes = img.read()

#         # Convert bytes to image
#         img = image.load_img(io.BytesIO(img_bytes), target_size=(176, 176))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)

#         # Make predictions
#         predictions = custom_inception_model.predict(img_array)
        
#         # Get the top-4 predicted classes
#         top4_classes = np.argsort(predictions[0])[-4:][::-1]

#         # Prepare the prediction result
#         results_m = []
#         for i, class_index in enumerate(top4_classes):
#             class_name = CLASSES[class_index]
#             score = predictions[0][class_index] # Convert to float
#             results_m += ({'rank': i + 1, 'class_name': class_name, 'score:.2f': score})

#         return results_m

# @app.route('/mri')
# def mri():
#     return render_template('mri.html')


######### END ############################################################################################



#########  REPORT  #########################################################################################


# Load the model from the joblib file
with open('report_model.pkl', 'rb') as f:
    model_r = joblib.load(f)

@app.route('/report', methods=['GET', 'POST'])
def report():
    prediction_r = None
    
    if request.method == 'POST':
        # Get user input from the form
        mf = request.form['mf']
        educ = request.form['educ']
        ses = request.form['ses']
        mmse = request.form['mmse']
        cdr = request.form['cdr']
        etiv = request.form['etiv']
        nwbv = request.form['nwbv']
        asf = request.form['asf']
        age_category = request.form['age_category']

        # Create a DataFrame with the user input
        user_data_r = pd.DataFrame({
            'Gender_(M/F)': [mf],
            'EDUC': [educ],
            'SES': [ses],
            'MMSE': [mmse],
            'CDR': [cdr],
            'eTIV': [etiv],
            'nWBV': [nwbv],
            'ASF': [asf],
            'Age_Category': [age_category]
        })

        # Make prediction_rs using the loaded model
        prediction_r = model_r.predict(user_data_r)

    # Render the template with the prediction_r
    return render_template('report.html', prediction_r=prediction_r)

######### END ############################################################################################



#########  Behaviour  #########################################################################################


with open('behavioral_model.pkl', 'rb') as f:
    model_b = joblib.load(f)

@app.route('/behaviour', methods=['GET', 'POST'])
def behaviour():
    prediction_b = None

    if request.method == 'POST':
        # Get user input from the form
        date_forget = request.form['date_forget']
        face_recall = request.form['face_recall']
        task_planning = request.form['task_planning']
        word_expression = request.form['word_expression']
        mood_shift = request.form['mood_shift']
        adaptability = request.form['adaptability']
        solution_finding = request.form['solution_finding']
        sleep_patterns = request.form['sleep_patterns']
        appetite_changes = request.form['appetite_changes']
        interest_enthusiasm = request.form['interest_enthusiasm']
        financial_matters = request.form['financial_matters']
        tech_adaptation = request.form['tech_adaptation']
        independence_decrease = request.form['independence_decrease']
        home_navigation = request.form['home_navigation']

        # Create a DataFrame with the user input
        user_data_b = pd.DataFrame({
            'Date_Forget': [date_forget],
            'Face_Recall': [face_recall],
            'Task_Planning': [task_planning],
            'Word_Expression': [word_expression],
            'Mood_Shift': [mood_shift],
            'Adaptability': [adaptability],
            'Solution_Finding': [solution_finding],
            'Sleep_Patterns': [sleep_patterns],
            'Appetite_Changes': [appetite_changes],
            'Interest_Enthusiasm': [interest_enthusiasm],
            'Financial_Matters': [financial_matters],
            'Tech_Adaptation': [tech_adaptation],
            'Independence_Decrease': [independence_decrease],
            'Home_Navigation': [home_navigation]
        })

        # Make predictions using the loaded model
        prediction_b = model_b.predict(user_data_b)

    # Render the template with the prediction
    return render_template('behaviour.html', prediction_b=prediction_b)

######### END ############################################################################################



#########  Analysis  #########################################################################################

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

######### END ############################################################################################


if (__name__) == '__main__':
    app.run(debug=True)