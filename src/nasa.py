import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

df = pd.read_csv('book1.csv')
label_encoder = LabelEncoder()

label_encoder_gender = LabelEncoder().fit(df['gender'])
label_encoder_marital_status = LabelEncoder().fit(df['marital_status'])
label_encoder_generation = LabelEncoder().fit(df['generation'])
label_encoder_city = LabelEncoder().fit(df['city'])

def safe_encode_value(label_encoder, value, default=-1):
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        print(f"Unseen category: '{value}' assigned default value {default}")
        return default

def encode_new_inputs(gender, marital_status, generation, city, tenant_id):
    return [
        safe_encode_value(label_encoder_city, city),
        safe_encode_value(label_encoder_generation, generation),
        safe_encode_value(label_encoder_gender, gender),
        safe_encode_value(label_encoder_marital_status, marital_status),
        tenant_id
    ]

df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
df['marital_status_encoded'] = label_encoder.fit_transform(df['marital_status'])
df['generation_encoded'] = label_encoder.fit_transform(df['generation'])
df['city_encoded'] = label_encoder.fit_transform(df['city'])

x= df[['city_encoded','generation_encoded','gender_encoded','marital_status_encoded', 'tenant_id']]
y=df['model_id']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
kFold = KFold(n_splits=10,shuffle= False)
x_1= np.array(x)
x_1
y_1= np.array(y)
for train_index,test_index in kFold.split(x):
    X_train, X_test, y_train, y_test = x_1[train_index], x_1[test_index],y_1[train_index], y_1[test_index]   
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    required_keys = ['city', 'generation', 'gender', 'marital_status', 'tenant_id']
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return jsonify({'error': f'Missing keys in request data: {", ".join(missing_keys)}'}), 400
    input_data = encode_new_inputs(data['city'], data['generation'], data['gender'], data['marital_status'], data['tenant_id'])
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = rf_model.predict(input_data_reshaped)
    model_usage = df['model_id'].value_counts().get(int(prediction[0]), 0)
    
    return jsonify({
    'recommended_model_id': int(prediction[0]),
    'recommended_model_usage_count': int(model_usage),
    'accuracy_score' : float(accuracy * 100)
})

if __name__ == '__main__':
    y_1 = np.array(y)
    for train_index, test_index in kFold.split(x):
        X_train, X_test, y_train, y_test = x_1[train_index], x_1[test_index], y_1[train_index], y_1[test_index]
        rf_model.fit(X_train, y_train)

    app.run(debug=True)


# input_data = encode_new_inputs("Male", "Married", "Millennial", "Wimbledon", "12500")
# input_data_as_numpy_array=np.asarray(input_data)
# input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# prediction=rf_model.predict(input_data_reshaped)
# print('The predicted model is', prediction[0])
# print('This model is used by', df['model_id'].value_counts().get(prediction[0], 0))

