import numpy as np
from flask import Flask,request,render_template
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model = load_model('model.h5')
sc=pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_scrap =[x for x in request.form.values()]
    extract=int_scrap[0].split()
    floatt=[float(x) for x in extract]
    float_features=np.array(floatt).reshape(1,-1)
    standard_data=sc.transform(float_features.reshape(1,-1))
    predict=model.predict(standard_data)

    if predict[0][0]==1:
        output="Malignant"
    else:
        output="Benign"



    return render_template('index.html', prediction='From above examined features cancer is of type {}'.format(output))


if __name__ == "__main__":
    app.run('127.0.0.1',port=5000)


