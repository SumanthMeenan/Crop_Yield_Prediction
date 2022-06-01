import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 
app = Flask(__name__)
lr_saved_model = pickle.load(open("lrmodel.pkl", 'rb'))
svr_saved_model = pickle.load(open("svregressormodel.pkl", 'rb'))
flights = pd.read_excel("DATASET.xlsx", engine='openpyxl')

ip_feat = ['Year','State','Area(hect)']
dictionary1 = {'ANDAMAN & NICOBAR ': 0,'Andhra Pradesh': 1,'Arunachal Pradesh': 2,'Assam': 3,'Bihar': 4,'DADRA AND NAGAR': 5,'DAMAN AND DIU': 6,
                'Goa': 7,'Gujarat': 8,'Haryana': 9,'Himachal Pradesh': 10,'Jammu and Kashmir': 11,'KARNATAKA': 12,'KERALA': 13,'MADHYA PRADESH': 14,
                'MAHARASHTRA': 15,'MANIPUR': 16,'MEGHALAYA': 17,'MIZORAM': 18,'NAGALAND': 19,'ORISSA': 20,'PUNJAB': 21,'RAJASTHAN': 22,'SIKKIM': 23,
                'TAMILNADU': 24,'TRIPURA': 25,'UTTAR PRADESH': 26,'WEST BENGAL': 27}

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        list1 = [0]*27
        input_values = [int(i) for i in request.form.values()]
        print("input_values: ", input_values)
        year = input_values[1]
        area = input_values[2]
        if input_values[0] == 0:
            list1.extend(input_values[1:])
        else:
            list1[input_values[0] - 1] = 1
            list1.extend(input_values[1:])
        print("list1: ", list1)
        final_features = [np.array(list1)]
        print("final_features: ", final_features)
        prediction = lr_saved_model.predict(final_features)
        return render_template('index11.html', prediction_text='Total crop production in the year {0} in the area {1} hectares will be {2} Tons.'.format(year, area, prediction[0]))
    except:
        return render_template('index11.html', prediction_text='Error')

if __name__ == "__main__":
    app.run(debug=True)