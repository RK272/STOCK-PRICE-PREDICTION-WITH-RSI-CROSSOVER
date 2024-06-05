from src.exception import SensorException
import os,sys
import pandas as pd
import numpy as np
from src.constant.training_pipeline import SAVED_MODEL_DIR
from src.utils.main_utils import  load_object
from src.logger import logging
from src.pipeline import training_pipeline
from src.pipeline.training_pipeline import TrainPipeline
import os
from wsgiref import simple_server
from flask import Flask, request, render_template,jsonify
from flask import Response
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from src.ml.model.estimator import ModelResolver

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index1.html')


"""@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient(self,modelevalutionartifact: ModelEvaluationArtifact):



    try:
        data=request.get_json()

        datetime = request.form['datetime']
        open_value = float(request.form['open'])
        high_value = float(request.form['high'])
        low_value = float(request.form['low'])
        close_value = float(request.form['close'])
        print(close_value)


        datetime = data['datetime']
        open_value = float(data['open'])
        high_value = float(data['high'])
        low_value = float(data['low'])
        close_value = float(data['close'])
        print(close_value)


        data_dict = {
            'datetime': [datetime],
            'open': [open_value],
            'high': [high_value],
            'low': [low_value],
            'close': [close_value]
        }
        df = pd.DataFrame(data_dict)
        self.modelevalutionartifact = modelevalutionartifact
        best_modelpath = self.modelevalutionartifact.best_model_path
        latest_model = load_object(file_path=best_modelpath)

        y_trained_pred = latest_model.predict(df)
        print("ronni")
        print(y_trained_pred)
        return jsonify({'prediction_result': 'Prediction successful!'})


    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)"""


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json()

        RSI_Crossed_40 = data['RSI_Crossed_40']
        rsi9m = data['rsi9m']
        EMA20 = data['EMA20']
        EMA5 = data['EMA5']

        # Printing the values for demonstration purposes
        print(RSI_Crossed_40)

        data_dict = {

            'RSI_Crossed_40': [RSI_Crossed_40],
            'rsi9m': [rsi9m],
            'EMA20': [EMA20],
            'EMA5': [EMA5]
        }
        df = pd.DataFrame(data_dict)
        print(df)
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("model is not available")
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        # print(model)
        # model=load_object(file_path='saved_models / 1707204331 / model.pkl')

        # model=model_pred()
        # df = df.drop(columns=['open'], axis=1)
        train_arr = np.c_[df]
        print("kokila")
        print(train_arr)
        print("kokila")
        print("rooo")
        pr = model.predict(train_arr)
        print("prediction result : if 0 f or :1 :s ", pr)
        prediction_result = pr.tolist()

        return jsonify({'prediction_result': prediction_result})

        # Perform your prediction logic here

        # Return a response (you can customize this based on your needs)
        #return jsonify({'prediction_result': 'Pr'})

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)}), 500


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRouteClient():
    try:


        # fyersdataconfig=fyersdataconfig()

        # a = fyersdatageneration(fyersdataconfig)
        # a.fyersinitiation()
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

    except Exception as e:
        print(e)
        logging.exception(e)


port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()







