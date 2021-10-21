from logging import debug
from flask import Flask, request, jsonify
import pickle
from model_deployment.ml_model import predict_mpg

app = Flask('app')

@app.route('/predict', methods=['POST'])
def predict():
    vehicle = request.get_json()
    print(vehicle)
    with open('./model_deployment/final_model_mpg_prediction.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle,model)

    result = {
        'mpg_prediction' : list(predictions)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)