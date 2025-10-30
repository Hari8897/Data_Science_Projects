from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)


# Enable CORS for all routes
CORS(app)
# Load the trained model
with open('flight_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Categorical features used in the model
airline_dict = {'SpiceJet': 0, 'AirAsia': 1, 'Vistara': 2, 'GO_FIRST': 3, 'Indigo': 4, 'Air_India': 5},
source_dict = {'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2, 'Kolkata': 3, 'Hyderabad': 4, 'Chennai': 5},
departure_dict = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5},
stops_dict = {'zero': 0, 'one': 1, 'two_or_more': 2},
arrival_dict = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5},
destination_dict = {'Mumbai': 0, 'Bangalore': 1, 'Kolkata': 2, 'Hyderabad': 3, 'Chennai': 4, 'Delhi': 5},
class_dict = {'Economy': 0, 'Business': 1}



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert categorical features to numerical using the dictionaries
    try:
        input_data = pd.DataFrame({
            'airline': [airline_dict[data['airline']]],
            'source_city': [source_dict[data['source_city']]],
            'departure_time': [departure_dict[data['departure_time']]],
            'stops': [stops_dict[data['stops']]],
            'arrival_time': [arrival_dict[data['arrival_time']]],
            'destination_city': [destination_dict[data['destination_city']]],
            'class': [class_dict[data['class']]],
            'duration': [data['duration']],
            'total_stops': [data['total_stops']]
        })
    except KeyError as e:
        return jsonify({'error': f'Invalid category value: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    # Make prediction
    prediction = model.predict(input_data)
    return jsonify({'predicted_price': prediction[0]})
if __name__ == '__main__':
    app.run(debug=True)