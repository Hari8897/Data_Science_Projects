import React, { useState } from 'react';
import './App.css';
import axios from 'axios';


function FlightPricePredictor() {
  const [formData, setFormData] = useState({
    airline: '',
    source_city: '',
    departure_time: '',
    total_stops: '',
    arrival_time: '',
    destination_city: '',
    class: '',
    departure_date: ''
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData);
      if (response.status === 200) {
        setPrediction(response.data.predicted_price);
      } else {
        console.error('Error fetching prediction:', response.statusText);
      }
    } catch (error) {
      console.error('Error during API call:', error);
    }
  };

  return (
    <div className="App">
      <h1 className="text-3xl font-semibold text-blue-700 mb-8">Flight Price Predictor</h1>
      <form className="space-y-6 big-shadow p-6 bg-white rounded-lg" onSubmit={handleSubmit}>
        {/* Airline Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Airline</label>
          <select name="airline" value={formData.airline} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Airline</option>
            <option value="SpiceJet">SpiceJet</option>
            <option value="AirAsia">AirAsia</option>
            <option value="Vistara">Vistara</option>
            <option value="GO_FIRST">GO_FIRST</option>
            <option value="Indigo">Indigo</option>
            <option value="Air_India">Air India</option>
          </select>
        </div>

        {/* Source City Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Source City</label>
          <select name="source_city" value={formData.source_city} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Source City</option>
            <option value="Delhi">Delhi</option>
            <option value="Mumbai">Mumbai</option>
            <option value="Bangalore">Bangalore</option>
            <option value="Kolkata">Kolkata</option>
            <option value="Hyderabad">Hyderabad</option>
            <option value="Chennai">Chennai</option>
          </select>
        </div>

        {/* Departure Time Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Departure Time</label>
          <select name="departure_time" value={formData.departure_time} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Departure Time</option>
            <option value="Afternoon">Afternoon</option>
            <option value="Evening">Evening</option>
            <option value="Morning">Morning</option>
            <option value="Night">Night</option>
          </select>
        </div>

        {/* Number of Stops Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Number of Stops</label>
          <select name="total_stops" value={formData.total_stops} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Stops</option>
            <option value="zero">Zero</option>
            <option value="one">One</option>
            <option value="two_or_more">Two or More</option>
          </select>
        </div>

        {/* Arrival Time Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Arrival Time</label>
          <select name="arrival_time" value={formData.arrival_time} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Arrival Time</option>
            <option value="Night">Night</option>
            <option value="Morning">Morning</option>
            <option value="Early_Morning">Early Morning</option>
            <option value="Afternoon">Afternoon</option>
            <option value="Evening">Evening</option>
            <option value="Late_Night">Late Night</option>
          </select>
        </div>

        {/* Destination City Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Destination City</label>
          <select name="destination_city" value={formData.destination_city} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Destination City</option>
            <option value="Delhi">Delhi</option>
            <option value="Mumbai">Mumbai</option>
            <option value="Bangalore">Bangalore</option>
            <option value="Kolkata">Kolkata</option>
            <option value="Hyderabad">Hyderabad</option>
            <option value="Chennai">Chennai</option>
          </select>
        </div>

        {/* Class Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Class</label>
          <select name="class" value={formData.class} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required>
            <option value="">Select Class</option>
            <option value="Economy">Economy</option>
            <option value="Business">Business</option>
          </select>
        </div>

        {/* Departure Date Field */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="text-gray-700">Departure Date</label>
          <input type="date" name="departure_date" value={formData.departure_date} onChange={handleChange} className="mt-1 block w-full p-2 border border-gray-300 rounded-md" required />
        </div>

        <button type="submit" className="w-full bg-blue-600 text-white p-3 rounded-md hover:bg-blue-700 transition">
          Predict Price
        </button>
      </form>

      <div className="mt-6 text-center text-xl font-semibold">
        {prediction !== null && (
          <div>
            Predicted Flight Price: <span className="text-green-600">₹{prediction}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default FlightPricePredictor;
