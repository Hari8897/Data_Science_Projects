import logo from './logo.svg';
import './App.css';

function App() {

  

  return (
    // Flight price prediction app main container
    <div className="container">
      <form className="flight-form">
        <h2>Flight Price Prediction</h2>
        <div className="form-group">
          <label htmlFor="airline">Airline:</label>
          <select id="airline" name="airline">
            <option value="Air India">Air India</option>
            <option value="IndiGo">IndiGo</option>
            <option value="Jet Airways">Jet Airways</option>
            <option value="SpiceJet">SpiceJet</option>
            <option value="Vistara">Vistara</option>
            <option value="GoAir">GoAir</option>
            <option value="Multiple carriers">Multiple carriers</option>
            <option value="Air Asia">Air Asia</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="source">Source City:</label>
          <select id="source" name="source">
            <option value="Delhi">Delhi</option>
            <option value="Kolkata">Kolkata</option>
            <option value="Mumbai">Mumbai</option>
            <option value="Chennai">Chennai</option>
            <option value="Bangalore">Bangalore</option>
            <option value="Hyderabad">Hyderabad</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="departure-time">Departure Time:</label>
          <select id="departure-time" name="departure-time">
            <option value="Early_Morning">Early Morning</option>  
            <option value="Morning">Morning</option>
            <option value="Afternoon">Afternoon</option>
            <option value="Evening">Evening</option>
            <option value="Night">Night</option>
            <option value="Late_Night">Late Night</option>
          </select>
        </div>  
        <div className="form-group">
          <label htmlFor="destination">Destination City:</label>
          <select id="destination" name="destination">
            <option value="Delhi">Delhi</option>
            <option value="Kolkata">Kolkata</option>
            <option value="Mumbai">Mumbai</option>
            <option value="Chennai">Chennai</option>
            <option value="Bangalore">Bangalore</option>
            <option value="Hyderabad">Hyderabad</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="arrival-time">Arrival Time:</label>
          <select id="departure-time" name="departure-time">
            <option value="Early_Morning">Early Morning</option>  
            <option value="Morning">Morning</option>
            <option value="Afternoon">Afternoon</option>
            <option value="Evening">Evening</option>
            <option value="Night">Night</option>
            <option value="Late_Night">Late Night</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="date">Date of Journey:</label>
          <input type="date" id="date" name="date" required />
        </div>
        <div className="form-group">
          <label htmlFor="stops">Total Stops:</label>
          <select id="stops" name="stops">
            <option value='0'>Zero</option>
            <option value='1'>One</option>
            <option value='2'>Two</option>
            
          </select>
        </div>
        <button type="submit" className="predict-button">Predict Price</button> 

      </form>

    </div>
      
    
  );
}

export default App;
