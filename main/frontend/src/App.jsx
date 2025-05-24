import { useState, useEffect } from 'react'
import './App.css'

export const BASE_URL = import.meta.env.MODE === "development"
  ? import.meta.env.VITE_API_URL
  : import.meta.env.VITE_API_URL_PROD;

console.log(BASE_URL)

function App() {
  const [games, setGames] = useState([])
  const [predictions, setPredictions] = useState([])

  useEffect(() => {
    fetchGames()
  }, []) //will run once on mount

  const fetchGames = async () => {
    try {
      console.log(BASE_URL)
      const response = await fetch(BASE_URL + "/games")
      if (!response.ok) throw new Error("Failed to fetch games")
      
      const data = await response.json()
      setGames(data.games) 
      console.log(data.games)
    } catch (error) {
      console.error("Error fetching games:", error)
    }
  }

  const fetchPrediction = async () => {
    try {
      const response = await fetch(BASE_URL + "/predict")
      if (!response.ok) throw new Error("Failed to predict games")
      
      const data = await response.json()
      setPredictions(data.predictions) 
      console.log(data.predictions)
    } catch (error) {
      console.error("Error predicting games:", error)
    }
  }

  return (
    <div>
      <h1>NHL Games</h1>
      <div className='home_away_label'>
        <h3>Home</h3>
        <h3>Away</h3>
      </div>
      <ul>
        {games.map((game, index) => (
          <li key={index}>{game.home} <b>vs</b> {game.away}</li>
        ))}
      </ul>
      <button onClick={fetchPrediction} className='generate_btn'>Generate</button>
      {predictions.length > 0 && (
        <div>
          <h2>Predictions</h2>
          <ul>
            {predictions.map((pred, index) => (
              <li key={index}>
                {pred.home} ({Number(pred.probabilities[1].toFixed(3))}) <b>vs</b> {pred.away} ({Number(pred.probabilities[0].toFixed(3))}) 
                - <b>Predicted Winner:</b> {pred.prediction}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;