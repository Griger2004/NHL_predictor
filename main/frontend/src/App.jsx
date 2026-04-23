import { useState, useEffect } from 'react'
import './App.css'
import GameCard from './GameCard'

export const BASE_URL = import.meta.env.MODE === "development"
  ? import.meta.env.VITE_API_URL
  : import.meta.env.VITE_API_URL_PROD;

console.log(BASE_URL)

function App() {
  const [games, setGames] = useState([])
  const [predictions, setPredictions] = useState([])
  const [loadingGames, setLoadingGames] = useState(false)
  const [loadingPredictions, setLoadingPredictions] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchGames()
  }, []) //will run once on mount

  const fetchGames = async () => {
    setError(null)
    setLoadingGames(true)
    try {
      const response = await fetch(BASE_URL + "/games")
      if (!response.ok) throw new Error("Failed to fetch games")
      const data = await response.json()
      setGames(data.games)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoadingGames(false)
    }
  }

  const fetchPrediction = async () => {
    setError(null)
    setLoadingPredictions(true)
    try {
      const response = await fetch(BASE_URL + "/predict")
      if (!response.ok) throw new Error("Failed to predict games")
      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoadingPredictions(false)
    }
  }

  return (
    <div>
      <h1>NHL Games <span style={{fontSize: '0.55em', fontWeight: 400, color: '#aaa', verticalAlign: 'middle'}}>{new Date().toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'})}</span></h1>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <ul>
        {games.map((game, index) => (
          <li key={index}>
            {game.away_team_name} ({game.away_team}) <b>@</b> {game.home_team_name} ({game.home_team})
            <span className='game_time'>
              {new Date(game.game_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </li>
        ))}
      </ul>
      <button
        onClick={fetchPrediction}
        className='generate_btn'
        disabled={loadingPredictions || loadingGames}
      >
        {loadingPredictions ? 'Predicting...' : 'Generate'}
      </button>
      {loadingPredictions && <div className='spinner' />}
      {predictions.length > 0 && (
        <div>
          <h2>Predictions</h2>
          <div className='prediction-notes'>
            <p className='prediction-note'>Not affected by live stats. Predictions are purely based on <span style={{color: '#4db6ac'}}>historical</span> data. Use for pre-game or start-game analysis.</p>
            <p className='prediction-note'>Note that pre-game predictions rely on the team's <span style={{color: '#4db6ac'}}>default</span> goalie which may not reflect the actual goalie for the game.</p>
            <p className='prediction-note'>Please allow until actual game <span style={{color: '#4db6ac'}}>start</span> to update correct goalie information.</p>
          </div>
          <ul>
            {predictions.map((pred, index) => (
              <GameCard key={index} prediction={pred} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;