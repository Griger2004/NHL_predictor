import { useState, useEffect } from 'react'
import './App.css'
import GameCard from './GameCard'

export const BASE_URL = import.meta.env.MODE === "development"
  ? import.meta.env.VITE_API_URL
  : import.meta.env.VITE_API_URL_PROD;

const todayStr = () => localDateStr(new Date())

const localDateStr = (d) => {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

const dateLabel = (dateStr) => {
  const today = localDateStr(new Date())
  const yesterday = localDateStr(new Date(Date.now() - 86400000))
  if (dateStr === today) return 'Today'
  if (dateStr === yesterday) return 'Yesterday'
  return new Date(dateStr + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function App() {
  const [games, setGames] = useState([])
  const [predictions, setPredictions] = useState([])
  const [predictionHistory, setPredictionHistory] = useState({}) // game_id → attempts[]
  const [hasGenerated, setHasGenerated] = useState(false)
  const [loadingGames, setLoadingGames] = useState(false)
  const [loadingPredictions, setLoadingPredictions] = useState(false)
  const [gamesError, setGamesError] = useState(null)
  const [predictionsError, setPredictionsError] = useState(null)

  useEffect(() => {
    fetchGames()
    fetchExistingPredictions()
  }, [])

  const fetchGames = async () => {
    setGamesError(null)
    setLoadingGames(true)
    try {
      const response = await fetch(BASE_URL + "/games")
      if (!response.ok) throw new Error("Failed to fetch games")
      const data = await response.json()
      setGames(data.games)
    } catch (err) {
      setGamesError(err instanceof TypeError ? "Unable to load games. Check your connection and try refreshing." : err.message)
    } finally {
      setLoadingGames(false)
    }
  }

  const buildHistoryMap = (histData) => {
    const byGame = {}
    histData.history.forEach(h => {
      if (!byGame[h.game_id]) byGame[h.game_id] = []
      byGame[h.game_id].push(h)
    })
    return byGame
  }

  const fetchExistingPredictions = async () => {
    try {
      const resp = await fetch(`${BASE_URL}/predictions/today?date=${todayStr()}`)
      if (!resp.ok) return
      const data = await resp.json()
      if (data.predictions?.length > 0) {
        setPredictions(data.predictions)
        const histResp = await fetch(`${BASE_URL}/predictions/history?date=${todayStr()}`)
        if (histResp.ok) {
          const histData = await histResp.json()
          setPredictionHistory(buildHistoryMap(histData))
        }
      }
    } catch {
      // non-critical — silently ignore
    }
  }

  const fetchPrediction = async () => {
    setPredictionsError(null)
    setLoadingPredictions(true)
    try {
      const response = await fetch(BASE_URL + "/predict")
      if (!response.ok) throw new Error("Failed to predict games")
      const data = await response.json()
      setPredictions(data.predictions)
      setHasGenerated(true)

      // Fetch prediction history to detect goalie changes
      const histResp = await fetch(`${BASE_URL}/predictions/history?date=${todayStr()}`)
      if (histResp.ok) {
        const histData = await histResp.json()
        setPredictionHistory(buildHistoryMap(histData))
      }
    } catch (err) {
      setPredictionsError(err instanceof TypeError ? "Unable to generate predictions. Check your connection and try again." : err.message)
    } finally {
      setLoadingPredictions(false)
    }
  }

  const gamesByDate = games.reduce((acc, g) => {
    ;(acc[g.date] ??= []).push(g)
    return acc
  }, {})

  const gameDates = Object.keys(gamesByDate).sort().reverse()

  const isGameFinished = (gameId) => {
    const liveGame = games.find(g => g.game_id === gameId)
    if (liveGame) return liveGame.game_state === 'FINAL' || liveGame.game_state === 'OFF'
    // Fall back to the stored game_state in the prediction (for page-load before games API responds)
    const pred = predictions.find(p => p.game_id === gameId)
    return pred?.game_state === 'FINAL' || pred?.game_state === 'OFF'
  }

  const finishedPredictions = predictions.filter(p => isGameFinished(p.game_id))
  const activePredictions = predictions.filter(p => !isGameFinished(p.game_id))

  const handleGenerateClick = () => {
    if (loadingPredictions || loadingGames) return
    fetchPrediction()
  }

  return (
    <div>
      <h1>
        NHL Games{' '}
        <span style={{ fontSize: '0.55em', fontWeight: 400, color: '#aaa', verticalAlign: 'middle' }}>
          {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </span>
      </h1>

      {gamesError && <div className='error-banner'>⚠ {gamesError}</div>}
      {gameDates.map(date => (
        <div key={date}>
          <h3 className='date-section-header'>{dateLabel(date)}</h3>
          <ul className={dateLabel(date) === 'Yesterday' ? 'games-yesterday' : 'games-today'}>
            {gamesByDate[date].map((game, i) => {
              const finished = game.game_state === 'FINAL' || game.game_state === 'OFF'
              return (
                <li key={i} className={finished ? 'game-finished' : ''}>
                  {game.away_team_name} ({game.away_team}) <b>@</b> {game.home_team_name} ({game.home_team})
                  <span className='game_time'>
                    {new Date(game.game_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
      {loadingGames && <div className='spinner' />}
      <button
        onClick={handleGenerateClick}
        className={`generate_btn${loadingPredictions ? ' predicting' : ''}`}
        disabled={loadingGames}
        aria-disabled={loadingGames || loadingPredictions}
      >
        {loadingPredictions ? <span className='predicting-text'>Predicting<span className='dots'><span>.</span><span>.</span><span>.</span></span></span> : 'Generate'}
      </button>
      {predictionsError && <div className='error-banner predictions-error-banner'>⚠ {predictionsError}</div>}

      {hasGenerated && activePredictions.length > 0 && (
        <div>
          <h2>Predictions</h2>
          <div className='prediction-notes'>
            <p className='prediction-note'>Not affected by live stats. Predictions are purely based on <span style={{ color: '#4db6ac' }}>historical</span> data. Use for pre-game or start-game analysis.</p>
            <p className='prediction-note'>Note that pre-game predictions rely on the team's <span style={{ color: '#4db6ac' }}>default</span> goalie which may not reflect the actual goalie for the game.</p>
            <p className='prediction-note'>Please allow until actual game <span style={{ color: '#4db6ac' }}>start</span> to update correct goalie information.</p>
          </div>
          <ul>
            {activePredictions.map((pred, index) => (
              <GameCard
                key={index}
                prediction={pred}
                gameStatus={games.find(g => g.game_id === pred.game_id)}
                history={predictionHistory[pred.game_id] || []}
              />
            ))}
          </ul>
        </div>
      )}

      {finishedPredictions.length > 0 && (
        <div>
          <h2>Results</h2>
          <ul>
            {finishedPredictions.map((pred, index) => (
              <GameCard
                key={index}
                prediction={pred}
                gameStatus={games.find(g => g.game_id === pred.game_id)}
                history={predictionHistory[pred.game_id] || []}
              />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
