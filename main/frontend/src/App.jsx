import './App.css'
import GameCard from './components/GameCard/GameCard'
import { useNHLData } from './hooks/useNHLData'
import { todayStr, yesterdayStr, dateLabel } from './utils/dates'
import { FINISHED_STATES } from './constants/gameStates'

function App() {
  const {
    games,
    predictions,
    yesterdayPredictions,
    predictionHistory,
    hasGenerated,
    resultsTab,
    setResultsTab,
    loadingGames,
    loadingPredictions,
    gamesError,
    predictionsError,
    fetchPrediction,
  } = useNHLData()

  const today = todayStr()
  const yesterday = yesterdayStr()

  const gamesByDate = games.reduce((acc, g) => {
    ;(acc[g.date] ??= []).push(g)
    return acc
  }, {})
  const gameDates = Object.keys(gamesByDate).sort().reverse()

  const allPredictions = [...predictions, ...yesterdayPredictions]

  const isGameFinished = (gameId) => {
    const liveGame = games.find(g => g.game_id === gameId)
    if (liveGame) return FINISHED_STATES.has(liveGame.game_state)
    const pred = allPredictions.find(p => p.game_id === gameId)
    return FINISHED_STATES.has(pred?.game_state)
  }

  const finishedPredictions = allPredictions.filter(p => isGameFinished(p.game_id))
  const activePredictions = allPredictions.filter(p => !isGameFinished(p.game_id))

  const hasAnyTodayFinished = finishedPredictions.some(p => p.date === today)
  const hasAnyYesterdayFinished = finishedPredictions.some(p => p.date === yesterday)
  const activeResultsTab = resultsTab ?? (hasAnyTodayFinished ? 'today' : 'yesterday')
  const visibleResults = finishedPredictions.filter(p =>
    p.date === (activeResultsTab === 'today' ? today : yesterday)
  )

  const handleGenerateClick = () => {
    if (loadingPredictions || loadingGames) return
    fetchPrediction()
  }

  return (
    <div>
      <h1>
        NHL Games{' '}
        <span className='page-title-date'>
          {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </span>
      </h1>

      {gamesError && <div className='error-banner'>⚠ {gamesError}</div>}
      {gameDates.map(date => (
        <div key={date}>
          <h3 className='date-section-header'>{dateLabel(date)}</h3>
          <ul className={dateLabel(date) === 'Yesterday' ? 'games-yesterday' : 'games-today'}>
            {gamesByDate[date].map(game => {
              const finished = FINISHED_STATES.has(game.game_state)
              return (
                <li key={game.game_id} className={finished ? 'game-finished' : ''}>
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
        disabled={loadingGames || loadingPredictions}
        aria-disabled={loadingGames || loadingPredictions}
      >
        {loadingPredictions
          ? <span className='predicting-text'>Predicting<span className='dots'><span>.</span><span>.</span><span>.</span></span></span>
          : 'Generate'
        }
      </button>

      {predictionsError && <div className='error-banner predictions-error-banner'>⚠ {predictionsError}</div>}

      {activePredictions.length > 0 && (hasGenerated || activePredictions.some(p => p.date < today)) && (
        <div>
          <h2>Predictions</h2>
          <div className='prediction-notes'>
            <p className='prediction-note'>Not affected by live stats. Predictions are purely based on <span className='accent'>historical</span> data. Use for pre-game or start-game analysis.</p>
            <p className='prediction-note'>Note that pre-game predictions rely on the team's <span className='accent'>default</span> goalie which may not reflect the actual goalie for the game.</p>
            <p className='prediction-note'>Please allow until actual game <span className='accent'>start</span> to update correct goalie information.</p>
            <p className='prediction-note'>Although shown, model is not tuned for playoff scenarios. Mainly use this application for <span className='accent'>regular</span> season predictions.</p>
          </div>
          {Object.entries(
            activePredictions.reduce((acc, p) => { ;(acc[p.date] ??= []).push(p); return acc }, {})
          ).sort(([a], [b]) => b.localeCompare(a)).map(([date, preds]) => (
            <div key={date} className={date === yesterday ? 'date-group yesterday-group' : 'date-group'}>
              {activePredictions.some(p => p.date !== preds[0].date) && (
                <h3 className='date-section-header'>{dateLabel(date) === 'Yesterday' ? 'Yesterday (live)' : 'Today'}</h3>
              )}
              <ul>
                {preds.map(pred => (
                  <GameCard
                    key={pred.game_id}
                    prediction={pred}
                    gameStatus={games.find(g => g.game_id === pred.game_id)}
                    history={predictionHistory[pred.game_id] || []}
                  />
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

      {finishedPredictions.length > 0 && (
        <div className='results-section'>
          <div className='results-nav'>
            <button
              className='results-nav-arrow'
              onClick={() => setResultsTab('yesterday')}
              disabled={activeResultsTab === 'yesterday' || !hasAnyYesterdayFinished}
              aria-label="Yesterday's results"
            >&lt;</button>
            <h2>Results</h2>
            <button
              className='results-nav-arrow'
              onClick={() => setResultsTab('today')}
              disabled={activeResultsTab === 'today'}
              aria-label="Today's results"
            >&gt;</button>
          </div>
          <p className='results-tab-label'>{activeResultsTab === 'today' ? 'Today' : 'Yesterday'}</p>
          <ul>
            {visibleResults.length > 0
              ? visibleResults.map(pred => (
                  <GameCard
                    key={pred.game_id}
                    prediction={pred}
                    gameStatus={games.find(g => g.game_id === pred.game_id)}
                    history={predictionHistory[pred.game_id] || []}
                  />
                ))
              : <li>{activeResultsTab === 'today' ? 'No games have finished yet.' : 'No results available.'}</li>
            }
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
