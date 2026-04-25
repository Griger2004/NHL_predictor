import { FINISHED_STATES, LIVE_STATES, STATUS_LABELS } from '../../constants/gameStates'
import { getGoalieChanges } from '../../utils/gameUtils'

function GameCard({ prediction: pred, gameStatus, history = [] }) {
  const homeFullName = gameStatus?.home_team_full_name || pred.home_team_name
  const awayFullName = gameStatus?.away_team_full_name || pred.away_team_name
  const winner = pred.pred_home_win === 1 ? homeFullName : awayFullName
  const homeProb = (pred.prob_home_win * 100).toFixed(1)
  const awayProb = (pred.prob_away_win * 100).toFixed(1)

  const gameStarted = new Date() > new Date(pred.time)
  const goalieLabel = gameStarted ? 'Goalie' : 'Goalie (unconfirmed)'

  const state = gameStatus?.game_state
  const isFinished = FINISHED_STATES.has(state)
  const isLive = LIVE_STATES.has(state)

  const actualWinner = isFinished
    ? (gameStatus.home_score > gameStatus.away_score ? homeFullName : awayFullName)
    : null
  const predCorrect = isFinished && winner === actualWinner
  const statusLabel = STATUS_LABELS[state] ?? null

  const goalieChanges = getGoalieChanges(history, pred.home_team, pred.away_team)

  return (
    <li className={`game-card${isFinished ? ' finished' : ''}${predCorrect ? ' correct' : ''}`}>
      {isFinished && (
        <div className='scoreboard'>
          <div className='scoreboard-row'>
            <span className='away-team-text'>{pred.away_team}</span>
            <span className='scoreboard-score'>{gameStatus.away_score}</span>
          </div>
          <div className='scoreboard-row'>
            <span className='home-team-text'>{pred.home_team}</span>
            <span className='scoreboard-score'>{gameStatus.home_score}</span>
          </div>
        </div>
      )}
      <div className='matchup'>
        <span className='away-team-text'>{pred.away_team}</span>{' '}
        <b>@</b>{' '}
        <span className='home-team-text'>{pred.home_team}</span>
        {statusLabel && (
          <span className={`game-status-badge${isLive ? ' live' : ''}`}>
            {isLive && <span className='live-dot' />}
            {statusLabel}
          </span>
        )}
        {goalieChanges.length > 0 && (
          <span className='repredicted-badge'>Goalies updated</span>
        )}
      </div>
      <div className='goalies'>
        <span>Away {goalieLabel}: <strong>{pred.away_goalie}</strong></span>
        <span>Home {goalieLabel}: <strong>{pred.home_goalie}</strong></span>
      </div>
      {goalieChanges.length > 0 && (
        <div className='goalie-changes'>
          {goalieChanges.map((c, i) => (
            <span key={i} className='goalie-change-row'>
              {c.side}: <span className='goalie-old'>{c.from}</span> → <span className='goalie-new'>{c.to}</span>
            </span>
          ))}
        </div>
      )}
      <div className='result'>
        <span className='winner-label'>Predicted winner: <strong>{winner}</strong></span>
        {isFinished && (
          <span className='actual-winner-label'> | Actual winner: <strong>{actualWinner}</strong></span>
        )}
      </div>
      <div className='prob-bar'>
        <span className='prob-bar-away' style={{ width: `${awayProb}%` }}>
          {pred.away_team}: {awayProb}%
        </span>
        <span className='prob-bar-home' style={{ width: `${homeProb}%` }}>
          {pred.home_team}: {homeProb}%
        </span>
      </div>
    </li>
  )
}

export default GameCard
