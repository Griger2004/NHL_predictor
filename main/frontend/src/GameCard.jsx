function GameCard({ prediction: pred, gameStatus }) {
  const homeFullName = gameStatus?.home_team_full_name || pred.home_team_name
  const awayFullName = gameStatus?.away_team_full_name || pred.away_team_name
  const winner = pred.pred_home_win === 1 ? homeFullName : awayFullName
  const homeProb = (pred.prob_home_win * 100).toFixed(1)
  const awayProb = (pred.prob_away_win * 100).toFixed(1)
  const gameStarted = new Date() > new Date(pred.time)
  const goalieLabel = gameStarted ? 'Goalie' : 'Goalie (Default)'

  const isFinished = gameStatus?.game_state === 'FINAL' || gameStatus?.game_state === 'OFF'
  const isLive = gameStatus?.game_state === 'LIVE' || gameStatus?.game_state === 'CRIT'
  const actualWinner = isFinished
    ? (gameStatus.home_score > gameStatus.away_score ? homeFullName : awayFullName)
    : null
  const predCorrect = isFinished && winner === actualWinner

  const statusLabel = {
    FUT: 'Upcoming', PRE: 'Pre-Game', LIVE: 'Live', CRIT: 'Live',
    FINAL: 'Final', OFF: 'Final',
  }[gameStatus?.game_state] ?? null

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
        <span className='away-team-text'>{pred.away_team}</span> <b>@</b> <span className='home-team-text'>{pred.home_team}</span>
        {statusLabel && (
          <span className={`game-status-badge${isLive ? ' live' : ''}`}>
            {isLive && <span className='live-dot' />}
            {statusLabel}
          </span>
        )}
      </div>
      <div className='goalies'>
        <span>Away {goalieLabel}: <strong>{pred.away_goalie}</strong></span>
        <span>Home {goalieLabel}: <strong>{pred.home_goalie}</strong></span>
      </div>
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
