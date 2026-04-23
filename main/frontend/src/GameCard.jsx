function GameCard({ prediction: pred, gameStatus }) {
  const winner = pred.pred_home_win === 1 ? pred.home_team_name : pred.away_team_name
  const homeProb = (pred.prob_home_win * 100).toFixed(1)
  const awayProb = (pred.prob_away_win * 100).toFixed(1)
  const gameStarted = new Date() > new Date(pred.time)
  const goalieLabel = gameStarted ? 'Goalie' : 'Goalie (Default)'

  const isFinished = gameStatus?.game_state === 'FINAL' || gameStatus?.game_state === 'OFF'
  const actualWinner = isFinished
    ? (gameStatus.home_score > gameStatus.away_score ? pred.home_team_name : pred.away_team_name)
    : null
  const predCorrect = isFinished && winner === actualWinner

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
