function GameCard({ prediction: pred }) {
  const winner = pred.pred_home_win === 1 ? pred.home_team_name : pred.away_team_name
  const homeProb = (pred.prob_home_win * 100).toFixed(1)
  const awayProb = (pred.prob_away_win * 100).toFixed(1)
  const gameStarted = new Date() > new Date(pred.time)
  const goalieLabel = gameStarted ? 'Goalie' : 'Goalie (Default)'

  return (
    <li className='game-card'>
      <div className='matchup'>
        <span className='away-team-text'>{pred.away_team}</span> <b>@</b> <span className='home-team-text'>{pred.home_team}</span>
      </div>
      <div className='goalies'>
        <span>Away {goalieLabel}: <strong>{pred.away_goalie}</strong></span>
        <span>Home {goalieLabel}: <strong>{pred.home_goalie}</strong></span>
      </div>
      <div className='result'>
        <span className='winner-label'>Predicted winner: {winner}</span>
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
