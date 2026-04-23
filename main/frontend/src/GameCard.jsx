function GameCard({ prediction: pred }) {
  const winner = pred.pred_home_win === 1 ? pred.home_team_name : pred.away_team_name
  const homeProb = (pred.prob_home_win * 100).toFixed(1)
  const awayProb = (pred.prob_away_win * 100).toFixed(1)

  return (
    <li className='game-card'>
      <div className='matchup'>
        {pred.away_team} ({pred.away_goalie}) <b>@</b> {pred.home_team} ({pred.home_goalie})
      </div>
      <div className='result'>
        <span className='winner-label'>Predicted winner: {winner}</span>
        {' | '}
        <span>Confidence: {(pred.confidence * 100).toFixed(1)}%</span>
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
