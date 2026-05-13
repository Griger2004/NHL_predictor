import { STATUS_LABELS } from '../../constants/gameStates'

function GameResultCard({ game }) {
  const winner = game.home_score > game.away_score
    ? game.home_team_full_name
    : game.away_team_full_name

  return (
    <li className='game-card finished'>
      <div className='scoreboard'>
        <div className='scoreboard-row'>
          <span className='away-team-text'>{game.away_team}</span>
          <span className='scoreboard-score'>{game.away_score}</span>
        </div>
        <div className='scoreboard-row'>
          <span className='home-team-text'>{game.home_team}</span>
          <span className='scoreboard-score'>{game.home_score}</span>
        </div>
      </div>
      <div className='matchup'>
        <span className='away-team-text'>{game.away_team}</span>{' '}
        <b>@</b>{' '}
        <span className='home-team-text'>{game.home_team}</span>
        <span className='game-status-badge'>
          {STATUS_LABELS[game.game_state] ?? game.game_state}
        </span>
      </div>
      <div className='result'>
        <span className='winner-label'>Winner: <strong>{winner}</strong></span>
        <span className='no-prediction-label'> · No prediction made</span>
      </div>
    </li>
  )
}

export default GameResultCard
