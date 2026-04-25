export const buildHistoryMap = (history) => {
  const byGame = {}
  history.forEach((h) => {
    if (!byGame[h.game_id]) byGame[h.game_id] = []
    byGame[h.game_id].push(h)
  })
  return byGame
}

export const getGoalieChanges = (history, homeTeam, awayTeam) => {
  if (history.length <= 1) return []
  const changes = []
  for (let i = 1; i < history.length; i++) {
    const prev = history[i - 1]
    const curr = history[i]
    if (prev.home_goalie !== curr.home_goalie)
      changes.push({ side: homeTeam, from: prev.home_goalie, to: curr.home_goalie })
    if (prev.away_goalie !== curr.away_goalie)
      changes.push({ side: awayTeam, from: prev.away_goalie, to: curr.away_goalie })
  }
  return changes
}
