export const GAME_STATE = {
  FUT: 'FUT',
  PRE: 'PRE',
  LIVE: 'LIVE',
  CRIT: 'CRIT',
  FINAL: 'FINAL',
  OFF: 'OFF',
}

export const FINISHED_STATES = new Set([GAME_STATE.FINAL, GAME_STATE.OFF])
export const LIVE_STATES = new Set([GAME_STATE.LIVE, GAME_STATE.CRIT])

export const STATUS_LABELS = {
  [GAME_STATE.FUT]: 'Upcoming',
  [GAME_STATE.PRE]: 'Pre-Game',
  [GAME_STATE.LIVE]: 'Live',
  [GAME_STATE.CRIT]: 'Live',
  [GAME_STATE.FINAL]: 'Final',
  [GAME_STATE.OFF]: 'Final',
}
