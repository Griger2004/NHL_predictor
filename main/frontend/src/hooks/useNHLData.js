import { useState, useEffect } from 'react'
import { BASE_URL } from '../utils/api'
import { buildHistoryMap } from '../utils/gameUtils'
import { todayStr, yesterdayStr } from '../utils/dates'

const fetchAllHistory = async (today, yesterday) => {
  const [todayResp, yestResp] = await Promise.all([
    fetch(`${BASE_URL}/predictions/history?date=${today}`),
    fetch(`${BASE_URL}/predictions/history?date=${yesterday}`),
  ])
  let history = []
  if (todayResp.ok) { const d = await todayResp.json(); history = [...history, ...(d.history || [])] }
  if (yestResp.ok) { const d = await yestResp.json(); history = [...history, ...(d.history || [])] }
  return history
}

export function useNHLData() {
  const [games, setGames] = useState([])
  const [predictions, setPredictions] = useState([])
  const [yesterdayPredictions, setYesterdayPredictions] = useState([])
  const [predictionHistory, setPredictionHistory] = useState({})
  const [hasGenerated, setHasGenerated] = useState(false)
  const [resultsTab, setResultsTab] = useState(null)
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
      const response = await fetch(`${BASE_URL}/games`)
      if (!response.ok) throw new Error('Failed to fetch games')
      const data = await response.json()
      setGames(data.games)
    } catch (err) {
      setGamesError(
        err instanceof TypeError
          ? 'Unable to load games. Check your connection and try refreshing.'
          : err.message
      )
    } finally {
      setLoadingGames(false)
    }
  }

  const fetchExistingPredictions = async () => {
    try {
      const today = todayStr()
      const yesterday = yesterdayStr()

      const [todayResp, yestResp] = await Promise.all([
        fetch(`${BASE_URL}/predictions/today?date=${today}`),
        fetch(`${BASE_URL}/predictions/today?date=${yesterday}`),
      ])

      let hasPredictions = false
      if (todayResp.ok) {
        const d = await todayResp.json()
        if (d.predictions?.length > 0) { setPredictions(d.predictions); hasPredictions = true }
      }
      if (yestResp.ok) {
        const d = await yestResp.json()
        if (d.predictions?.length > 0) { setYesterdayPredictions(d.predictions); hasPredictions = true }
      }

      if (hasPredictions) {
        const history = await fetchAllHistory(today, yesterday)
        if (history.length > 0) setPredictionHistory(buildHistoryMap(history))
      }
    } catch {
      // non-critical — silently ignore
    }
  }

  const fetchPrediction = async () => {
    setPredictionsError(null)
    setLoadingPredictions(true)
    try {
      const response = await fetch(`${BASE_URL}/predict`)
      if (!response.ok) throw new Error('Failed to predict games')
      const data = await response.json()
      setPredictions(data.predictions)
      setHasGenerated(true)
      setResultsTab(null)

      try {
        const today = todayStr()
        const yesterday = yesterdayStr()
        const history = await fetchAllHistory(today, yesterday)
        if (history.length > 0) setPredictionHistory(buildHistoryMap(history))
      } catch {
        // Best effort to update history: prediction generation already succeeded, so silently ignore any errors here
      }
    } catch (err) {
      setPredictionsError(
        err instanceof TypeError
          ? 'Unable to generate predictions. Check your connection and try again.'
          : err.message
      )
    } finally {
      setLoadingPredictions(false)
    }
  }

  return {
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
  }
}
