const MS_PER_DAY = 86_400_000

export const localDateStr = (d) => {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

export const todayStr = () => localDateStr(new Date())
export const yesterdayStr = () => localDateStr(new Date(Date.now() - MS_PER_DAY))

export const dateLabel = (dateStr) => {
  const today = todayStr()
  const yesterday = yesterdayStr()
  if (dateStr === today) return 'Today'
  if (dateStr === yesterday) return 'Yesterday'
  return new Date(`${dateStr}T12:00:00`).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
