import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <footer className='app-footer'>
      <span>&copy; {new Date().getFullYear()} Gal Riger</span>
    </footer>
  </StrictMode>,
)
