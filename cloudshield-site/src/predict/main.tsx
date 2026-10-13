import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import PredictApp from './predict-app'
import StreamlitWindow from './streamlit-window'
import { LIVE_MODE } from './space'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {LIVE_MODE === 'gradio' ? <PredictApp /> : <StreamlitWindow />}
  </StrictMode>,
)
