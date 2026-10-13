/**
 * Where the live models run. Two deployments of the same three models exist:
 * - Streamlit (default): the page embeds the Streamlit app.
 * - Gradio on Hugging Face: the page calls its API with its own form. Set VITE_LIVE_MODE=gradio.
 */
export const LIVE_MODE = ((import.meta.env.VITE_LIVE_MODE as string | undefined) ?? 'streamlit') as 'streamlit' | 'gradio'

/** Streamlit Community Cloud app URL (chosen when deploying app/streamlit_app.py). */
export const STREAMLIT_URL = ((import.meta.env.VITE_STREAMLIT_URL as string | undefined) ?? 'https://cloudshield-live.streamlit.app').replace(/\/$/, '')

/** The Hugging Face Space that serves the Gradio app. VITE_SPACE can point at a local `python app.py` instead. */
export const SPACE = (import.meta.env.VITE_SPACE as string | undefined) ?? 'pranayraparla/Ml_Capstone'
export const SPACE_URL = SPACE.startsWith('http') ? SPACE : `https://huggingface.co/spaces/${SPACE}`
