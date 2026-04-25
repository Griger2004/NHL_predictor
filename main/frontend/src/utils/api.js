export const BASE_URL =
  import.meta.env.MODE === 'development'
    ? import.meta.env.VITE_API_URL
    : import.meta.env.VITE_API_URL_PROD
