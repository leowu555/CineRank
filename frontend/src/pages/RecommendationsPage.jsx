import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { API_BASE_URL, authHeaders } from '../api'
import { useAuth } from '../context/AuthContext'
import { usePageTitle } from '../hooks/usePageTitle'
import './moviePages.css'

function RecommendationsPage() {
  const { token, isAuthenticated, loading: authLoading } = useAuth()
  const navigate = useNavigate()
  const [results, setResults] = useState([])
  const [method, setMethod] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)

  usePageTitle('Recommendations')

  useEffect(() => {
    async function load() {
      if (!token) return
      setLoading(true)
      try {
        const res = await fetch(`${API_BASE_URL}/api/recommendations/`, {
          headers: authHeaders(token),
        })
        const data = await res.json()
        setResults(data.results || [])
        setMethod(data.method || '')
        setMessage(data.message || '')
      } catch {
        setMessage('Could not load recommendations.')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [token])

  if (authLoading) return <div className="page"><p>Loading...</p></div>
  if (!isAuthenticated) return <Navigate to="/login" replace />

  return (
    <div className="page">
      <section className="page-intro">
        <h1 className="page-title">Recommended for you</h1>
        <p className="page-intro-text">
          Unseen titles from other people&apos;s ratings. If overlap is too thin,
          the list falls back to titles with the most ratings.
        </p>
        {method && (
          <p className="method-badge">Method: {method.replaceAll('_', ' ')}</p>
        )}
      </section>

      {loading && <p>Loading recommendations...</p>}
      {!loading && message && <p className="page-intro-text">{message}</p>}

      {!loading && results.length > 0 && (
        <ul className="results-grid">
          {results.map((movie, index) => (
            <li
              key={movie.movie_id}
              className="poster-card movie-card-enter"
              style={{ animationDelay: `${Math.min(index, 8) * 0.06}s` }}
            >
              <button
                type="button"
                className="poster-card-hit"
                onClick={() => navigate(`/movie/${movie.movie_id}`)}
                aria-label={`View ${movie.title}`}
              >
                <div className="poster-card-media">
                  {movie.poster_url ? (
                    <img src={movie.poster_url} alt="" />
                  ) : (
                    <div className="poster-card-empty">No poster</div>
                  )}
                  <div className="poster-card-shade">
                    <h2 className="poster-card-title">{movie.title}</h2>
                    {movie.reason && <p className="poster-card-meta">{movie.reason}</p>}
                  </div>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}

      {!loading && results.length === 0 && !message && (
        <div className="empty-state">
          <p>No unseen titles are left to recommend from current ratings.</p>
          <Link to="/" className="btn btn-primary empty-state-btn">
            Search movies
          </Link>
        </div>
      )}
    </div>
  )
}

export default RecommendationsPage
