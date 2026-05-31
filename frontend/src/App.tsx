import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import ChatBot, { type Flow, type Settings, type Styles } from 'react-chatbotify'
import bamboo from './assets/bamboo.jpg'
import blackFluff from './assets/black-fluff.png'
import botAvatar from './assets/panda-bot-avatar.png'
import pandaCatalogMark from './assets/panda-catalog.png'
import pandaConnectMark from './assets/panda-connect-logo.png'
import userAvatar from './assets/panda-user-avatar.png'
import whiteFluff from './assets/white-fluff.png'
import './App.css'

type ChatResponse = {
  session_id: string
  intent: string
  response: string
  data: Record<string, unknown> | null
}

type MatchRow = {
  focal_panda_name?: string
  candidate_panda_name?: string
  recommendation_rank_v2?: number
  recommendation_rank?: number
  age_gap_years?: number
  top_positive_factors?: string
  focal_photo_url?: string | null
  candidate_photo_url?: string | null
  candidate_city_region?: string | null
  candidate_country?: string | null
  focal_city_region?: string | null
  focal_country?: string | null
  bio_component?: number
  behavior_component?: number
  candidate_panda_babies?: number
}

type PandaCatalogItem = {
  source_id: string
  name: string
  sex: string | null
  age_years: number | null
  zoo_or_facility: string | null
  city_region: string | null
  country: string | null
  status: string | null
  photo_url: string | null
  photo_source_url: string | null
}

type CatalogFilters = {
  query: string
  sex: string
  age: string
  location: string
}

const ageRanges = [
  { label: 'All ages', value: 'all' },
  { label: '0-4', value: '0-4' },
  { label: '5-9', value: '5-9' },
  { label: '10-14', value: '10-14' },
  { label: '15-19', value: '15-19' },
  { label: '20+', value: '20-plus' },
]

function getLocationText(panda: PandaCatalogItem) {
  return [panda.city_region, panda.country].filter(Boolean).join(', ')
}

function formatSex(sex: string | null) {
  if (!sex) return null
  return sex.charAt(0).toUpperCase() + sex.slice(1).toLowerCase()
}

function matchesAgeRange(age: number | null, range: string) {
  if (range === 'all') return true
  if (age === null) return false
  if (range === '20-plus') return age >= 20
  const [min, max] = range.split('-').map(Number)
  return age >= min && age <= max
}

function splitFactors(value: string | undefined) {
  return (value ?? '')
    .split(';')
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean)
}

function matchRank(row: MatchRow, fallbackRank: number) {
  return row.recommendation_rank_v2 ?? row.recommendation_rank ?? fallbackRank
}

function shortTransferLine(row: MatchRow) {
  const focal = row.focal_city_region ?? row.focal_country
  const candidate = row.candidate_city_region ?? row.candidate_country
  if (!focal || !candidate) return null
  if (focal === candidate) {
    return `Both are in ${focal}, so transfer looks straightforward`
  }
  return `${focal} and ${candidate} are manageable for a transfer`
}

function compatibilityBullets(row: MatchRow) {
  const bullets: string[] = []
  const factors = splitFactors(row.top_positive_factors)

  if (factors.includes('low combined health risk')) {
    bullets.push('Both pandas appear healthy')
  }
  if (factors.includes('high logistical feasibility')) {
    const transfer = shortTransferLine(row)
    if (transfer) bullets.push(transfer)
  }
  if (typeof row.age_gap_years === 'number') {
    bullets.push(`Age gap is workable (${Math.round(row.age_gap_years)} years)`)
  }
  if (typeof row.bio_component === 'number') {
    if (row.bio_component >= 0.8) {
      bullets.push('Strong biological compatibility')
    } else if (row.bio_component >= 0.68) {
      bullets.push('Solid biological compatibility')
    }
  }
  if (typeof row.behavior_component === 'number' && row.behavior_component >= 0.62) {
    bullets.push('Behavior styles look compatible')
  }
  if (row.candidate_panda_babies === 0) {
    bullets.push('No recorded cubs yet, which helps breeding potential')
  }

  return Array.from(new Set(bullets)).slice(0, 3)
}

function parseCardReasoning(response: string) {
  const map = new Map<string, string[]>()
  const blocks = response.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean)

  for (const block of blocks) {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean)
    if (lines.length < 2) continue
    const heading = lines[0]
    const bullets = lines
      .filter((line) => line.startsWith('- '))
      .map((line) => line.replace(/^- /, '').trim())
      .filter(Boolean)

    if (bullets.length === 0) continue
    if (heading.endsWith(':')) {
      map.set(heading.slice(0, -1).trim(), bullets)
    }
  }

  return map
}

function parseBestOverallReasoning(response: string) {
  const blocks = response.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean)
  for (const block of blocks) {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean)
    if (lines[0] === 'Why this pair stands out:') {
      return lines
        .filter((line) => line.startsWith('- '))
        .map((line) => line.replace(/^- /, '').trim())
        .filter(Boolean)
    }
  }
  return []
}

function MatchCardsMessage({
  heading,
  matches,
  reasoningByName,
  accentTop = true,
  showPairImages = false,
}: {
  heading: string
  matches: MatchRow[]
  reasoningByName?: Map<string, string[]>
  accentTop?: boolean
  showPairImages?: boolean
}) {
  return (
    <section className="match-cards-message" aria-label={heading}>
      <div className="match-cards-heading">{heading}</div>
      <div className="match-cards-grid">
        {matches.map((row, index) => {
          const name = row.candidate_panda_name ?? `Match ${index + 1}`
          const rank = matchRank(row, index + 1)
          const bullets = reasoningByName?.get(name) ?? compatibilityBullets(row)
          const isTop = accentTop && index === 0
          return (
            <article
              className={`match-result-card${isTop ? ' match-result-card--top' : ''}`}
              key={`${name}-${rank}-${index}`}
            >
              <span className="match-rank-badge">#{rank}</span>
              {showPairImages && row.focal_panda_name ? (
                <div className="match-pair-images">
                  <div className="match-image-frame">
                    {row.focal_photo_url ? (
                      <img
                        className="match-image"
                        src={row.focal_photo_url}
                        alt={row.focal_panda_name}
                        loading="lazy"
                      />
                    ) : (
                      <div className="match-image match-image--placeholder" aria-hidden="true">
                        <span>{row.focal_panda_name.slice(0, 1)}</span>
                      </div>
                    )}
                  </div>
                  <div className="match-image-frame">
                    {row.candidate_photo_url ? (
                      <img className="match-image" src={row.candidate_photo_url} alt={name} loading="lazy" />
                    ) : (
                      <div className="match-image match-image--placeholder" aria-hidden="true">
                        <span>{name.slice(0, 1)}</span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="match-image-frame">
                  {row.candidate_photo_url ? (
                    <img className="match-image" src={row.candidate_photo_url} alt={name} loading="lazy" />
                  ) : (
                    <div className="match-image match-image--placeholder" aria-hidden="true">
                      <span>{name.slice(0, 1)}</span>
                    </div>
                  )}
                </div>
              )}
              <div className="match-card-body">
                <h3>{name}</h3>
                <ul>
                  {bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}

function buildStructuredResponse(payload: ChatResponse): ReactElement | null {
  const data = payload.data ?? {}

  if (payload.intent === 'top_matches') {
    const matches = Array.isArray(data.matches) ? (data.matches as MatchRow[]) : []
    const focalName =
      (typeof data.panda_name === 'string' && data.panda_name)
      || matches[0]?.focal_panda_name
      || 'Top panda matches'
    if (matches.length > 0) {
      return (
        <MatchCardsMessage
          heading={`${focalName}'s top matches`}
          matches={matches}
          reasoningByName={parseCardReasoning(payload.response)}
        />
      )
    }
  }

  if (payload.intent === 'best_overall') {
    const bestMatch = data.best_match as MatchRow | undefined
    if (bestMatch) {
      const focalName = bestMatch.focal_panda_name ?? 'Best overall match'
      const reasoning = parseBestOverallReasoning(payload.response)
      const reasoningByName = new Map<string, string[]>()
      if (bestMatch.candidate_panda_name && reasoning.length > 0) {
        reasoningByName.set(bestMatch.candidate_panda_name, reasoning)
      }
      return (
        <MatchCardsMessage
          heading={`Best overall match for ${focalName}`}
          matches={[bestMatch]}
          reasoningByName={reasoningByName}
          showPairImages
        />
      )
    }
  }

  if (payload.intent === 'compare_candidates') {
    const candidateA = data.candidate_a as MatchRow | undefined
    const candidateB = data.candidate_b as MatchRow | undefined
    const focalName = typeof data.focal_panda_name === 'string' ? data.focal_panda_name : ''
    const matches = [candidateA, candidateB].filter(Boolean) as MatchRow[]
    if (matches.length > 0) {
      const heading = focalName
        ? `Best comparison for ${focalName}`
        : 'Candidate comparison'
      return (
        <MatchCardsMessage
          heading={heading}
          matches={matches}
          reasoningByName={parseCardReasoning(payload.response)}
        />
      )
    }
  }

  return null
}

function PandaCatalog() {
  const [pandas, setPandas] = useState<PandaCatalogItem[]>([])
  const [filters, setFilters] = useState<CatalogFilters>({
    query: '',
    sex: 'all',
    age: 'all',
    location: 'all',
  })
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading')

  useEffect(() => {
    let cancelled = false

    async function loadPandas() {
      try {
        const response = await fetch('/pandas')
        if (!response.ok) {
          throw new Error('Failed to load panda catalog')
        }
        const payload = (await response.json()) as PandaCatalogItem[]
        if (!cancelled) {
          setPandas(payload)
          setLoadState('ready')
        }
      } catch {
        if (!cancelled) {
          setLoadState('error')
        }
      }
    }

    void loadPandas()

    return () => {
      cancelled = true
    }
  }, [])

  const countries = useMemo(
    () =>
      Array.from(new Set(pandas.map((panda) => panda.country).filter(Boolean))).sort() as string[],
    [pandas],
  )

  const filteredPandas = useMemo(() => {
    const query = filters.query.trim().toLowerCase()

    return pandas.filter((panda) => {
      const location = getLocationText(panda).toLowerCase()
      const name = panda.name.toLowerCase()
      const sex = panda.sex?.toLowerCase() ?? ''

      return (
        (!query || name.includes(query)) &&
        (filters.sex === 'all' || sex === filters.sex) &&
        matchesAgeRange(panda.age_years, filters.age) &&
        (filters.location === 'all' || panda.country === filters.location || location.includes(filters.location.toLowerCase()))
      )
    })
  }, [filters, pandas])

  return (
    <aside className="panda-catalog" aria-label="Panda catalog">
      <div className="catalog-heading">
        <img
          className="catalog-header-logo"
          src={pandaCatalogMark}
          alt="Panda Catalog"
        />
        <p className="catalog-count">{filteredPandas.length} of {pandas.length || 140}</p>
      </div>

      <div className="catalog-controls">
        <label className="catalog-search">
          <span>Name</span>
          <input
            type="search"
            placeholder="Search panda..."
            value={filters.query}
            onChange={(event) => setFilters((current) => ({ ...current, query: event.target.value }))}
          />
        </label>

        <div className="catalog-filter-grid">
          <label>
            <span>Sex</span>
            <select
              value={filters.sex}
              onChange={(event) => setFilters((current) => ({ ...current, sex: event.target.value }))}
            >
              <option value="all">All</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
            </select>
          </label>

          <label>
            <span>Age</span>
            <select
              value={filters.age}
              onChange={(event) => setFilters((current) => ({ ...current, age: event.target.value }))}
            >
              {ageRanges.map((range) => (
                <option key={range.value} value={range.value}>{range.label}</option>
              ))}
            </select>
          </label>

          <label>
            <span>Location</span>
            <select
              value={filters.location}
              onChange={(event) => setFilters((current) => ({ ...current, location: event.target.value }))}
            >
              <option value="all">All</option>
              {countries.map((country) => (
                <option key={country} value={country}>{country}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="catalog-list">
        {loadState === 'loading' && <p className="catalog-status">Loading panda photos...</p>}
        {loadState === 'error' && <p className="catalog-status">Could not load the panda catalog.</p>}
        {loadState === 'ready' && filteredPandas.length === 0 && (
          <p className="catalog-status">No pandas match those filters.</p>
        )}
        {filteredPandas.map((panda) => (
          <article className="panda-card" key={panda.source_id}>
            {panda.photo_url && <img src={panda.photo_url} alt={panda.name} loading="lazy" />}
            <div className="panda-card-body">
              <h3>{panda.name}</h3>
              <p>{[formatSex(panda.sex), panda.age_years === null ? null : `${panda.age_years} years`].filter(Boolean).join(', ')}</p>
              <p>{getLocationText(panda) || 'Location unknown'}</p>
            </div>
          </article>
        ))}
      </div>
    </aside>
  )
}

function App() {
  const sessionId = useRef<string | null>(null)
  const pendingStructuredResponse = useRef<ReactElement | null>(null)
  const isLearnMorePage = window.location.pathname === '/learn-more'

  const flow = useMemo<Flow>(
    () => ({
      start: {
        message:
          'Hello I am your panda matching assistant!\n\n'
          + 'Here are some questions I can help you with:\n'
          + '- best overall panda match\n'
          + '- top 5 matches for Ai Bao\n'
          + '- who is the best match for Bao Li\n'
          + '- why is Xi Lan Ai Bao\'s first top match\n'
          + '- does Er Shun have any cubs\n'
          + '- who is the oldest panda in the dataset',
        path: 'chat',
      },
      chat: {
        message: async (params) => {
          const message = params.userInput.trim()

          if (!message) {
            return 'Ask about panda matches when you are ready.'
          }

          try {
            const response = await fetch('/agent/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                message,
                session_id: sessionId.current,
              }),
            })

            const payload = (await response.json()) as ChatResponse | { detail?: string }

            if (!response.ok) {
              return 'detail' in payload && payload.detail
                ? payload.detail
                : 'The matching API returned an error.'
            }

            const chatPayload = payload as ChatResponse
            sessionId.current = chatPayload.session_id
            pendingStructuredResponse.current = buildStructuredResponse(chatPayload)
            if (pendingStructuredResponse.current) {
              return
            }
            return chatPayload.response
          } catch {
            return 'I could not reach the matching API. Make sure FastAPI is running on port 8000.'
          }
        },
        component: () => {
          const structured = pendingStructuredResponse.current
          pendingStructuredResponse.current = null
          return structured ?? undefined
        },
        path: 'chat',
      },
    }),
    [],
  )

  const settings = useMemo<Settings>(
    () => ({
      general: {
        embedded: true,
        primaryColor: '#050505',
        secondaryColor: '#f5f5f5',
        fontFamily:
          '"Google Sans", "Product Sans", Arial, ui-sans-serif, system-ui, sans-serif',
        showFooter: false,
      },
      header: {
        title: (
          <span className="chat-brand">
            <span className="chat-title">
              <img
                className="panda-connect-mark"
                src={pandaConnectMark}
                alt="Panda Connect"
              />
            </span>
            <span className="chat-subtitle">Panda Matching Assistant</span>
          </span>
        ),
        showAvatar: false,
        buttons: [
          <button
            className="learn-more-button"
            type="button"
            onClick={() => {
              window.location.assign('/learn-more')
            }}
          >
            Panda Breeding Info
          </button>,
        ],
      },
      chatInput: {
        enabledPlaceholderText: 'Ask about panda matches...',
        botDelay: 250,
        blockSpam: true,
      },
      chatWindow: {
        defaultOpen: true,
        showTypingIndicator: true,
      },
      botBubble: {
        showAvatar: true,
        avatar: botAvatar,
      },
      userBubble: {
        showAvatar: true,
        avatar: userAvatar,
      },
      chatHistory: {
        storageKey: 'panda_matching_chat',
      },
      emoji: {
        disabled: true,
      },
      fileAttachment: {
        disabled: true,
      },
      notification: {
        disabled: true,
      },
      audio: {
        disabled: true,
      },
      voice: {
        disabled: true,
      },
    }),
    [],
  )

  const styles = useMemo<Styles>(
    () => ({
      chatWindowStyle: {
        width: 'min(730px, calc(100vw - 32px))',
        height: 'min(720px, calc(100svh - 48px))',
        borderRadius: 8,
        boxShadow: '0 24px 70px rgba(0, 0, 0, 0.12)',
        border: 'none',
      },
      headerStyle: {
        borderRadius: '8px 8px 0 0',
        background:
          'linear-gradient(90deg, rgba(255, 255, 255, 0.95) 0%, rgba(246, 239, 222, 0.94) 58%, rgba(202, 185, 154, 0.9) 100%)',
        borderBottom: '1px solid rgba(95, 78, 54, 0.28)',
        boxSizing: 'border-box',
        height: 132,
        minHeight: 132,
        alignItems: 'center',
      },
      bodyStyle: {
        background: 'rgba(255, 252, 245, 0.88)',
      },
      userBubbleStyle: {
        backgroundColor: 'rgba(255, 252, 245, 0.82)',
        backgroundImage: `linear-gradient(rgba(255, 252, 245, 0.18), rgba(255, 252, 245, 0.18)), url(${whiteFluff})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        color: '#241f19',
      },
      botBubbleStyle: {
        backgroundColor: 'rgba(35, 29, 22, 0.88)',
        backgroundImage: `linear-gradient(rgba(35, 29, 22, 0.15), rgba(35, 29, 22, 0.15)), url(${blackFluff})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        color: '#fff8ec',
        border: '1px solid rgba(73, 59, 43, 0.22)',
      },
      chatInputAreaStyle: {
        borderRadius: 8,
        background: 'rgba(255, 252, 245, 0.92)',
        border: '1px solid rgba(161, 137, 104, 0.38)',
      },
      sendButtonStyle: {
        borderRadius: 8,
        backgroundColor: 'rgba(128, 109, 73, 0.7)',
        backgroundImage: `linear-gradient(rgba(255, 252, 245, 0.28), rgba(75, 61, 37, 0.24)), url(${bamboo})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        border: '1px solid rgba(92, 75, 45, 0.28)',
      },
      rcbTypingIndicatorContainerStyle: {
        background: 'rgba(255, 252, 245, 0.92)',
        border: '1px solid rgba(161, 137, 104, 0.38)',
        borderRadius: 20,
        padding: '10px 14px',
        marginTop: 8,
        marginLeft: 16,
      },
    }),
    [],
  )

  if (isLearnMorePage) {
    return (
      <main className="app-shell">
        <section className="learn-more-page" aria-labelledby="learn-more-title">
          <div className="learn-more-header">
            <div>
              <h1 id="learn-more-title">How Pandas Breed</h1>
              <p>Panda matching assistant</p>
            </div>
            <img src={pandaConnectMark} alt="Two panda faces with a heart" />
          </div>

          <div className="learn-more-content">
            <h2>The 48-Hour Window</h2>
            <p>
              Female pandas are only fertile once a year, with a window of roughly
              24-72 hours. Miss it, and an entire year is lost. Keepers monitor
              females constantly using hormonal tests and behavioral cues to catch
              the exact moment.
            </p>
            <p>
              Attraction is everything and zoos make it hard. Pandas have genuine
              preferences. Scent is the most important factor: chemical signals
              communicate age, health, and identity. Personality matters too, a
              nervous panda paired with an aggressive one risks injury, not offspring.
            </p>
            <p>
              In the wild, a female would choose from multiple males. In a zoo, she
              typically has access to one. If the chemistry is not there, there is no
              fallback.
            </p>

            <h2>Very Few Cubs in a Lifetime</h2>
            <p>
              Females mature around age four and may breed for another ten to fifteen
              years, but most produce only two to five surviving cubs in their entire
              lifetime. Twins are common but mothers usually cannot raise both, so
              zoos sometimes rotate cubs between the mother and an incubator to save
              both.
            </p>
            <p>
              Cubs are born the size of a stick of butter and completely helpless.
              Pregnancy itself is unpredictable due to delayed implantation, making
              it hard to even confirm until late.
            </p>

            <h2>Why It Is So Hard to Conserve</h2>
            <p>
              With around 1,800 pandas left globally, every pairing decision matters.
              International transfers are expensive and stressful, most zoo pandas are
              on diplomatic loan, and genetic diversity must be carefully managed
              through a global studbook. All of this for a species that gets one shot
              per year.
            </p>

            <h2>The Panda Compatibility Chatbot</h2>
            <p>
              This chatbot aims to help identify which pandas might be good matches
              using all available data. Panda data is limited, personality and health
              records are inconsistently documented, but a solid dataset was assembled
              from Wikipedia, zoo websites, and conservation resources like the Black
              and White Bear website.
            </p>

            <h2>How the Score Works</h2>
            <p>
              Eligible pairs are first filtered for basics: breeding age, sex, cub
              count, and known incompatibilities. Then each pair gets a compatibility
              score from four components:
            </p>
            <ul>
              <li>Biology (35%): age gap and combined health risk</li>
              <li>Behavior (30%): personality compatibility, aggression risk, past breeding success</li>
              <li>Logistics (20%): same zoo scores best; international transfers and loan status are penalized</li>
              <li>Pair history (15%): currently held at a neutral baseline due to limited data</li>
            </ul>
            <p>
              These combine into a composite score, which adjusts a base
              recommendation up or down. The result is a 0-100 score; higher means
              biologically sound, behaviorally compatible, and practically feasible.
            </p>
            <p>
              The tool does not replace keeper and veterinarian judgment, but gives
              conservation programs a structured way to prioritize pairings in a
              situation where every breeding season counts.
            </p>
          </div>

          <button
            className="back-to-chat-button"
            type="button"
            onClick={() => {
              window.location.assign('/')
            }}
          >
            Back to Chat
          </button>
        </section>
      </main>
    )
  }

  return (
    <main className="app-shell">
      <div className="assistant-layout">
        <section className="chat-workspace" aria-label="Panda matching chatbot">
          <ChatBot id="panda-matching-chat" flow={flow} settings={settings} styles={styles} />
        </section>
        <PandaCatalog />
      </div>
      <a
        className="page-source-link"
        href="https://blackandwhitebear.com/"
        target="_blank"
        rel="noreferrer"
      >
        Pandas and images were extracted from: blackandwhitebear.com
      </a>
    </main>
  )
}

export default App
