import { useMemo, useRef } from 'react'
import ChatBot, { type Flow, type Settings, type Styles } from 'react-chatbotify'
import bamboo from './assets/bamboo.jpg'
import blackFluff from './assets/black-fluff.png'
import botAvatar from './assets/panda-bot-avatar.png'
import pandaConnectMark from './assets/panda-connect.png'
import userAvatar from './assets/panda-user-avatar.png'
import whiteFluff from './assets/white-fluff.png'
import './App.css'

type ChatResponse = {
  session_id: string
  intent: string
  response: string
  data: Record<string, unknown> | null
}

function App() {
  const sessionId = useRef<string | null>(null)
  const isLearnMorePage = window.location.pathname === '/learn-more'

  const flow = useMemo<Flow>(
    () => ({
      start: {
        message:
          'Chat ready. Try "best overall panda match", "top 5 matches for Ai Bao", or "tell me about a panda match".',
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
            return chatPayload.response
          } catch {
            return 'I could not reach the matching API. Make sure FastAPI is running on port 8000.'
          }
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
              <span>PandaConnect</span>
              <img
                className="panda-connect-mark"
                src={pandaConnectMark}
                alt="Two panda faces with a heart"
              />
            </span>
            <span className="chat-subtitle">Panda matching assistant</span>
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
            Learn More
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
        width: 'min(920px, calc(100vw - 32px))',
        height: 'min(720px, calc(100svh - 48px))',
        borderRadius: 8,
        boxShadow: '0 24px 70px rgba(0, 0, 0, 0.12)',
        border: 'none',
      },
      headerStyle: {
        borderRadius: '8px 8px 0 0',
        background:
          'linear-gradient(90deg, rgba(255, 255, 255, 0.96) 0%, rgba(246, 239, 222, 0.94) 62%, rgba(126, 107, 78, 0.78) 78%, rgba(48, 39, 29, 0.82) 100%)',
        borderBottom: '1px solid rgba(95, 78, 54, 0.28)',
        minHeight: 76,
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
              <h1 id="learn-more-title">PandaConnect</h1>
              <p>Panda matching assistant</p>
            </div>
            <img src={pandaConnectMark} alt="Two panda faces with a heart" />
          </div>

          <div className="learn-more-content">
            <p>
              PandaConnect helps explore panda breeding compatibility using profile data,
              eligibility rules, ranked pair scoring, and explainable match factors.
            </p>
            <p>
              Ask it for top matches, best overall pairings, health or personality notes,
              blocker reasons, and ranking explanations grounded in the current database.
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
      <section className="chat-workspace" aria-label="Panda matching chatbot">
        <ChatBot id="panda-matching-chat" flow={flow} settings={settings} styles={styles} />
      </section>
    </main>
  )
}

export default App
