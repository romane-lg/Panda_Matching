from __future__ import annotations


def render_chat_ui() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Panda Matching Chat</title>
    <style>
      :root {
        --bg: #f7f4ed;
        --card: #fffdf8;
        --ink: #1f2937;
        --muted: #6b7280;
        --accent: #1f7a5c;
        --border: #e5dccd;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(1200px 500px at 10% -10%, #d7efe6 0%, transparent 60%),
          radial-gradient(800px 400px at 90% -5%, #f4e7c6 0%, transparent 60%),
          var(--bg);
      }
      .wrap {
        max-width: 900px;
        margin: 32px auto;
        padding: 0 16px;
      }
      .panel {
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,.06);
        overflow: hidden;
      }
      .head {
        padding: 16px 20px;
        border-bottom: 1px solid var(--border);
      }
      .head h1 {
        margin: 0;
        font-size: 20px;
      }
      .head p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 14px;
      }
      .log {
        height: 58vh;
        overflow: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .msg {
        max-width: 85%;
        padding: 12px 14px;
        border-radius: 12px;
        line-height: 1.35;
        white-space: pre-wrap;
      }
      .user {
        align-self: flex-end;
        background: #ddf2eb;
        border: 1px solid #bce5d8;
      }
      .bot {
        align-self: flex-start;
        background: #fff;
        border: 1px solid var(--border);
      }
      .composer {
        border-top: 1px solid var(--border);
        padding: 12px;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 10px;
      }
      input[type="text"] {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        font-size: 14px;
      }
      button {
        border: 0;
        border-radius: 10px;
        padding: 0 16px;
        background: var(--accent);
        color: white;
        font-weight: 600;
        cursor: pointer;
      }
      button:disabled { opacity: .6; cursor: not-allowed; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="panel">
        <div class="head">
          <h1>Panda Matching Chat</h1>
          <p>Try: "top 5 matches for Ai Bao", "explain 16 81", "blockers for 16".</p>
        </div>
        <div id="log" class="log"></div>
        <form id="form" class="composer">
          <input id="input" type="text" placeholder="Ask about panda matches..." />
          <button id="send" type="submit">Send</button>
        </form>
      </div>
    </div>
    <script>
      const log = document.getElementById("log");
      const form = document.getElementById("form");
      const input = document.getElementById("input");
      const send = document.getElementById("send");
      let sessionId = null;

      function addMessage(kind, text) {
        const div = document.createElement("div");
        div.className = `msg ${kind}`;
        div.textContent = text;
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
      }

      async function sendMessage(message) {
        send.disabled = true;
        try {
          const res = await fetch("/agent/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId })
          });
          const contentType = res.headers.get("content-type") || "";
          const payload = contentType.includes("application/json")
            ? await res.json()
            : { detail: await res.text() };
          if (!res.ok) {
            addMessage("bot", payload.detail || "Request failed");
            return;
          }
          sessionId = payload.session_id;
          addMessage("bot", payload.response);
        } catch (err) {
          addMessage("bot", "Network error while calling /agent/chat");
        } finally {
          send.disabled = false;
        }
      }

      addMessage("bot", "Chat ready. Type 'help' for supported commands.");
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message) return;
        addMessage("user", message);
        input.value = "";
        await sendMessage(message);
      });
    </script>
  </body>
</html>
"""
