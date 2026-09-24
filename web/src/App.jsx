import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Bot,
  Brain,
  Check,
  ChevronDown,
  Code2,
  Copy,
  Database,
  Menu,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Terminal,
  Wrench,
  X,
} from "lucide-react";
import "./App.css";

const USER_KEY = "sally-web-user-id";
const REFRESH_MS = 30000;

function getUserId() {
  let id = localStorage.getItem(USER_KEY);

  if (!id) {
    id =
      globalThis.crypto?.randomUUID?.() ||
      `web-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem(USER_KEY, id);
  }

  return id;
}

function formatTime(value) {
  if (!value) return "";

  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDate(value) {
  if (!value) return "";

  return new Date(value).toLocaleString([], {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function shortModel(path) {
  if (!path) return "Unknown model";
  return path.split("/").pop();
}

function formatRuntime(value) {
  if (value == null) return "—";
  return String(value);
}

function App() {
  const userId = useMemo(getUserId, []);

  useEffect(() => {
    const cleanupKey = "sally-legacy-cache-cleaned-v1";

    if (sessionStorage.getItem(cleanupKey) === "1") {
      return;
    }

    let active = true;

    (async () => {
      try {
        if ("serviceWorker" in navigator) {
          const registrations = await navigator.serviceWorker.getRegistrations();
          await Promise.all(
            registrations.map((registration) => registration.unregister())
          );
        }

        if ("caches" in window) {
          const cacheNames = await window.caches.keys();
          await Promise.all(cacheNames.map((name) => window.caches.delete(name)));
        }
      } catch {
        // Legacy local service-worker cleanup is best effort.
      }

      sessionStorage.setItem(cleanupKey, "1");

      if (active) {
        window.setTimeout(() => window.location.reload(), 0);
      }
    })();

    return () => {
      active = false;
    };
  }, []);
  const chatEndRef = useRef(null);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState("chat");
  const [conversationId, setConversationId] = useState(null);
  const [conversationTitle, setConversationTitle] = useState("New Conversation");
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [tools, setTools] = useState([]);
  const [memories, setMemories] = useState([]);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [copiedId, setCopiedId] = useState(null);

  const loadHealth = async () => {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("SALLY gateway health check failed.");
    setHealth(await response.json());
  };

  const loadSystem = async () => {
    const response = await fetch("/system/stats");
    if (!response.ok) throw new Error("Could not load system status.");
    setStats(await response.json());
  };

  const loadTools = async () => {
    const response = await fetch("/tools");
    if (!response.ok) throw new Error("Could not load tools.");
    const data = await response.json();
    setTools(data.tools || []);
  };

  const loadConversations = async () => {
    const response = await fetch(
      `/conversations?user_id=${encodeURIComponent(userId)}`
    );

    if (!response.ok) throw new Error("Could not load conversations.");

    const data = await response.json();
    const next = data.conversations || [];
    setConversations(next);
    return next;
  };

  const loadConversation = async (id) => {
    const response = await fetch(
      `/conversations/${encodeURIComponent(id)}?user_id=${encodeURIComponent(
        userId
      )}`
    );

    if (!response.ok) throw new Error("Could not load conversation.");

    const data = await response.json();

    setConversationId(data.conversation.id);
    setConversationTitle(data.conversation.title);
    setMessages(data.messages || []);
    setView("chat");
    setSidebarOpen(false);
    setError("");
  };

  const loadMemories = async () => {
    const endpoint = memoryQuery.trim()
      ? `/memory?q=${encodeURIComponent(
          memoryQuery.trim()
        )}&limit=50`
      : "/memory/recent?limit=50";

    const response = await fetch(endpoint);

    if (!response.ok) throw new Error("Could not load memory.");

    const data = await response.json();
    setMemories(data.results || []);
  };

  const loadEverything = async () => {
    try {
      await Promise.all([
        loadHealth(),
        loadSystem(),
        loadTools(),
        loadConversations(),
        loadMemories(),
      ]);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  const newConversation = () => {
    setConversationId(null);
    setConversationTitle("New Conversation");
    setMessages([]);
    setMessage("");
    setError("");
    setView("chat");
    setSidebarOpen(false);
  };

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [nextConversations] = await Promise.all([
          loadConversations(),
          loadHealth(),
          loadSystem(),
          loadTools(),
          loadMemories(),
        ]);

        if (!cancelled && nextConversations[0]) {
          await loadConversation(nextConversations[0].id);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();

    const timer = window.setInterval(() => {
      Promise.all([loadHealth(), loadSystem()])
        .catch((err) => setError(err.message));
    }, REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, busy]);

  useEffect(() => {
    if (view === "memory") {
      loadMemories().catch((err) => setError(err.message));
    }

    if (view === "tools") {
      loadTools().catch((err) => setError(err.message));
    }

    if (view === "system") {
      Promise.all([loadHealth(), loadSystem()])
        .catch((err) => setError(err.message));
    }
  }, [view]);

  const sendMessage = async (prefilled) => {
    const text = (prefilled ?? message).trim();

    if (!text || busy) return;

    const optimistic = {
      id: `local-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };

    setMessages((current) => [...current, optimistic]);
    setMessage("");
    setBusy(true);
    setError("");

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: text,
          user_id: userId,
          source: "web",
          conversation_id: conversationId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "SALLY could not process the request."
        );
      }

      setConversationId(data.conversation_id);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${data.task_id}`,
          role: "assistant",
          content: data.answer,
          created_at: new Date().toISOString(),
          agent_name: data.agent_name,
          route: data.route,
          elapsed_ms: data.elapsed_ms,
        },
      ]);

      await loadConversations();

      if (!conversationId) {
        await loadConversation(data.conversation_id);
      }
    } catch (err) {
      setError(err.message);
      setMessages((current) =>
        current.filter((item) => item.id !== optimistic.id)
      );
    } finally {
      setBusy(false);
    }
  };

  const copyMessage = async (item) => {
    try {
      await navigator.clipboard.writeText(item.content);
      setCopiedId(item.id);

      window.setTimeout(() => {
        setCopiedId((current) => (current === item.id ? null : current));
      }, 1400);
    } catch {
      setError("Clipboard access is unavailable.");
    }
  };

  const navItems = [
    ["chat", MessageSquare, "Conversations"],
    ["memory", Brain, "Memory"],
    ["tools", Wrench, "Tools"],
    ["system", Activity, "System"],
  ];

  const connectionLabel =
    health?.status === "ok" ? "Local gateway" : "Connecting…";

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      {sidebarOpen && (
        <div
          className="mobile-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="sidebar-top">
          <div className="brand">
            <div className="sally-mark">
              <Bot size={25} strokeWidth={1.8} />
            </div>

            <div>
              <div className="brand-name">SALLY</div>
              <div className="brand-subtitle">Personal AI</div>
            </div>
          </div>

          <button
            className="icon-button mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close navigation"
          >
            <X size={19} />
          </button>
        </div>

        <button className="new-chat-button" onClick={newConversation}>
          <span className="new-chat-icon">
            <Plus size={17} />
          </span>
          <span>New conversation</span>
        </button>

        <nav className="navigation">
          <div className="navigation-label">WORKSPACE</div>

          {navItems.map(([key, Icon, label]) => (
            <button
              key={key}
              className={`nav-item ${view === key ? "active" : ""}`}
              onClick={() => {
                setView(key);
                setSidebarOpen(false);
              }}
            >
              <Icon size={18} strokeWidth={1.8} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="recent-section">
          <div className="section-heading">
            <span>RECENT</span>
            <span className="section-count">
              {conversations.length}
            </span>
          </div>

          <div className="recent-list">
            {conversations.length === 0 ? (
              <div className="empty-sidebar">
                Your conversations will appear here.
              </div>
            ) : (
              conversations.map((conversation) => (
                <button
                  key={conversation.id}
                  className={`conversation-item ${
                    conversation.id === conversationId ? "selected" : ""
                  }`}
                  onClick={() =>
                    loadConversation(conversation.id).catch((err) =>
                      setError(err.message)
                    )
                  }
                >
                  <span>{conversation.title}</span>
                  <small>{formatDate(conversation.updated_at)}</small>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="privacy-card">
            <div className="privacy-icon">
              <ShieldCheck size={17} />
            </div>

            <div>
              <strong>Local & private</strong>
              <span>Your browser connects to SALLY on this machine.</span>
            </div>
          </div>

          <div className="session-line">
            <div className="session-avatar">
              <Terminal size={15} />
            </div>

            <div className="session-copy">
              <strong>Browser session</strong>
              <span>{userId.slice(0, 12)}…</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={21} />
          </button>

          <div className="topbar-center">
            <button
              className="conversation-heading"
              onClick={() => setView("chat")}
              title="Return to conversation"
            >
              <span>{conversationTitle}</span>
              <ChevronDown size={15} />
            </button>

            <div className="status-pill">
              <span className="status-dot" />
              <span>{connectionLabel}</span>
            </div>
          </div>

          <div className="topbar-actions">
            <button
              className="icon-button"
              onClick={loadEverything}
              title="Refresh SALLY"
              aria-label="Refresh SALLY"
            >
              <RefreshCw size={17} />
            </button>

            <button
              className="icon-button accent-icon-button"
              onClick={newConversation}
              title="New conversation"
              aria-label="New conversation"
            >
              <Plus size={18} />
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button
              className="error-close"
              onClick={() => setError("")}
              aria-label="Dismiss error"
            >
              <X size={15} />
            </button>
          </div>
        )}

        {view === "chat" && (
          <section className="chat-view">
            <div className="chat-scroll">
              <div className="message-column">
                {messages.length === 0 ? (
                  <div className="welcome-state">
                    <div className="welcome-mark">
                      <div className="welcome-orbit" />
                      <Bot size={38} strokeWidth={1.6} />
                    </div>

                    <div className="welcome-kicker">
                      Science · Artificial · Learning · Logic · You
                    </div>

                    <h1>How can SALLY help?</h1>
                    <p>
                      A local-first AI workspace for conversations,
                      reasoning, tools, and persistent memory.
                    </p>

                    <div className="starter-grid">
                      {[
                        "Explain how your memory works",
                        "Calculate 25 × 40",
                        "Show me the tools you currently have",
                      ].map((prompt) => (
                        <button
                          key={prompt}
                          className="starter-card"
                          onClick={() => sendMessage(prompt)}
                        >
                          <Sparkles size={16} />
                          <span>{prompt}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  messages.map((item) => (
                    <div
                      key={item.id}
                      className={`message-row ${
                        item.role === "user" ? "user" : "assistant"
                      }`}
                    >
                      {item.role !== "user" && (
                        <div className="message-avatar">
                          <Bot size={18} strokeWidth={1.7} />
                        </div>
                      )}

                      <div className="message-stack">
                        <div
                          className={`message-bubble ${
                            item.role === "user" ? "user-bubble" : ""
                          }`}
                        >
                          <div className="message-text">
                            {item.content}
                          </div>

                          <div className="message-meta">
                            <span>{formatTime(item.created_at)}</span>
                            {item.route && (
                              <>
                                <span className="meta-separator">·</span>
                                <span>{item.route}</span>
                              </>
                            )}
                            {item.elapsed_ms != null && (
                              <>
                                <span className="meta-separator">·</span>
                                <span>{item.elapsed_ms} ms</span>
                              </>
                            )}
                          </div>
                        </div>

                        {item.role === "assistant" && (
                          <div className="message-tools">
                            <button
                              onClick={() => copyMessage(item)}
                              className="message-tool"
                              title="Copy response"
                            >
                              {copiedId === item.id ? (
                                <>
                                  <Check size={14} />
                                  <span>Copied</span>
                                </>
                              ) : (
                                <>
                                  <Copy size={14} />
                                  <span>Copy</span>
                                </>
                              )}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}

                {busy && (
                  <div className="message-row assistant">
                    <div className="message-avatar">
                      <Bot size={18} strokeWidth={1.7} />
                    </div>

                    <div className="message-stack">
                      <div className="message-bubble thinking-bubble">
                        <div className="thinking-indicator">
                          <span />
                          <span />
                          <span />
                          <em>SALLY is thinking</em>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>
            </div>

            <div className="composer-zone">
              <div className="composer-shell">
                <textarea
                  value={message}
                  disabled={busy}
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder="Message SALLY…"
                  rows={1}
                  aria-label="Message SALLY"
                />

                <div className="composer-footer">
                  <div className="composer-note">
                    <span className="composer-live-dot" />
                    <span>{connectionLabel}</span>
                    <span className="composer-divider">·</span>
                    <span>Enter to send</span>
                    <span className="composer-divider">·</span>
                    <span>Shift + Enter for a new line</span>
                  </div>

                  <button
                    className="send-button"
                    onClick={() => sendMessage()}
                    disabled={!message.trim() || busy}
                    aria-label="Send message"
                  >
                    <Send size={17} />
                  </button>
                </div>
              </div>

              <div className="composer-disclaimer">
                SALLY can make mistakes. Verify important information.
              </div>
            </div>
          </section>
        )}

        {view === "memory" && (
          <section className="dashboard-view">
            <div className="view-header">
              <div>
                <span className="eyebrow">PERSISTENT MEMORY</span>
                <h2>Memory</h2>
                <p>
                  Direct access to the SQLite memory store used by SALLY.
                </p>
              </div>

              <button
                className="secondary-button"
                onClick={() =>
                  loadMemories().catch((err) => setError(err.message))
                }
              >
                <RefreshCw size={15} />
                Refresh
              </button>
            </div>

            <form
              className="search-box"
              onSubmit={(event) => {
                event.preventDefault();
                loadMemories().catch((err) => setError(err.message));
              }}
            >
              <Search size={17} />
              <input
                value={memoryQuery}
                onChange={(event) => setMemoryQuery(event.target.value)}
                placeholder="Search memory…"
              />
              <button type="submit">Search</button>
            </form>

            <div className="data-list">
              {memories.length === 0 ? (
                <div className="empty-state">
                  <Brain size={22} />
                  <strong>No matching memories</strong>
                  <span>SALLY has nothing to show for this query yet.</span>
                </div>
              ) : (
                memories.map((memory) => (
                  <article className="data-card" key={memory.id}>
                    <div className="data-card-top">
                      <span className="badge">{memory.type}</span>
                      <span>{formatDate(memory.created_at)}</span>
                    </div>

                    <p>{memory.content}</p>

                    <div className="data-card-bottom">
                      <span>Importance</span>
                      <strong>{Math.round(memory.importance * 100)}%</strong>
                    </div>
                  </article>
                ))
              )}
            </div>
          </section>
        )}

        {view === "tools" && (
          <section className="dashboard-view">
            <div className="view-header">
              <div>
                <span className="eyebrow">RUNTIME CAPABILITIES</span>
                <h2>Tools</h2>
                <p>
                  Tools registered by the current SALLY runtime.
                </p>
              </div>
            </div>

            <div className="tool-grid">
              {tools.map((tool) => (
                <article className="tool-card" key={tool.name}>
                  <div className="tool-icon">
                    <Wrench size={17} />
                  </div>

                  <div>
                    <div className="tool-title-row">
                      <h3>{tool.name}</h3>
                      <span>{tool.safety}</span>
                    </div>

                    <p>{tool.description}</p>

                    <small>
                      {tool.requires_inference
                        ? "Verified result + model explanation"
                        : "Deterministic execution"}
                    </small>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {view === "system" && (
          <section className="dashboard-view">
            <div className="view-header">
              <div>
                <span className="eyebrow">LIVE RUNTIME</span>
                <h2>System</h2>
                <p>
                  Live information from the machine running SALLY.
                </p>
              </div>

              <button
                className="secondary-button"
                onClick={() =>
                  Promise.all([loadHealth(), loadSystem()])
                    .catch((err) => setError(err.message))
                }
              >
                <RefreshCw size={15} />
                Refresh
              </button>
            </div>

            <div className="stats-grid">
              {[
                ["CPU", stats ? `${stats.cpu_percent}%` : "—"],
                [
                  "RAM",
                  stats
                    ? `${stats.ram_used_gb} / ${stats.ram_total_gb} GB`
                    : "—",
                ],
                [
                  "Disk",
                  stats ? `${stats.disk_percent}% used` : "—",
                ],
                [
                  "Memory DB",
                  stats ? `${stats.memory_db_mb} MB` : "—",
                ],
                [
                  "Context",
                  stats ? `${stats.context_tokens} tokens` : "—",
                ],
                [
                  "Web port",
                  stats?.web_port ?? 5678,
                ],
              ].map(([label, value]) => (
                <article className="stat-card" key={label}>
                  <span>{label}</span>
                  <strong>{formatRuntime(value)}</strong>
                </article>
              ))}
            </div>

            <div className="runtime-card">
              <div className="runtime-card-icon">
                <Database size={19} />
              </div>

              <div>
                <span className="eyebrow">MODEL</span>
                <h3>{shortModel(stats?.model)}</h3>
                <p>
                  SALLY v{health?.version ?? "—"} · Local gateway ·
                  {stats?.context_tokens ?? "—"} context tokens
                </p>
              </div>

              <div className="runtime-health">
                <span className="status-dot" />
                <span>{health?.status === "ok" ? "Healthy" : "Checking"}</span>
              </div>
            </div>
          </section>
        )}
      </main>

      <aside className="right-panel">
        <div className="right-panel-inner">
          <div className="identity-block">
            <div className="identity-mark">
              <Bot size={29} strokeWidth={1.6} />
            </div>

            <div>
              <div className="identity-name">SALLY</div>
              <div className="identity-description">
                Science Artificial Learning Logic And You
              </div>
            </div>
          </div>

          <div className="panel-section-block">
            <div className="panel-label">
              <span>LIVE RUNTIME</span>
              <span className="panel-live">
                <span className="status-dot" />
                {connectionLabel}
              </span>
            </div>

            <div className="metric-list">
              <div className="metric-row">
                <div className="metric-icon"><Activity size={15} /></div>
                <div>
                  <span>Gateway</span>
                  <strong>{health?.status === "ok" ? "Ready" : "Checking"}</strong>
                </div>
              </div>

              <div className="metric-row">
                <div className="metric-icon"><Database size={15} /></div>
                <div>
                  <span>Memory</span>
                  <strong>{stats ? `${stats.memory_db_mb} MB` : "—"}</strong>
                </div>
              </div>

              <div className="metric-row">
                <div className="metric-icon"><Terminal size={15} /></div>
                <div>
                  <span>Model</span>
                  <strong>{shortModel(stats?.model)}</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="panel-section-block">
            <div className="panel-label">
              <span>ACTIVE TOOLS</span>
              <span className="panel-count">{tools.length}</span>
            </div>

            <div className="tool-list">
              {tools.slice(0, 6).map((tool) => (
                <div className="mini-tool" key={tool.name}>
                  <div className="mini-tool-icon">
                    <Code2 size={15} />
                  </div>

                  <div>
                    <strong>{tool.name}</strong>
                    <span>{tool.requires_inference ? "Inference assisted" : "Deterministic"}</span>
                  </div>
                </div>
              ))}

              {tools.length === 0 && (
                <div className="panel-empty">No tools reported.</div>
              )}
            </div>
          </div>

          <div className="panel-section-block panel-session">
            <div className="panel-label">
              <span>SESSION</span>
            </div>

            <div className="session-stat">
              <strong>{conversations.length}</strong>
              <span>saved conversation{conversations.length === 1 ? "" : "s"}</span>
            </div>

            <div className="session-stat secondary">
              <strong>{userId.slice(0, 12)}…</strong>
              <span>browser identity</span>
            </div>
          </div>

          <div className="right-panel-footer">
            <Sparkles size={15} />
            <span>Local-first · no demo data</span>
          </div>
        </div>
      </aside>
    </div>
  );
}

export default App;
