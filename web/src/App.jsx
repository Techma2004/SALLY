import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Bot,
  Brain,
  Check,
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
  Square,
  Terminal,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import "./App.css";

const USER_KEY = "sally-web-user-id";
const CLEANUP_KEY = "sally-legacy-cache-cleaned-v1";
const REFRESH_MS = 30000;

const STARTERS = [
  "Calculate 25 × 40",
  "What time is it?",
  "Convert 5 miles to kilometers",
];

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

// "chat → conversation (0.70): reason" -> "chat · conversation"
function routeLabel(route) {
  if (!route) return "";
  return route.split(" (")[0].replace(" → ", " · ");
}

function formatElapsed(ms) {
  if (ms == null) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

async function api(path, options) {
  let response;

  try {
    response = await fetch(path, options);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error("Can't reach SALLY. Is the server running?");
  }

  if (!response.ok) {
    let detail = "";

    try {
      detail = (await response.json()).detail;
    } catch {
      // Non-JSON error body.
    }

    throw new Error(detail || `Request failed (${response.status}).`);
  }

  return response.json();
}

// Reads a text/event-stream response and calls onEvent for each data: line.
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const line = block.split("\n").find((entry) => entry.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false);

  const newline = code.indexOf("\n");
  const first = newline >= 0 ? code.slice(0, newline).trim() : "";
  const hasLanguage = newline >= 0 && /^[\w+#.-]{1,20}$/.test(first);
  const body = (hasLanguage ? code.slice(newline + 1) : code).replace(/\n$/, "");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(body);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard can be unavailable on insecure origins.
    }
  };

  return (
    <div className="code-block">
      <div className="code-block-bar">
        <span>{hasLanguage ? first : "code"}</span>
        <button type="button" onClick={copy} aria-label="Copy code">
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>{body}</code>
      </pre>
    </div>
  );
}

function InlineText({ text }) {
  return text
    .split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g)
    .map((piece, index) => {
      if (piece.length > 2 && piece.startsWith("`") && piece.endsWith("`")) {
        return (
          <code className="inline-code" key={index}>
            {piece.slice(1, -1)}
          </code>
        );
      }

      if (piece.length > 4 && piece.startsWith("**") && piece.endsWith("**")) {
        return <strong key={index}>{piece.slice(2, -2)}</strong>;
      }

      return piece;
    });
}

// Tiny, dependency-free renderer: fenced code, `inline code`, **bold**.
// Everything is rendered as React text nodes, never as raw HTML.
function MessageContent({ text, streaming }) {
  const parts = text.split("```");

  return (
    <>
      {parts.map((part, index) =>
        index % 2 === 1 ? (
          <CodeBlock code={part} key={index} />
        ) : (
          <InlineText text={part} key={index} />
        )
      )}
      {streaming && <span className="stream-cursor" aria-hidden="true" />}
    </>
  );
}

function ThinkingBubble() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="message-row assistant">
      <div className="message-avatar">
        <Bot size={18} strokeWidth={1.7} />
      </div>

      <div className="message-stack">
        <div className="message-bubble thinking-bubble">
          <div className="thinking-indicator" role="status">
            <span />
            <span />
            <span />
            <em>
              SALLY is thinking{seconds >= 3 ? ` · ${seconds}s` : ""}
            </em>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const userId = useMemo(getUserId, []);

  const scrollRef = useRef(null);
  const stickRef = useRef(true);
  const textareaRef = useRef(null);
  const abortRef = useRef(null);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState("chat");
  const [conversationId, setConversationId] = useState(null);
  const [conversationTitle, setConversationTitle] = useState("New Conversation");
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [online, setOnline] = useState(null);
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [tools, setTools] = useState([]);
  const [memories, setMemories] = useState([]);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [copiedId, setCopiedId] = useState(null);

  const loadHealth = async () => {
    setHealth(await api("/health"));
  };

  const loadSystem = async () => {
    setStats(await api("/system/stats"));
  };

  const loadTools = async () => {
    const data = await api("/tools");
    setTools(data.tools || []);
  };

  const loadConversations = async () => {
    const data = await api(
      `/conversations?user_id=${encodeURIComponent(userId)}`
    );
    const next = data.conversations || [];
    setConversations(next);
    return next;
  };

  const loadConversation = async (id) => {
    const data = await api(
      `/conversations/${encodeURIComponent(id)}?user_id=${encodeURIComponent(
        userId
      )}`
    );

    stickRef.current = true;
    setConversationId(data.conversation.id);
    setConversationTitle(data.conversation.title);
    setMessages(data.messages || []);
    setView("chat");
    setSidebarOpen(false);
    setError("");
  };

  const loadMemories = async () => {
    const query = memoryQuery.trim();
    const data = await api(
      query
        ? `/memory?q=${encodeURIComponent(query)}&limit=50`
        : "/memory/recent?limit=50"
    );
    setMemories(data.results || []);
  };

  const refreshStatus = () =>
    Promise.all([loadHealth(), loadSystem()])
      .then(() => setOnline(true))
      .catch((err) => {
        setOnline(false);
        throw err;
      });

  const loadEverything = async () => {
    try {
      await Promise.all([
        refreshStatus(),
        loadTools(),
        loadConversations(),
        view === "memory" ? loadMemories() : Promise.resolve(),
      ]);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  const newConversation = () => {
    abortRef.current?.abort();
    stickRef.current = true;
    setConversationId(null);
    setConversationTitle("New Conversation");
    setMessages([]);
    setMessage("");
    setError("");
    setView("chat");
    setSidebarOpen(false);
    textareaRef.current?.focus();
  };

  const openConversation = (id) => {
    if (busy) return;
    loadConversation(id).catch((err) => setError(err.message));
  };

  const deleteConversation = async (conversation) => {
    if (busy) return;

    const confirmed = window.confirm(
      `Delete "${conversation.title}"? This can't be undone.`
    );
    if (!confirmed) return;

    try {
      await api(
        `/conversations/${encodeURIComponent(
          conversation.id
        )}?user_id=${encodeURIComponent(userId)}`,
        { method: "DELETE" }
      );

      if (conversation.id === conversationId) newConversation();
      await loadConversations();
    } catch (err) {
      setError(err.message);
    }
  };

  const forgetMemory = async (memory) => {
    const confirmed = window.confirm(
      "Forget this memory? SALLY will no longer use it."
    );
    if (!confirmed) return;

    try {
      await api(`/memory/${encodeURIComponent(memory.id)}`, {
        method: "DELETE",
      });
      setMemories((current) => current.filter((item) => item.id !== memory.id));
    } catch (err) {
      setError(err.message);
    }
  };

  // One-time cleanup of the old service worker / caches (no forced reload).
  useEffect(() => {
    if (localStorage.getItem(CLEANUP_KEY) === "1") return;

    (async () => {
      try {
        if ("serviceWorker" in navigator) {
          const registrations = await navigator.serviceWorker.getRegistrations();
          await Promise.all(registrations.map((entry) => entry.unregister()));
        }

        if ("caches" in window) {
          const names = await window.caches.keys();
          await Promise.all(names.map((name) => window.caches.delete(name)));
        }
      } catch {
        // Best effort only.
      }

      localStorage.setItem(CLEANUP_KEY, "1");
    })();
  }, []);

  // Initial load, then poll status only while the tab is visible.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [nextConversations] = await Promise.all([
          loadConversations(),
          refreshStatus(),
          loadTools(),
        ]);

        if (!cancelled && nextConversations[0]) {
          await loadConversation(nextConversations[0].id);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();

    const poll = () => {
      if (document.hidden) return;
      refreshStatus().catch(() => {});
    };

    const timer = window.setInterval(poll, REFRESH_MS);
    document.addEventListener("visibilitychange", poll);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", poll);
    };
  }, []);

  // Stay pinned to the newest message unless the reader scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages, busy, view]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  // Auto-grow the composer.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 155)}px`;
  }, [message, view]);

  // Escape closes the mobile sidebar.
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (view === "tools") loadTools().catch((err) => setError(err.message));
    if (view === "system") refreshStatus().catch((err) => setError(err.message));
  }, [view]);

  // Memory view: load on open and (debounced) while typing a search.
  useEffect(() => {
    if (view !== "memory") return undefined;

    const timer = window.setTimeout(() => {
      loadMemories().catch((err) => setError(err.message));
    }, 250);

    return () => window.clearTimeout(timer);
  }, [view, memoryQuery]);

  const stopGenerating = () => abortRef.current?.abort();

  const sendMessage = async (prefilled) => {
    const text = (prefilled ?? message).trim();

    if (!text || busy) return;

    const controller = new AbortController();
    abortRef.current = controller;

    const stamp = Date.now();
    const userItem = {
      id: `local-${stamp}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    const replyId = `reply-${stamp}`;
    let accepted = false;
    let activeId = conversationId;

    const patchReply = (patch) =>
      setMessages((current) =>
        current.map((item) =>
          item.id === replyId ? { ...item, ...patch } : item
        )
      );

    stickRef.current = true;
    setMessages((current) => [...current, userItem]);
    setMessage("");
    setBusy(true);
    setError("");

    try {
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          user_id: userId,
          source: "web",
          conversation_id: conversationId,
        }),
        signal: controller.signal,
      }).catch((err) => {
        if (err.name === "AbortError") throw err;
        throw new Error("Can't reach SALLY. Is the server running?");
      });

      if (!response.ok) {
        let detail = "";
        try {
          detail = (await response.json()).detail;
        } catch {
          // Non-JSON error body.
        }
        throw new Error(detail || "SALLY could not process the request.");
      }

      await readEvents(response, (event) => {
        if (event.type === "start") {
          accepted = true;
          activeId = event.conversation_id;
          setConversationId(event.conversation_id);
          setMessages((current) => [
            ...current,
            {
              id: replyId,
              role: "assistant",
              content: "",
              created_at: new Date().toISOString(),
              streaming: true,
              route: event.route,
            },
          ]);
        } else if (event.type === "token") {
          setMessages((current) =>
            current.map((item) =>
              item.id === replyId
                ? { ...item, content: item.content + event.text }
                : item
            )
          );
        } else if (event.type === "done") {
          if (!event.answer) {
            setMessages((current) =>
              current.filter((item) => item.id !== replyId)
            );
            setError(event.error || "SALLY returned an empty response.");
          } else {
            patchReply({
              id: `assistant-${event.task_id}`,
              content: event.answer,
              streaming: false,
              agent_name: event.agent_name,
              route: event.route,
              elapsed_ms: event.elapsed_ms,
            });
          }
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      });
    } catch (err) {
      if (err.name === "AbortError") {
        patchReply({ streaming: false, stopped: true });
      } else {
        setError(err.message);

        setMessages((current) =>
          current.filter(
            (item) =>
              !(item.id === replyId && !item.content) &&
              !(item.id === userItem.id && !accepted)
          )
        );

        patchReply({ streaming: false });

        if (!accepted) setMessage(text);
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
      textareaRef.current?.focus();

      loadConversations()
        .then((list) => {
          const match = list.find((item) => item.id === activeId);
          if (match) setConversationTitle(match.title);
        })
        .catch(() => {});
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
    ["chat", MessageSquare, "Chat"],
    ["memory", Brain, "Memory"],
    ["tools", Wrench, "Tools"],
    ["system", Activity, "System"],
  ];

  const connectionLabel =
    online === false
      ? "Offline"
      : health?.status === "ok"
        ? "Local gateway"
        : "Connecting…";

  const dotClass = `status-dot ${online === false ? "offline" : ""}`;
  const awaitingFirstToken =
    busy && !messages.some((item) => item.streaming);

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

        <nav className="navigation" aria-label="Workspace">
          <div className="navigation-label">WORKSPACE</div>

          {navItems.map(([key, Icon, label]) => (
            <button
              key={key}
              className={`nav-item ${view === key ? "active" : ""}`}
              aria-current={view === key ? "page" : undefined}
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
            <span className="section-count">{conversations.length}</span>
          </div>

          <div className="recent-list">
            {conversations.length === 0 ? (
              <div className="empty-sidebar">
                Your conversations will appear here.
              </div>
            ) : (
              conversations.map((conversation) => (
                <div className="conversation-row" key={conversation.id}>
                  <button
                    className={`conversation-item ${
                      conversation.id === conversationId ? "selected" : ""
                    }`}
                    onClick={() => openConversation(conversation.id)}
                  >
                    <span>{conversation.title}</span>
                    <small>{formatDate(conversation.updated_at)}</small>
                  </button>

                  <button
                    className="conversation-delete"
                    onClick={() => deleteConversation(conversation)}
                    aria-label={`Delete conversation ${conversation.title}`}
                    title="Delete conversation"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
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
            <h1 className="conversation-heading">{conversationTitle}</h1>

            <div className="status-pill">
              <span className={dotClass} />
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
            <div className="chat-scroll" ref={scrollRef} onScroll={handleScroll}>
              <div
                className="message-column"
                role="log"
                aria-live="polite"
                aria-label="Conversation"
              >
                {messages.length === 0 && !busy ? (
                  <div className="welcome-state">
                    <div className="welcome-mark">
                      <div className="welcome-orbit" />
                      <Bot size={38} strokeWidth={1.6} />
                    </div>

                    <div className="welcome-kicker">
                      Science · Artificial · Learning · Logic · You
                    </div>

                    <h2 className="welcome-title">How can SALLY help?</h2>
                    <p>
                      A local-first AI workspace for conversations,
                      reasoning, tools, and persistent memory.
                    </p>

                    <div className="starter-grid">
                      {STARTERS.map((prompt) => (
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
                            <MessageContent
                              text={item.content}
                              streaming={item.streaming}
                            />
                          </div>

                          <div className="message-meta">
                            <span>{formatTime(item.created_at)}</span>
                            {item.role === "assistant" && item.route && (
                              <>
                                <span className="meta-separator">·</span>
                                <span>{routeLabel(item.route)}</span>
                              </>
                            )}
                            {item.elapsed_ms != null && (
                              <>
                                <span className="meta-separator">·</span>
                                <span>{formatElapsed(item.elapsed_ms)}</span>
                              </>
                            )}
                            {item.stopped && (
                              <>
                                <span className="meta-separator">·</span>
                                <span>Stopped</span>
                              </>
                            )}
                          </div>
                        </div>

                        {item.role === "assistant" && !item.streaming && (
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

                {awaitingFirstToken && <ThinkingBubble />}
              </div>
            </div>

            <div className="composer-zone">
              <div className="composer-shell">
                <textarea
                  ref={textareaRef}
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder={
                    busy ? "SALLY is replying…" : "Message SALLY…"
                  }
                  rows={1}
                  aria-label="Message SALLY"
                />

                <div className="composer-footer">
                  <div className="composer-note">
                    <span className="composer-live-dot" />
                    <span>{connectionLabel}</span>
                    <span className="composer-divider">·</span>
                    <span>Enter to send</span>
                    <span className="composer-divider composer-hint">·</span>
                    <span className="composer-hint">
                      Shift + Enter for a new line
                    </span>
                  </div>

                  {busy ? (
                    <button
                      className="send-button stop-button"
                      onClick={stopGenerating}
                      aria-label="Stop generating"
                      title="Stop generating"
                    >
                      <Square size={15} fill="currentColor" />
                    </button>
                  ) : (
                    <button
                      className="send-button"
                      onClick={() => sendMessage()}
                      disabled={!message.trim()}
                      aria-label="Send message"
                    >
                      <Send size={17} />
                    </button>
                  )}
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
                aria-label="Search memory"
              />
              {memoryQuery && (
                <button
                  type="button"
                  className="search-clear"
                  onClick={() => setMemoryQuery("")}
                  aria-label="Clear search"
                >
                  <X size={14} />
                </button>
              )}
              <button type="submit">Search</button>
            </form>

            <div className="data-list">
              {memories.length === 0 ? (
                <div className="empty-state">
                  <Brain size={22} />
                  <strong>
                    {memoryQuery.trim() ? "No matching memories" : "No memories yet"}
                  </strong>
                  <span>
                    {memoryQuery.trim()
                      ? "Try a different search term."
                      : "Saved facts and preferences will appear here."}
                  </span>
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
                      <button
                        type="button"
                        className="forget-button"
                        onClick={() => forgetMemory(memory)}
                        aria-label="Forget this memory"
                      >
                        <Trash2 size={13} />
                        Forget
                      </button>
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
                  SALLY v{health?.version ?? "—"} · Local gateway ·{" "}
                  {stats?.context_tokens ?? "—"} context tokens
                </p>
              </div>

              <div className="runtime-health">
                <span className={dotClass} />
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
                <span className={dotClass} />
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
