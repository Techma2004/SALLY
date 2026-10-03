import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  Check,
  Copy,
  History,
  Moon,
  Plus,
  Search,
  Square,
  Sun,
  Trash2,
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

function Thinking() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setSeconds((v) => v + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="turn assistant">
      <div className="who">S</div>
      <div className="thinking" role="status">
        <i /><i /><i />
        <span>{seconds >= 3 ? `Thinking · ${seconds}s` : "Thinking"}</span>
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
  const [theme, setTheme] = useState(
    () => localStorage.getItem("sally-theme") || "light"
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("sally-theme", theme);
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#121413" : "#f6f5f1");
  }, [theme]);

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
      window.setTimeout(
        () => setCopiedId((current) => (current === item.id ? null : current)),
        1400
      );
    } catch {
      setError("Clipboard access is unavailable.");
    }
  };

  const tabs = [
    ["chat", "Chat"],
    ["memory", "Memory"],
    ["tools", "Tools"],
    ["system", "System"],
  ];

  const stateLabel =
    online === false ? "Offline" : health?.status === "ok" ? "Ready" : "Connecting";
  const dotClass = `dot ${online === false ? "off" : online ? "on" : ""}`;
  const awaitingFirstToken = busy && !messages.some((item) => item.streaming);

  return (
    <div className="shell">
      <header className="bar">
        <button
          className="ghost"
          onClick={() => setSidebarOpen(true)}
          aria-label="Conversation history"
          title="History"
        >
          <History size={18} />
        </button>

        <div className="brand">
          <strong>SALLY</strong>
          <span className="state" title={stateLabel}>
            <span className={dotClass} />
            <em>{stateLabel}</em>
          </span>
        </div>

        <nav className="tabs" aria-label="Sections">
          {tabs.map(([key, label]) => (
            <button
              key={key}
              className={view === key ? "tab on" : "tab"}
              aria-current={view === key ? "page" : undefined}
              onClick={() => setView(key)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="bar-actions">
          <button
            className="ghost"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle dark mode"
            title="Toggle theme"
          >
            {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <button
            className="ghost"
            onClick={newConversation}
            aria-label="New conversation"
            title="New conversation"
          >
            <Plus size={19} />
          </button>
        </div>
      </header>

      {sidebarOpen && (
        <div className="scrim" onClick={() => setSidebarOpen(false)} />
      )}

      <aside
        className={sidebarOpen ? "drawer open" : "drawer"}
        aria-hidden={!sidebarOpen}
      >
        <div className="drawer-head">
          <strong>Conversations</strong>
          <button
            className="ghost"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close history"
          >
            <X size={18} />
          </button>
        </div>

        <button className="primary wide" onClick={newConversation}>
          <Plus size={16} /> New conversation
        </button>

        <ul className="history">
          {conversations.length === 0 && (
            <li className="muted pad">Nothing saved yet.</li>
          )}
          {conversations.map((conversation) => (
            <li key={conversation.id}>
              <button
                className={
                  conversation.id === conversationId ? "row current" : "row"
                }
                onClick={() => openConversation(conversation.id)}
                tabIndex={sidebarOpen ? 0 : -1}
              >
                <span>{conversation.title}</span>
                <small>{formatDate(conversation.updated_at)}</small>
              </button>
              <button
                className="ghost del"
                onClick={() => deleteConversation(conversation)}
                aria-label={`Delete ${conversation.title}`}
                tabIndex={sidebarOpen ? 0 : -1}
              >
                <Trash2 size={15} />
              </button>
            </li>
          ))}
        </ul>

        <p className="muted foot">
          Stored locally on this machine · {userId.slice(0, 8)}…
        </p>
      </aside>

      {error && (
        <div className="notice" role="alert">
          <span>{error}</span>
          <button className="ghost" onClick={() => setError("")} aria-label="Dismiss">
            <X size={15} />
          </button>
        </div>
      )}

      {view === "chat" && (
        <main className="chat">
          <div className="scroller" ref={scrollRef} onScroll={handleScroll}>
            <div className="column" role="log" aria-live="polite" aria-label="Conversation">
              {messages.length === 0 && !busy ? (
                <div className="hello">
                  <h1>Good to see you.</h1>
                  <p>Ask a question, run a calculation, or just talk.</p>
                  <div className="starters">
                    {STARTERS.map((prompt) => (
                      <button key={prompt} onClick={() => sendMessage(prompt)}>
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {conversationId && (
                    <h2 className="title">{conversationTitle}</h2>
                  )}

                  {messages.map((item) =>
                    item.role === "user" ? (
                      <div className="turn user" key={item.id}>
                        <div className="pill">
                          <MessageContent text={item.content} />
                        </div>
                      </div>
                    ) : (
                      <div className="turn assistant" key={item.id}>
                        <div className="who">S</div>
                        <div className="body">
                          <div className="text">
                            <MessageContent
                              text={item.content}
                              streaming={item.streaming}
                            />
                          </div>

                          {!item.streaming && (
                            <div className="meta">
                              <span>{formatTime(item.created_at)}</span>
                              {item.route && <span>{routeLabel(item.route)}</span>}
                              {item.elapsed_ms != null && (
                                <span>{formatElapsed(item.elapsed_ms)}</span>
                              )}
                              {item.stopped && <span>Stopped</span>}
                              <button
                                className="link"
                                onClick={() => copyMessage(item)}
                              >
                                {copiedId === item.id ? (
                                  <>
                                    <Check size={13} /> Copied
                                  </>
                                ) : (
                                  <>
                                    <Copy size={13} /> Copy
                                  </>
                                )}
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  )}

                  {awaitingFirstToken && <Thinking />}
                </>
              )}
            </div>
          </div>

          <div className="dock">
            <div className="composer">
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
                placeholder={busy ? "SALLY is replying…" : "Message SALLY"}
                rows={1}
                aria-label="Message SALLY"
              />
              {busy ? (
                <button
                  className="send stop"
                  onClick={stopGenerating}
                  aria-label="Stop generating"
                  title="Stop"
                >
                  <Square size={14} fill="currentColor" />
                </button>
              ) : (
                <button
                  className="send"
                  onClick={() => sendMessage()}
                  disabled={!message.trim()}
                  aria-label="Send message"
                >
                  <ArrowUp size={18} />
                </button>
              )}
            </div>
            <p className="hint">
              Enter to send · Shift + Enter for a new line · SALLY can make mistakes
            </p>
          </div>
        </main>
      )}

      {view === "memory" && (
        <main className="page">
          <h1>Memory</h1>
          <p className="muted">What SALLY has saved. Forget anything you don't want kept.</p>

          <form
            className="search"
            onSubmit={(event) => {
              event.preventDefault();
              loadMemories().catch((err) => setError(err.message));
            }}
          >
            <Search size={16} />
            <input
              value={memoryQuery}
              onChange={(event) => setMemoryQuery(event.target.value)}
              placeholder="Search memory"
              aria-label="Search memory"
            />
            {memoryQuery && (
              <button
                type="button"
                className="ghost"
                onClick={() => setMemoryQuery("")}
                aria-label="Clear search"
              >
                <X size={14} />
              </button>
            )}
          </form>

          {memories.length === 0 ? (
            <p className="empty">
              {memoryQuery.trim()
                ? "No matching memories."
                : "No memories yet. Saved facts and preferences will appear here."}
            </p>
          ) : (
            <ul className="list">
              {memories.map((memory) => (
                <li key={memory.id}>
                  <div>
                    <p>{memory.content}</p>
                    <small>
                      <span className="chip">{memory.type}</span>
                      {formatDate(memory.created_at)} · importance{" "}
                      {Math.round(memory.importance * 100)}%
                    </small>
                  </div>
                  <button
                    className="ghost danger"
                    onClick={() => forgetMemory(memory)}
                    aria-label="Forget this memory"
                    title="Forget"
                  >
                    <Trash2 size={15} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </main>
      )}

      {view === "tools" && (
        <main className="page">
          <h1>Tools</h1>
          <p className="muted">What SALLY can do on its own, right now.</p>

          {tools.length === 0 ? (
            <p className="empty">No tools reported.</p>
          ) : (
            <ul className="list">
              {tools.map((tool) => (
                <li key={tool.name}>
                  <div>
                    <p>
                      <strong>{tool.name}</strong>
                      <span className="chip">{tool.safety}</span>
                    </p>
                    <small>{tool.description}</small>
                    <small>
                      {tool.requires_inference
                        ? "Verified result, explained by the model"
                        : "Deterministic"}
                    </small>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </main>
      )}

      {view === "system" && (
        <main className="page">
          <h1>System</h1>
          <p className="muted">
            {shortModel(stats?.model)} · SALLY v{health?.version ?? "—"}
          </p>

          <dl className="facts">
            {[
              ["CPU", stats ? `${stats.cpu_percent}%` : "—"],
              ["RAM", stats ? `${stats.ram_used_gb} / ${stats.ram_total_gb} GB` : "—"],
              ["Disk", stats ? `${stats.disk_percent}% used` : "—"],
              ["Memory database", stats ? `${stats.memory_db_mb} MB` : "—"],
              ["Context window", stats ? `${stats.context_tokens} tokens` : "—"],
              ["Web port", stats?.web_port ?? 5678],
              ["Gateway", stateLabel],
            ].map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{formatRuntime(value)}</dd>
              </div>
            ))}
          </dl>

          <button
            className="secondary"
            onClick={loadEverything}
          >
            Refresh
          </button>
        </main>
      )}
    </div>
  );
}

export default App;
