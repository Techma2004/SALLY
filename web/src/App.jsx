import { useEffect, useMemo, useState } from "react";
import {
  Activity, Bot, Brain, Check, ChevronDown, Code2, Copy, Database,
  FileText, Globe, Home, Menu, MessageSquare, MoreVertical, Plus,
  RefreshCw, Search, Send, Settings, ShieldCheck, Sparkles, Terminal,
  ThumbsDown, ThumbsUp, Trash2, Wrench, X
} from "lucide-react";
import "./App.css";

const USER_KEY = "sally-web-user-id";

function getUserId() {
  let id = localStorage.getItem(USER_KEY);
  if (!id) {
    id = globalThis.crypto?.randomUUID?.() || `web-${Date.now()}-${Math.random().toString(36).slice(2)}`;
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

function App() {
  const userId = useMemo(getUserId, []);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState("chat");
  const [conversationId, setConversationId] = useState(null);
  const [conversationTitle, setConversationTitle] = useState("New Conversation");
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stats, setStats] = useState(null);
  const [tools, setTools] = useState([]);
  const [memories, setMemories] = useState([]);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [copiedId, setCopiedId] = useState(null);

  const loadConversations = async () => {
    const response = await fetch(`/conversations?user_id=${encodeURIComponent(userId)}`);
    if (!response.ok) throw new Error("Could not load conversations.");
    const data = await response.json();
    setConversations(data.conversations || []);
  };

  const loadConversation = async (id) => {
    const response = await fetch(
      `/conversations/${encodeURIComponent(id)}?user_id=${encodeURIComponent(userId)}`
    );
    if (!response.ok) throw new Error("Could not load conversation.");
    const data = await response.json();
    setConversationId(data.conversation.id);
    setConversationTitle(data.conversation.title);
    setMessages(data.messages || []);
    setView("chat");
    setSidebarOpen(false);
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

  const loadMemories = async () => {
    const endpoint = memoryQuery.trim()
      ? `/memory?q=${encodeURIComponent(memoryQuery.trim())}&limit=50`
      : "/memory/recent?limit=50";
    const response = await fetch(endpoint);
    if (!response.ok) throw new Error("Could not load memory.");
    const data = await response.json();
    setMemories(data.results || []);
  };

  useEffect(() => {
    Promise.all([loadConversations(), loadSystem(), loadTools(), loadMemories()])
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (view === "system") loadSystem().catch((err) => setError(err.message));
    if (view === "tools") loadTools().catch((err) => setError(err.message));
    if (view === "memory") loadMemories().catch((err) => setError(err.message));
  }, [view]);

  const sendMessage = async () => {
    const text = message.trim();
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          user_id: userId,
          source: "web",
          conversation_id: conversationId,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "SALLY could not process the request.");

      setConversationId(data.conversation_id);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${data.task_id}`,
          role: "assistant",
          content: data.answer,
          created_at: new Date().toISOString(),
          agent_name: data.agent_name,
        },
      ]);
      await loadConversations();

      if (!conversationId) {
        await loadConversation(data.conversation_id);
      }
    } catch (err) {
      setError(err.message);
      setMessages((current) => current.filter((item) => item.id !== optimistic.id));
    } finally {
      setBusy(false);
    }
  };

  const copyMessage = async (item) => {
    try {
      await navigator.clipboard.writeText(item.content);
      setCopiedId(item.id);
      setTimeout(() => setCopiedId(null), 1400);
    } catch {
      setError("Clipboard access is unavailable.");
    }
  };

  const nav = [
    ["chat", MessageSquare, "Conversations"],
    ["memory", Brain, "Memory"],
    ["tools", Wrench, "Tools"],
    ["system", Activity, "System"],
  ];

  return (
    <div className="app">
      {sidebarOpen && <div className="mobile-overlay" onClick={() => setSidebarOpen(false)} />}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="sally-logo"><Bot size={27} /></div>
            <div><h1>SALLY</h1><span>Personal AI</span></div>
          </div>
          <button className="mobile-close" onClick={() => setSidebarOpen(false)}><X size={20} /></button>
        </div>

        <button className="new-chat" onClick={newConversation}>
          <Plus size={18} /> <span>New Conversation</span>
        </button>

        <nav className="navigation">
          {nav.map(([key, Icon, label]) => (
            <button
              key={key}
              className={`nav-item ${view === key ? "active" : ""}`}
              onClick={() => { setView(key); setSidebarOpen(false); }}
            >
              <Icon size={19} /><span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="recent-section">
          <div className="section-title">RECENT CONVERSATIONS</div>
          <div className="recent-list">
            {conversations.length === 0 ? (
              <p className="empty-small">No conversations yet.</p>
            ) : (
              conversations.map((conversation) => (
                <button
                  className={`recent-chat ${conversation.id === conversationId ? "selected" : ""}`}
                  key={conversation.id}
                  onClick={() => loadConversation(conversation.id).catch((err) => setError(err.message))}
                >
                  <span>{conversation.title}</span>
                  <small>{formatDate(conversation.updated_at)}</small>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="profile">
          <div className="profile-avatar"><Terminal size={18} /></div>
          <div className="profile-info"><strong>Local Browser</strong><span>Private session</span></div>
          <ShieldCheck size={17} />
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setSidebarOpen(true)}><Menu size={22} /></button>
          <button className="conversation-title" onClick={() => setView("chat")}>
            <span>{conversationTitle}</span><ChevronDown size={16} />
          </button>
          <div className="top-actions">
            <button title="Refresh data" onClick={() => {
              Promise.all([loadConversations(), loadSystem(), loadTools(), loadMemories()])
                .catch((err) => setError(err.message));
            }}><RefreshCw size={18} /></button>
            <button title="New conversation" onClick={newConversation}><Plus size={19} /></button>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <span>{error}</span><button onClick={() => setError("")}><X size={15} /></button>
          </div>
        )}

        {view === "chat" && (
          <>
            <section className="chat-container">
              <div className="messages">
                {messages.length === 0 ? (
                  <div className="empty-chat">
                    <div className="empty-logo"><Bot size={34} /></div>
                    <h2>Start a conversation</h2>
                    <p>This is the real SALLY gateway. Nothing is preloaded here.</p>
                  </div>
                ) : (
                  messages.map((item) => (
                    <div key={item.id} className={`message-row ${item.role === "user" ? "user" : "assistant"}`}>
                      {item.role !== "user" && <div className="message-avatar"><Bot size={20} /></div>}
                      <div className="message-content">
                        <div className={`message-bubble ${item.role === "user" ? "user-bubble" : ""}`}>
                          <div className="message-text">{item.content}</div>
                          <div className="message-time">
                            {formatTime(item.created_at)}
                            {item.agent_name && <span>· {item.agent_name}</span>}
                          </div>
                        </div>
                        {item.role === "assistant" && (
                          <div className="message-controls">
                            <button onClick={() => copyMessage(item)} title="Copy">
                              {copiedId === item.id ? <Check size={15} /> : <Copy size={15} />}
                            </button>
                            <button title="Helpful"><ThumbsUp size={15} /></button>
                            <button title="Not helpful"><ThumbsDown size={15} /></button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                {busy && (
                  <div className="message-row assistant">
                    <div className="message-avatar"><Bot size={20} /></div>
                    <div className="message-content">
                      <div className="message-bubble"><span className="typing">SALLY is thinking…</span></div>
                    </div>
                  </div>
                )}
              </div>
            </section>

            <div className="composer-wrapper">
              <div className="composer">
                <textarea
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder="Message SALLY…"
                  rows={2}
                  disabled={busy}
                />
                <div className="composer-bottom">
                  <div className="composer-tools">
                    <span className="composer-status"><span /> Local gateway</span>
                  </div>
                  <button className="send-button" onClick={sendMessage} disabled={!message.trim() || busy}>
                    <Send size={17} />
                  </button>
                </div>
              </div>
              <p className="disclaimer">SALLY runs through your local gateway. Verify important information.</p>
            </div>
          </>
        )}

        {view === "memory" && (
          <section className="dashboard-view">
            <div className="view-heading"><div><span className="eyebrow">PERSISTENT MEMORY</span><h2>Memory</h2><p>Stored memories from SALLY's SQLite memory store.</p></div><button className="secondary-button" onClick={() => loadMemories().catch((err) => setError(err.message))}><RefreshCw size={16} /> Refresh</button></div>
            <form className="search-box" onSubmit={(event) => { event.preventDefault(); loadMemories().catch((err) => setError(err.message)); }}>
              <Search size={17} /><input value={memoryQuery} onChange={(event) => setMemoryQuery(event.target.value)} placeholder="Search memory…" /><button type="submit">Search</button>
            </form>
            <div className="data-list">
              {memories.length === 0 ? <div className="empty-state">No memories found.</div> : memories.map((memory) => (
                <article className="data-card" key={memory.id}>
                  <div className="data-card-top"><span className="badge">{memory.type}</span><span>{formatDate(memory.created_at)}</span></div>
                  <p>{memory.content}</p>
                  <small>Importance {Math.round(memory.importance * 100)}%</small>
                </article>
              ))}
            </div>
          </section>
        )}

        {view === "tools" && (
          <section className="dashboard-view">
            <div className="view-heading"><div><span className="eyebrow">RUNTIME</span><h2>Tools</h2><p>Tools actually registered with the current SALLY runtime.</p></div></div>
            <div className="tool-grid">
              {tools.map((tool) => (
                <article className="tool-card" key={tool.name}>
                  <div className="tool-icon"><Wrench size={18} /></div>
                  <div><h3>{tool.name}</h3><p>{tool.description}</p><span className="tool-meta">{tool.requires_inference ? "LLM explanation" : "Deterministic"} · {tool.safety}</span></div>
                </article>
              ))}
            </div>
          </section>
        )}

        {view === "system" && (
          <section className="dashboard-view">
            <div className="view-heading"><div><span className="eyebrow">LIVE RUNTIME</span><h2>System</h2><p>Current machine and SALLY runtime information.</p></div><button className="secondary-button" onClick={() => loadSystem().catch((err) => setError(err.message))}><RefreshCw size={16} /> Refresh</button></div>
            <div className="stats-grid">
              {[
                ["CPU", stats ? `${stats.cpu_percent}%` : "—"],
                ["RAM", stats ? `${stats.ram_used_gb} / ${stats.ram_total_gb} GB` : "—"],
                ["Disk", stats ? `${stats.disk_percent}% used` : "—"],
                ["Memory DB", stats ? `${stats.memory_db_mb} MB` : "—"],
                ["Context", stats ? `${stats.context_tokens} tokens` : "—"],
                ["Model", stats ? stats.model.split("/").pop() : "—"],
              ].map(([label, value]) => (
                <article className="stat-card" key={label}><span>{label}</span><strong>{value}</strong></article>
              ))}
            </div>
            <div className="system-note"><Database size={19} /><div><strong>Local-first runtime</strong><p>SALLY's current web interface is connected directly to the local FastAPI gateway, SQLite memory store, registered tools, and local LLM configuration.</p></div></div>
          </section>
        )}
      </main>

      <aside className="info-panel">
        <div className="sally-profile">
          <div className="large-logo"><Bot size={36} /></div>
          <div><h2>SALLY</h2><p>Science Artificial Learning Logic And You</p><div className="online"><span /> Local gateway</div></div>
        </div>
        <div className="panel-divider" />
        <section className="panel-section">
          <label>RUNTIME</label>
          <div className="runtime-row"><Activity size={17} /><span>{stats ? "Gateway online" : "Loading status…"}</span></div>
          <div className="runtime-row"><Database size={17} /><span>{stats ? `${stats.memory_db_mb} MB memory DB` : "—"}</span></div>
        </section>
        <div className="panel-divider" />
        <section className="panel-section">
          <label>ACTIVE TOOLS</label>
          {tools.slice(0, 6).map((tool) => (
            <div className="capability" key={tool.name}>
              <div className="capability-icon"><Code2 size={17} /></div>
              <div><strong>{tool.name}</strong><p>{tool.description}</p></div>
            </div>
          ))}
          {tools.length === 0 && <p>No tool data available.</p>}
        </section>
        <div className="panel-divider" />
        <section className="panel-section">
          <label>SESSION</label>
          <p>{conversations.length} persisted conversation{conversations.length === 1 ? "" : "s"} in this browser session.</p>
        </section>
      </aside>
    </div>
  );
}

export default App;
