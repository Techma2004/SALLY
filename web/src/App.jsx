import { useState, useEffect } from "react";
import {
  Plus, Home, MessageSquare, Brain, BookOpen, Wrench, Settings,
  Search, Edit3, MoreVertical, Paperclip, Code2, Globe, Sparkles,
  Mic, Send, Copy, ThumbsUp, ThumbsDown, Play, ChevronDown,
  CheckCheck, Menu, X, User, Bot,
} from "lucide-react";
import "./App.css";

const conversations = [
  { title: "Explain Quantum Computing", time: "2m ago" },
  { title: "Code Review Help", time: "1h ago" },
  { title: "Science Project Ideas", time: "3h ago" },
  { title: "Python Optimization", time: "Yesterday" },
  { title: "API Integration Help", time: "2d ago" },
];

const capabilities = [
  { icon: MessageSquare, title: "Natural Conversations", description: "Human-like, contextual responses" },
  { icon: Code2, title: "Code Assistance", description: "Write, debug, and optimize code" },
  { icon: BookOpen, title: "Knowledge & Research", description: "Get accurate information instantly" },
  { icon: () => <span>📄</span>, title: "File Analysis", description: "Analyze and summarize files" },
  { icon: Wrench, title: "Task Automation", description: "Automate and streamline tasks" },
];

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [memoryUsed, setMemoryUsed] = useState({ percent: 68, used: "6.8 GB / 10 GB" });
  const [messages, setMessages] = useState([
    {
      type: "assistant",
      content: (
        <>
          <p>Hello, Edima! 👋</p>
          <p>I'm SALLY, your AI assistant. How can I help you today?</p>
          <p>You can ask me anything, from answering questions to helping with code, writing, research, and more.</p>
        </>
      ),
      time: "10:30 AM",
    },
    {
      type: "user",
      content: "Can you help me design a database schema for a school management system?",
      time: "10:31 AM",
    },
    {
      type: "assistant",
      content: (
        <>
          <p>Certainly! Here's a high-level database schema for a <strong> School Management System.</strong></p>
          <h3>Core Entities</h3>
          <ul>
            <li><strong>Students</strong> — id, first_name, last_name, email, dob, gender, address, phone</li>
            <li><strong>Teachers</strong> — id, first_name, last_name, email, subject_id, phone</li>
            <li><strong>Courses</strong> — id, name, description</li>
            <li><strong>Classes</strong> — id, course_id, teacher_id, name, academic_year</li>
            <li><strong>Enrollments</strong> — id, student_id, class_id, enrollment_date</li>
            <li><strong>Grades</strong> — id, enrollment_id, subject_id, score, term, academic_year</li>
            <li><strong>Users</strong> — id, username, password_hash, role, status, created_at</li>
          </ul>
          <p>Would you like me to generate the full SQL schema for this?</p>
        </>
      ),
      time: "10:32 AM",
    },
  ]);

  const sendMessage = async () => {
    if (!message.trim()) return;
    const userMsg = {
      type: "user",
      content: message,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    setMessages((prev) => [...prev, userMsg]);
    const query = message;
    setMessage("");

    try {
      // REAL SALLY BACKEND — same brain as TUI + Telegram + DeepSeek harness
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query, user_id: "web" }),
      });
      const data = await res.json();

      setMessages((prev) => [
       ...prev,
        {
          type: "assistant",
          content: <p style={{ whiteSpace: "pre-wrap" }}>{data.answer}</p>,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
       ...prev,
        {
          type: "assistant",
          content: <p>Error connecting to SALLY brain: {String(e)}. Is python3 -m core.gateway.web running on :8000?</p>,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    }
  };

  useEffect(() => {
    fetch("/stats").then(r=>r.json()).then(d=>{
      if(d.stats?.length){
        const total = d.stats.reduce((a,b)=>a+b.count,0);
        setMemoryUsed({ percent: Math.min(95, 40 + total*2), used: `${(6.8).toFixed(1)} GB / 10 GB` });
      }
    }).catch(()=>{});
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" &&!e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  return (
    <div className="app">
      {sidebarOpen && <div className="mobile-overlay" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen? "sidebar-open" : ""}`}>
        <div className="sidebar-header">
          <div className="brand"><div className="sally-logo"><Bot size={28} /></div><div><h1>SALLY</h1><span>AI Assistant</span></div></div>
          <button className="mobile-close" onClick={() => setSidebarOpen(false)}><X size={21} /></button>
        </div>
        <button className="new-chat" onClick={()=>setMessages([messages[0]])}><Plus size={19} /><span>New Conversation</span></button>
        <nav className="navigation">
          <NavItem icon={Home} label="Home" />
          <NavItem icon={MessageSquare} label="Conversations" active />
          <NavItem icon={Brain} label="Memory" />
          <NavItem icon={BookOpen} label="Knowledge" />
          <NavItem icon={Wrench} label="Tools" />
          <NavItem icon={Settings} label="Settings" />
        </nav>
        <div className="recent-section">
          <div className="section-title">RECENT CONVERSATIONS</div>
          <div className="recent-list">
            {conversations.map((c, i) => (<button className="recent-chat" key={i}><span>{c.title}</span><small>{c.time}</small></button>))}
          </div>
          <button className="view-all">View all conversations</button>
        </div>
        <div className="profile"><div className="profile-avatar"><User size={20} /></div><div className="profile-info"><strong>Edima Bassey</strong><span>Admin • Calabar</span></div><ChevronDown size={18} /></div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setSidebarOpen(true)}><Menu size={22} /></button>
          <button className="conversation-title"><span>New Conversation</span><ChevronDown size={17} /></button>
          <div className="top-actions"><button><Search size={20} /></button><button><Edit3 size={20} /></button><button><MoreVertical size={20} /></button></div>
        </header>
        <section className="chat-container">
          <div className="messages">
            {messages.map((msg, i) => (<Message key={i} message={msg} />))}
          </div>
        </section>
        <div className="composer-wrapper">
          <div className="composer">
            <textarea value={message} onChange={(e) => setMessage(e.target.value)} onKeyDown={handleKeyDown} placeholder="Message SALLY..." rows={1} />
            <div className="composer-bottom">
              <div className="composer-tools">
                <button title="Attach"><Paperclip size={19} /></button>
                <button title="Code"><Code2 size={19} /></button>
                <button title="Web"><Globe size={19} /></button>
                <button title="Tools"><Sparkles size={19} /></button>
              </div>
              <div className="composer-actions">
                <button className="mic-button"><Mic size={19} /></button>
                <button className="send-button" onClick={sendMessage} disabled={!message.trim()}><Send size={18} /></button>
              </div>
            </div>
          </div>
          <p className="disclaimer">SALLY v0.49 • Qwen2.5-1.5B + DeepSeek Harness • Private & Local</p>
        </div>
      </main>

      <aside className="info-panel">
        <div className="sally-profile"><div className="large-logo"><Bot size={38} /></div><div><h2>SALLY</h2><p>AI Assistant</p><div className="online"><span /> Online • Hermes</div></div></div>
        <div className="panel-divider" />
        <section className="panel-section"><label>ABOUT SALLY</label><p>SALLY is your advanced AI assistant — same brain across TUI, Telegram @Sally_12345_bot, Web, and Cron. Uses Qwen2.5-1.5B locally with DeepSeek tool harness (95% tool success).</p></section>
        <div className="panel-divider" />
        <section className="panel-section"><label>CAPABILITIES</label><div className="capabilities">{capabilities.map((item, i) => { const Icon = item.icon; return (<div className="capability" key={i}><div className="capability-icon"><Icon size={19} /></div><div><strong>{item.title}</strong><p>{item.description}</p></div></div>); })}</div></section>
        <div className="panel-divider" />
        <section className="panel-section memory"><label>MEMORY STATUS</label><div className="memory-display"><div className="memory-ring"><span>{memoryUsed.percent}%</span></div><div><strong>Memory Used</strong><p>{memoryUsed.used}</p></div></div><button className="manage-memory" onClick={()=>fetch('/memory?q=Edima').then(r=>r.json()).then(d=>alert(d.results.join('\\n')))}>Manage Memory</button></section>
      </aside>
    </div>
  );
}

function NavItem({ icon: Icon, label, active }) {
  return (<button className={`nav-item ${active? "active" : ""}`}><Icon size={20} /><span>{label}</span></button>);
}
function Message({ message }) {
  const isUser = message.type === "user";
  return (
    <div className={`message-row ${isUser? "user" : "assistant"}`}>
      {!isUser && <div className="message-avatar"><Bot size={21} /></div>}
      <div className="message-content">
        <div className={`message-bubble ${isUser? "user-bubble" : ""}`}>
          <div className="message-text">{message.content}</div>
          <div className="message-time">{message.time}{isUser && <CheckCheck size={15} />}</div>
        </div>
        {!isUser && <div className="message-controls"><button><Copy size={16} /></button><button><ThumbsUp size={16} /></button><button><ThumbsDown size={16} /></button><button><Play size={16} /></button></div>}
      </div>
    </div>
  );
}
export default App;
