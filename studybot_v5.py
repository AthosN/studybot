"""
StudyBot v5 — CIS Study Assistant
=====================================
Built for AACC by Athanasios (Athos) Nathanail, Ph.D.
Tech stack: Python 3.9+ · Streamlit · Anthropic Claude API (claude-sonnet-4-6)

INSTALL:  pip install streamlit anthropic pdfplumber pypdf streamlit-js-eval
RUN:      streamlit run studybot_v5.py

CHANGES IN v5:
  - Voice input bug fixed: stale transcript never injected again
    (counter-based key + localStorage cleared when recording starts)
  - Embedded agent synced with studybot_agent_v2.py:
    search_web tool, improved prompts, higher token limits

VOICE INPUT FLOW:
  1. Click "Start speaking" — browser records via Web Speech API (Chrome/Edge)
  2. Transcript appears in the voice widget panel
  3. Click "Insert in chat field" — transcript is injected into the chat input box
  4. Review and edit the text if needed, then press Enter to send
"""

import streamlit as st
import streamlit.components.v1 as components
import anthropic
import io
import json
from datetime import datetime
from pathlib import Path

try:
    from streamlit_js_eval import streamlit_js_eval
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

client = anthropic.Anthropic()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

CHAT_SYSTEM_PROMPT = """You are StudyBot, a patient and encouraging study assistant for
CIS (Computer Information Systems) students at Anne Arundel Community College (AACC).

LANGUAGE RULE: Always respond in the exact language the student uses.
If they write in Spanish, respond in Spanish. If they write in French,
respond in French. If they switch languages mid-conversation, switch with them.
Never force English if the student writes in another language.

YOUR ROLE:
- Help students understand CIS concepts clearly and at the right level
- Guide students to think through problems rather than just handing them answers
- Use real-world analogies and familiar examples to make abstract concepts concrete
- Celebrate progress and encourage students who are struggling or frustrated

YOUR TOPICS:
Networking basics (IP addresses, DNS, HTTP, protocols), programming concepts,
databases and SQL, cybersecurity fundamentals, cloud computing, AI and automation,
software development lifecycle, operating systems, IT support operations,
help desk processes, data management, and general technology literacy.

YOUR APPROACH:
- Always use plain, accessible language — define technical terms when you first use them
- If a student seems confused, try a completely different explanation approach
- Keep responses focused: not too long, not too short — calibrated to the question
- Ask a follow-up question to check understanding when appropriate
- If you do not know something or are uncertain, say so clearly and honestly

IMPORTANT LIMITS:
- You are a study assistant, not a replacement for instructor guidance
- Do not complete assignments for students — guide them to the answer
- Always encourage students to attend office hours and contact their instructor
- Acknowledge when a topic needs more depth than a chat conversation can provide

ACCESSIBILITY COMMITMENT:
- Use clear, direct sentences — avoid idioms that may confuse non-native English speakers
- Break complex processes into numbered steps for clarity
- Avoid walls of text — use short paragraphs and spacing where helpful
- Be patient with repeated or basic questions — there are no stupid questions"""

AGENT_SYSTEM_PROMPT = """You are a CIS Exam Prep Agent for AACC students.

When a student tells you about an upcoming exam or topic:

STEP 1: Call analyze_topic to understand the full scope.

STEP 2: Call search_web 2-3 times with targeted queries to find current study resources.
        Example queries: '[topic] study guide', '[topic] practice exam questions',
        '[topic] common mistakes community college'.

STEP 3: Call build_study_schedule using concepts from Step 1 and resources from Step 2.

STEP 4: Call generate_practice_quiz for a complete, well-formatted practice exam.

STEP 5: Call save_study_plan to write everything to a file the student can keep.

Rules:
- Start immediately — no clarifying questions.
- Make reasonable assumptions (default 1.5 hours/day if not stated).
- Use all five steps. After saving, give a short encouraging summary (under 150 words)
  naming the file and key tips from the plan."""

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_TOOLS = [
    {
        "name": "analyze_topic",
        "description": "Analyze a CIS exam topic: key concepts, exam question types, hardest parts. CALL FIRST.",
        "input_schema": {"type": "object",
            "properties": {"topic": {"type": "string"}, "student_concerns": {"type": "string"}},
            "required": ["topic"]}
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for current, high-quality study resources on a CIS topic. "
            "Use this to find study guides, practice questions, official documentation, "
            "and exam prep tips. Call 2-3 times with specific queries. "
            "Only search for educational content relevant to CIS exam preparation."
        ),
        "input_schema": {"type": "object",
            "properties": {
                "query": {"type": "string", "description": "Specific search query"},
                "num_results": {"type": "integer", "description": "Results to return (default 5, max 8)"}
            },
            "required": ["query"]}
    },
    {
        "name": "build_study_schedule",
        "description": "Create a detailed hour-by-hour study schedule. Call after analyze_topic.",
        "input_schema": {"type": "object",
            "properties": {
                "topic": {"type": "string"}, "days_available": {"type": "integer"},
                "key_concepts": {"type": "string"}, "hours_per_day": {"type": "number"}},
            "required": ["topic", "days_available", "key_concepts"]}
    },
    {
        "name": "generate_practice_quiz",
        "description": "Generate a fully formatted quiz with complete MC options and a detailed answer key. Call after build_study_schedule.",
        "input_schema": {"type": "object",
            "properties": {
                "topic": {"type": "string"}, "num_questions": {"type": "integer"},
                "concepts_to_test": {"type": "string"}},
            "required": ["topic", "num_questions"]}
    },
    {
        "name": "save_study_plan",
        "description": "Save the complete study plan to a file. CALL LAST.",
        "input_schema": {"type": "object",
            "properties": {
                "filename": {"type": "string"}, "topic": {"type": "string"},
                "key_concepts_summary": {"type": "string"}, "study_schedule": {"type": "string"},
                "practice_quiz": {"type": "string"}, "study_tips": {"type": "string"}},
            "required": ["filename", "topic", "study_schedule", "practice_quiz"]}
    }
]


def analyze_topic(topic: str, student_concerns: str = "") -> str:
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=8000,
        system=(
            "You are a CIS curriculum expert. Analyze the topic and return a plain-text breakdown:\n"
            "KEY CONCEPTS (5-7 items)\nCOMMON EXAM QUESTION TYPES (3-5 types)\n"
            "HARDEST PARTS (2-3 areas)\nPREREQUISITE KNOWLEDGE\n\nBe specific and practical."
        ),
        messages=[{"role": "user", "content": f"Topic: {topic}\nConcerns: {student_concerns or 'None'}"}])
    return r.content[0].text


def search_web(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo (no API key needed)."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=min(num_results, 8)):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL:   {r.get('href', 'N/A')}\n"
                    f"Info:  {r.get('body', 'N/A')}"
                )
        if not results:
            return f"No results for: {query!r}. Try different terms."
        return f"Search: {query!r}\n\n" + "\n\n---\n\n".join(results)
    except ImportError:
        return "Web search unavailable. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {e}"


def build_study_schedule(topic: str, days_available: int, key_concepts: str, hours_per_day: float = 1.5) -> str:
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=4000,
        system=(
            "You are a CIS study coach. Create a highly detailed hour-by-hour study schedule.\n\n"
            "For EACH DAY use exactly this format:\n\n"
            "DAY [N]: [Title] - [X] hours total\n"
            "----------------------------------------\n"
            "HOUR 1 ([time]-[time]): [Activity]\n"
            "  * WHAT TO DO: [specific task]\n"
            "  * HOW: [exact method]\n"
            "  * GOAL: [what you achieve this hour]\n\n"
            "HOUR 2 ...\n\n"
            "END-OF-DAY CHECK: [specific self-test]\n"
            "WATCH OUT FOR: [most common mistake]\n\nBe concrete."
        ),
        messages=[{"role": "user", "content": f"Topic: {topic}\nDays: {days_available}\nHours/day: {hours_per_day}\nConcepts:\n{key_concepts}"}])
    return r.content[0].text


def generate_practice_quiz(topic: str, num_questions: int, concepts_to_test: str = "") -> str:
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=8000,
        system=(
            "You are a CIS exam question writer. Follow this EXACT format:\n\n"
            "==============================================\n"
            "PRACTICE QUIZ\n"
            "==============================================\n\n"
            "For MULTIPLE CHOICE (at least half):\n\n"
            "Q[N]. [Complete question ending with ?]\n\n"
            "   A) [Complete option as full phrase]\n"
            "   B) [Complete option]\n"
            "   C) [Complete option]\n"
            "   D) [Complete option]\n\n"
            "For SHORT ANSWER:\n\n"
            "Q[N]. [Complete question]\n\n"
            "For SCENARIO (at least one):\n\n"
            "Q[N]. SCENARIO: [2-3 sentence situation.]\n"
            "      Question: [Specific question?]\n\n"
            "Then add:\n\n"
            "==============================================\n"
            "ANSWER KEY\n"
            "==============================================\n\n"
            "For MC:\n"
            "Q[N]. Correct: [Letter]) [full option text]\n"
            "      Why correct: [1-2 sentences explaining the concept]\n"
            "      Most tempting wrong: [Letter]) - [why students choose this]\n\n"
            "For SA: Q[N]. Model answer: [Complete 3-5 sentence answer]\n\n"
            "For scenario: Q[N]. Answer: [Full explanation]\n\n"
            "RULES: Write ALL FOUR options. Never 'All of the above'. Every option = complete phrase."
        ),
        messages=[{"role": "user", "content": f"Topic: {topic}\nQuestions: {num_questions}\nFocus: {concepts_to_test or 'all key concepts'}"}])
    return r.content[0].text


def save_study_plan(filename: str, topic: str, study_schedule: str, practice_quiz: str,
                    key_concepts_summary: str = "", study_tips: str = "") -> str:
    output_dir = Path("study_plans")
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = output_dir / f"{filename}_{ts}.txt"
    divider = "=" * 65
    thin    = "-" * 65
    sections = [
        divider, "AACC CIS - PERSONALISED STUDY PLAN",
        f"Topic:   {topic}", f"Created: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        "Tool:    StudyBot v4 - Exam Prep Agent",
        divider, "",
        "KEY CONCEPTS & TOPIC BREAKDOWN", thin,
        key_concepts_summary or "(See study schedule below.)", "",
        divider, "STUDY SCHEDULE", thin, study_schedule, "",
        divider, "PRACTICE QUIZ", thin, practice_quiz, "",
    ]
    if study_tips:
        sections += [divider, "ADDITIONAL TIPS & RESOURCES", thin, study_tips, ""]
    sections += [divider, "StudyBot v4 - AACC Dept. of Management Information Systems & Applied AI", divider]
    content = "\n".join(sections)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    st.session_state.agent_last_content  = content
    st.session_state.agent_last_filename = f"{filename}_{ts}.txt"
    return f"Saved to {filepath} ({len(content):,} characters)"


def execute_agent_tool(name: str, inputs: dict) -> str:
    dispatch = {
        "analyze_topic":          analyze_topic,
        "search_web":             search_web,
        "build_study_schedule":   build_study_schedule,
        "generate_practice_quiz": generate_practice_quiz,
        "save_study_plan":        save_study_plan,
    }
    return dispatch[name](**inputs) if name in dispatch else f"Unknown tool: {name}"


def run_agent_streamlit(user_input: str) -> str:
    LABELS = {
        "analyze_topic":          "🔍 Analyzing topic...",
        "search_web":             "🌐 Searching the web for resources...",
        "build_study_schedule":   "📅 Building hour-by-hour schedule...",
        "generate_practice_quiz": "✏️  Generating quiz with answer key...",
        "save_study_plan":        "💾 Saving study plan...",
    }
    messages = [{"role": "user", "content": user_input}]
    for _ in range(12):
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=8096,
            system=AGENT_SYSTEM_PROMPT, tools=AGENT_TOOLS, messages=messages)
        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "Done.")
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    st.session_state.agent_log.append(LABELS.get(block.name, f"⚙️ {block.name}..."))
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": execute_agent_tool(block.name, block.input)})
            messages.append({"role": "user", "content": results})
        else:
            return f"Stopped: {response.stop_reason}"
    return "Agent reached step limit."


# ═══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_uploaded_file(uploaded_file) -> tuple:
    file_bytes = uploaded_file.read()
    name = uploaded_file.name
    file_text = ""
    if uploaded_file.type == "application/pdf":
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    parts = [p.extract_text() for p in pdf.pages if p.extract_text()]
                    file_text = "\n".join(parts)
            except Exception:
                pass
        if not file_text and PYPDF_AVAILABLE:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                parts = [p.extract_text() for p in reader.pages if p.extract_text()]
                file_text = "\n".join(parts)
            except Exception:
                pass
        if not file_text:
            return "", f"Warning: {name} returned 0 characters — likely a scanned PDF."
    else:
        file_text = file_bytes.decode("utf-8", errors="replace")
    return file_text, f"Loaded {name} — {len(file_text):,} characters"


# ═══════════════════════════════════════════════════════════════════════════════
# ACCESSIBILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_accessibility_styles() -> None:
    sizes = {"Small": "13px", "Medium": "16px", "Large": "20px", "Extra Large": "24px"}
    fs = sizes.get(st.session_state.get("font_size", "Medium"), "16px")
    hc = st.session_state.get("high_contrast", False)
    bg, fg, card, focus = ("#000","#fff","#111","#ff0") if hc else ("transparent","inherit","inherit","#0D9488")
    st.markdown(f"""<style>
    html,body,.stApp{{background-color:{bg}!important;color:{fg}!important;font-size:{fs}!important;}}
    .stMarkdown,.stText,p,li,label,span{{font-size:{fs}!important;color:{fg}!important;}}
    [data-testid="stChatMessage"]{{background-color:{card}!important;}}
    .stTextInput input,.stTextArea textarea{{font-size:{fs}!important;color:{fg}!important;background-color:{card}!important;}}
    button:focus-visible,input:focus-visible{{outline:3px solid {focus}!important;outline-offset:2px!important;}}
    </style>""", unsafe_allow_html=True)


def tts_component(text: str, auto_speak: bool = False) -> None:
    js_text = json.dumps(text[:1000])
    lang = st.session_state.get("tts_lang", "en-US")
    auto_js = "speak();" if auto_speak else ""
    components.html(f"""
    <div style="display:flex;gap:8px;padding:4px 0 8px;font-family:sans-serif;">
      <button onclick="speak()" style="background:#0D9488;color:#fff;border:none;
        padding:5px 14px;border-radius:6px;cursor:pointer;font-size:13px;">
        🔊 Read aloud</button>
      <button onclick="window.speechSynthesis.cancel()" style="background:transparent;
        border:1px solid #94A3B8;color:#64748B;padding:5px 14px;border-radius:6px;
        cursor:pointer;font-size:13px;">⏹ Stop</button>
    </div>
    <script>
      function speak(){{window.speechSynthesis.cancel();
        var u=new SpeechSynthesisUtterance({js_text});
        u.lang='{lang}';u.rate=0.88;window.speechSynthesis.speak(u);}}
      {auto_js}
    </script>""", height=52)


def inject_into_chat_input(text: str) -> None:
    """
    Populate the st.chat_input textarea with the voice transcript via JavaScript.

    How it works:
      - st.chat_input renders a React-controlled <textarea> with
        data-testid="stChatInput". React overrides the native 'value' setter,
        so we must use Object.getOwnPropertyDescriptor to get the ORIGINAL
        HTMLTextAreaElement setter and call that — otherwise React ignores
        the programmatic value change.
      - After setting the value we dispatch an 'input' event so React detects
        the change and enables the send button.
      - We retry every 100 ms for up to 3 seconds to handle the render delay
        between this sidebar component and the main-area chat input.
      - height=0 keeps the component invisible.
    """
    js_text = json.dumps(text)
    components.html(
        f"""<script>
        (function(){{
          function inject(){{
            var ta = window.parent.document.querySelector(
              '[data-testid="stChatInput"] textarea');
            if (!ta) return false;
            // Use the original HTMLTextAreaElement value setter so React notices
            var setter = Object.getOwnPropertyDescriptor(
              window.parent.HTMLTextAreaElement.prototype, 'value').set;
            setter.call(ta, {js_text});
            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
            ta.focus();
            return true;
          }}
          // Try immediately, then retry until the textarea is in the DOM
          var attempts = 0;
          var timer = setInterval(function() {{
            if (inject() || attempts++ > 30) clearInterval(timer);
          }}, 100);
          inject();
        }})();
        </script>""",
        height=0
    )


def voice_input_widget() -> None:
    """
    Voice recorder using browser Web Speech API (Chrome/Edge only).
    Transcript is written to window.parent.localStorage['studybot_voice'].
    All Streamlit iframes share the same origin (localhost:8501), so the
    parent localStorage is accessible from both this component and streamlit_js_eval.
    """
    components.html("""
    <div style="border:1px solid #E2E8F0;border-radius:8px;padding:10px;
                font-family:sans-serif;font-size:13px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
        <button id="startBtn" onclick="startListening()"
          style="background:#0D9488;color:#fff;border:none;padding:6px 14px;
                 border-radius:6px;cursor:pointer;font-size:13px;">
          🎙️ Start speaking</button>
        <button id="stopBtn" onclick="stopListening()"
          style="display:none;background:#DC2626;color:#fff;border:none;
                 padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px;">
          ⏹ Stop</button>
        <span id="status" aria-live="polite" style="color:#64748B;font-size:12px;">
          Chrome / Edge only</span>
      </div>
      <div id="transcript" aria-live="polite"
        style="padding:6px 10px;background:#F8FAFC;border-radius:5px;
               min-height:32px;color:#1E293B;font-size:13px;">
        Your words will appear here...
      </div>
    </div>
    <script>
    var rec = null;
    function startListening() {
      var SRC = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SRC) { document.getElementById('status').textContent='Use Chrome or Edge.'; return; }
      // Clear any previous transcript before starting a new recording.
      // This prevents the old value from being injected if the user
      // records again without clicking "Insert" in between.
      try { window.parent.localStorage.removeItem('studybot_voice'); } catch(e) {}
      rec = new SRC(); rec.continuous=false; rec.interimResults=true; rec.lang='en-US';
      document.getElementById('startBtn').style.display='none';
      document.getElementById('stopBtn').style.display='inline';
      document.getElementById('status').textContent='Listening...';
      rec.onresult = function(e) {
        var t=''; for(var i=0;i<e.results.length;i++){t+=e.results[i][0].transcript;}
        document.getElementById('transcript').textContent = t;
        if (e.results[e.results.length-1].isFinal) {
          try { window.parent.localStorage.setItem('studybot_voice', t); }
          catch(err) { localStorage.setItem('studybot_voice', t); }
        }
      };
      rec.onerror = function(e) {
        document.getElementById('status').textContent='Error: '+e.error; reset(); };
      rec.onend = function() {
        document.getElementById('status').textContent='Done — click Insert below'; reset(); };
      rec.start();
    }
    function stopListening() { if(rec) rec.stop(); }
    function reset() {
      document.getElementById('startBtn').style.display='inline';
      document.getElementById('stopBtn').style.display='none'; }
    </script>""", height=130)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION LOG
# ═══════════════════════════════════════════════════════════════════════════════

def format_conversation_log() -> str:
    """Format the full conversation as a readable exportable text file."""
    divider = "=" * 60
    thin    = "-" * 45
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    turns = st.session_state.get("total_turns", 0)
    msgs  = st.session_state.get("messages", [])

    lines = [
        divider,
        "STUDYBOT CONVERSATION LOG",
        "AACC - CIS Study Assistant",
        f"Exported:      {now}",
        f"Total turns:   {turns}",
        f"Total messages:{len(msgs)}",
        divider,
        "",
    ]
    for i, msg in enumerate(msgs, 1):
        role_label = "STUDENT" if msg["role"] == "user" else "STUDYBOT"
        lines.append(f"[{role_label}]  (message {i})")
        lines.append(msg["content"])
        lines.append("")
        lines.append(thin)
        lines.append("")

    if not msgs:
        lines.append("(No messages in this session yet.)")
        lines.append("")

    lines += [
        divider,
        "End of conversation log",
        "StudyBot v4 - AACC Dept. of Management Information Systems & Applied AI",
        divider,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="StudyBot v5 - CIS Study Assistant",
    page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

# Session state defaults
_defaults = {
    "messages": [], "total_turns": 0, "pending_message": None,
    "last_reply": "", "tts_should_speak": False,
    "uploaded_context": "", "uploaded_filename": None,
    "high_contrast": False, "font_size": "Medium",
    "auto_speak": False, "tts_lang": "en-US",
    # Voice input (v5 — counter-based, no stale caching)
    "voice_counter":       0,    # incremented on each button click → forces fresh JS eval
    "voice_inject_pending":False,# True while waiting for fresh transcript from JS
    "voice_inject_text":   "",   # transcript queued for DOM injection
    # Agent
    "agent_log": [], "agent_result": "",
    "agent_last_content": "", "agent_last_filename": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

apply_accessibility_styles()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 StudyBot <sup style='font-size:0.6em;color:#0D9488;'>v5</sup>",
                unsafe_allow_html=True)
    st.markdown("_AACC · CIS Study Assistant_")
    st.markdown("---")

    # ── Accessibility ─────────────────────────────────────────────────────────
    st.markdown("### ♿ Accessibility")
    c1, c2 = st.columns(2)
    hc = c1.toggle("High contrast", value=st.session_state.high_contrast, key="_hc")
    if hc != st.session_state.high_contrast:
        st.session_state.high_contrast = hc; st.rerun()
    as_ = c2.toggle("Auto-read", value=st.session_state.auto_speak, key="_as")
    if as_ != st.session_state.auto_speak:
        st.session_state.auto_speak = as_; st.rerun()

    font_opts = ["Small", "Medium", "Large", "Extra Large"]
    fs = st.selectbox("Font size", font_opts, index=font_opts.index(st.session_state.font_size), key="_fs")
    if fs != st.session_state.font_size:
        st.session_state.font_size = fs; st.rerun()

    lang_opts = ["en-US","es-ES","fr-FR","el-GR", "zh-CN","ar-SA","pt-BR","hi-IN","ko-KR","vi-VN"]
    lang_lbl  = ["English (US)","Spanish","French", "Greek", "Mandarin","Arabic","Portuguese","Hindi","Korean","Vietnamese"]
    lv = st.selectbox("Read-aloud language", lang_opts,
                      index=lang_opts.index(st.session_state.tts_lang),
                      format_func=lambda x: lang_lbl[lang_opts.index(x)], key="_lang")
    if lv != st.session_state.tts_lang:
        st.session_state.tts_lang = lv; st.rerun()

    st.markdown("---")

    # ── Voice Input ────────────────────────────────────────────────────────────
    st.markdown("### 🎙️ Voice Input")

    if VOICE_AVAILABLE:
        # ── HOW VOICE INPUT WORKS (v5 — counter-based, no stale caching) ─────────
        #
        # WHAT WAS WRONG IN v4:
        #   voice_ready was cached in session_state on every rerun. A new recording
        #   would set localStorage, but if a Streamlit rerun hadn't picked it up
        #   yet, the OLD cached value got injected instead of the new one.
        #
        # THE FIX:
        #   1. The voice widget JS clears localStorage BEFORE each new recording.
        #   2. The button click increments voice_counter, forcing streamlit_js_eval
        #      to use a NEW key → guarantees a FRESH evaluation of localStorage.
        #   3. No caching in session_state — we always read the live value.
        # ────────────────────────────────────────────────────────────────────────

        _vc = st.session_state.voice_counter

        # Counter-keyed eval: a new key on each click forces fresh JS evaluation
        _vjs = streamlit_js_eval(
            js_expressions=(
                "(function(){"
                "  try { return window.parent.localStorage.getItem('studybot_voice') || ''; }"
                "  catch(e) { return ''; }"
                "})()"
            ),
            key=f"vjs_{_vc}"
        )

        # If button was clicked and we now have a fresh value → queue injection
        if st.session_state.voice_inject_pending:
            _v = str(_vjs).strip() if _vjs else ""
            if _v and _v not in ("None", "null", ""):
                st.session_state.voice_inject_text     = _v
                st.session_state.voice_inject_pending  = False
                # Clear localStorage so next recording starts clean
                streamlit_js_eval(
                    js_expressions=(
                        "(function(){"
                        "  try { window.parent.localStorage.removeItem('studybot_voice'); }"
                        "  catch(e) {}"
                        "  return '';"
                        "})()"
                    ),
                    key=f"vclr_{_vc}"
                )
                st.session_state.voice_counter += 1
                st.rerun()

        voice_input_widget()

        if st.button("📋 Insert in chat field", use_container_width=True,
                     help="Copies your spoken words into the chat input — review and edit before pressing Enter"):
            # Increment counter → forces FRESH localStorage read on next rerun
            st.session_state.voice_counter       += 1
            st.session_state.voice_inject_pending = True
            st.rerun()

    else:
        st.info("Install streamlit-js-eval for voice input.")
        st.code("pip install streamlit-js-eval", language="bash")


    # ── File Upload ────────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Course Material")
    uploaded_file = st.file_uploader(
        "PDF, TXT, or Markdown", type=["txt","md","pdf"],
        help="Lecture notes, syllabus, or study guides for context",
        label_visibility="collapsed"
    )
    if uploaded_file:
        file_text, status_msg = process_uploaded_file(uploaded_file)
        if file_text:
            st.session_state.uploaded_context  = file_text
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(status_msg)
        else:
            st.warning(status_msg)

    if st.session_state.uploaded_filename and st.session_state.uploaded_context:
        st.info(f"📄 Active: **{st.session_state.uploaded_filename}**")
        if st.button("✖ Remove context", use_container_width=True):
            st.session_state.uploaded_context  = ""
            st.session_state.uploaded_filename = None
            st.rerun()

    st.markdown("---")

    with st.expander("📋 System Prompt (AI instructions)", expanded=False):
        st.code(CHAT_SYSTEM_PROMPT, language="text")

    st.markdown("---")

    # ── Session Stats ─────────────────────────────────────────────────────────
    st.markdown("**📊 Session**")
    s1, s2 = st.columns(2)
    s1.metric("Messages", len(st.session_state.messages))
    s2.metric("Turns", st.session_state.total_turns)

    # ── Conversation Export ────────────────────────────────────────────────────
    if st.session_state.messages:
        st.download_button(
            label="📥 Export conversation log",
            data=format_conversation_log(),
            file_name=f"studybot_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download the full conversation as a text file"
        )

    st.markdown("---")

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("**🛠 Tech Stack**")
    st.markdown("- 🐍 Python 3.9+ · Streamlit")
    st.markdown("- 🤖 Claude API `claude-sonnet-4-6`")
    st.markdown("- 📄 pdfplumber · pypdf")
    st.markdown("- 🎙️ Web Speech API (voice, Chrome/Edge)")
    st.markdown("- 🔊 speechSynthesis (TTS)")
    st.markdown("- 💾 Streamlit Session State")

    st.markdown("---")
    if st.button("🗑️ Clear chat", type="secondary", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.total_turns     = 0
        st.session_state.pending_message = None
        st.session_state.last_reply      = ""
        st.session_state.tts_should_speak= False
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — header + tabs
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<h1 style='margin-bottom:2px;'>🤖 StudyBot "
            "<span style='font-size:0.45em;color:#0D9488;font-weight:400;'>v5</span></h1>",
            unsafe_allow_html=True)
st.markdown("**AACC · CIS Study Assistant** — Powered by Claude AI")

tab_chat, tab_agent = st.tabs(["💬 Study Chat", "📋 Exam Prep Agent"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDY CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:

    # ── Voice injection ────────────────────────────────────────────────────────
    # If a transcript is queued, inject it into the chat input now.
    # This block runs BEFORE the chat_input renders, but the JS retries
    # for up to 3 seconds so it finds the textarea after rendering completes.
    if st.session_state.voice_inject_text:
        _inject_txt = st.session_state.voice_inject_text
        st.session_state.voice_inject_text = ""
        inject_into_chat_input(_inject_txt)
        st.info(
            f"🎙️ Voice transcript inserted into the chat field below: "
            f"*\"{_inject_txt[:80]}{'...' if len(_inject_txt) > 80 else ''}\"* "
            f"— edit if needed, then press **Enter** to send."
        )

    st.info(
        "♿ **Accessibility:** Plain-text responses — screen-reader compatible. "
        "Use **Voice Input** in the sidebar (Chrome/Edge). "
        "Toggle **Auto-read** for TTS. Ask in any language."
    )

    st.markdown("**Try one of these to get started:**")
    b1, b2, b3 = st.columns(3)
    if b1.button("💡 What is an IP address?", use_container_width=True):
        st.session_state.pending_message = "Can you explain what an IP address is like I have never heard of it before."
    if b2.button("🦠 Virus vs. malware?", use_container_width=True):
        st.session_state.pending_message = "What is the difference between a virus and malware? Are they the same thing?"
    if b3.button("🌐 Starting networking class", use_container_width=True):
        st.session_state.pending_message = "I am starting a networking class next week with no background. What should I understand first?"

    st.markdown("---")

    # Conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if st.session_state.pending_message:
        prompt = st.session_state.pending_message
        st.session_state.pending_message = None
    else:
        prompt = st.chat_input("Ask StudyBot a CIS question...", key="chat_input_field_v5")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("StudyBot is thinking..."):
                try:
                    effective_system = CHAT_SYSTEM_PROMPT
                    if st.session_state.get("uploaded_context"):
                        effective_system += "\n\nADDITIONAL CONTEXT FROM UPLOADED MATERIAL:\n" + st.session_state.uploaded_context
                    response = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=1000,
                        system=effective_system, messages=st.session_state.messages)
                    reply = response.content[0].text
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.total_turns    += 1
                    st.session_state.last_reply      = reply
                    st.session_state.tts_should_speak = st.session_state.auto_speak
                except anthropic.AuthenticationError:
                    st.error("API key not found. Set ANTHROPIC_API_KEY and restart.")
                except anthropic.RateLimitError:
                    st.warning("Rate limit reached. Please wait and try again.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # TTS
    if st.session_state.last_reply:
        tts_component(st.session_state.last_reply, auto_speak=st.session_state.tts_should_speak)
        st.session_state.tts_should_speak = False

    st.markdown("---")

    # Footer with export button
    fc, fm = st.columns([1, 3])
    with fc:
        if st.session_state.messages:
            st.download_button(
                label="📥 Export log",
                data=format_conversation_log(),
                file_name=f"studybot_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download this conversation as a .txt file"
            )
    with fm:
        st.caption(
            "StudyBot may make mistakes — always verify with your instructor. "
            "| AACC Dept. of Management Information Systems & Applied AI"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXAM PREP AGENT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown("### 📋 Exam Prep Agent")
    st.markdown(
        "Tell the agent what you need to study. It will automatically **analyze the topic**, "
        "build a **detailed hour-by-hour schedule**, generate a **full practice quiz with answer key**, "
        "and produce a **downloadable study plan**."
    )

    with st.form("agent_form"):
        cl, cr = st.columns([3, 1])
        with cl:
            agent_topic    = st.text_input("Topic *", placeholder="e.g. subnetting, SQL JOINs, OSI model")
            agent_concerns = st.text_input("What are you struggling with? (optional)",
                                           placeholder="e.g. I mix up authentication vs authorisation")
        with cr:
            agent_days  = st.number_input("Days until exam", min_value=1, max_value=30, value=3)
            agent_hours = st.number_input("Study hours/day", min_value=0.5, max_value=8.0, value=1.5, step=0.5)
        submitted = st.form_submit_button("🚀 Generate my study plan", type="primary", use_container_width=True)

    if submitted:
        if not agent_topic.strip():
            st.warning("Please enter a topic.")
        else:
            st.session_state.agent_log = []
            st.session_state.agent_result = ""
            st.session_state.agent_last_content = ""
            st.session_state.agent_last_filename = ""
            goal = (
                f"I have an exam on '{agent_topic.strip()}' in {int(agent_days)} day(s). "
                f"I have {agent_hours} hours/day to study. "
                + (f"My specific struggle: {agent_concerns.strip()}" if agent_concerns.strip() else "")
            )
            with st.status("🤖 Agent working...", expanded=True) as agt:
                result = run_agent_streamlit(goal)
                for step in st.session_state.agent_log:
                    st.write(step)
                st.session_state.agent_result = result
                agt.update(label="✅ Study plan ready!", state="complete")

    if st.session_state.agent_result:
        st.markdown("---")
        st.success(st.session_state.agent_result)
        if st.session_state.agent_last_content:
            st.download_button(
                "⬇️ Download study plan (.txt)",
                data=st.session_state.agent_last_content,
                file_name=st.session_state.agent_last_filename or "study_plan.txt",
                mime="text/plain", use_container_width=True
            )
            with st.expander("📄 Preview study plan"):
                st.text(st.session_state.agent_last_content)
