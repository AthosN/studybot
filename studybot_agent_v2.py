"""
studybot_agent_v2.py — CIS Exam Prep Agent (v2)
=========================================
Standalone command-line agent for AACC CIS students.
Built by Athanasios (Athos) Nathanail, Ph.D.

INSTALL:
  pip install anthropic duckduckgo-search

VERSION: v2 — increased token limits for complete output

RUN:
  python studybot_agent.py

WHAT IT DOES:
  Student states a goal (e.g. "exam on subnetting in 3 days").
  The agent autonomously:
    1. Analyzes the topic (calls Claude with expert system prompt)
    2. Searches the web for current resources (DuckDuckGo, no API key needed)
    3. Builds a detailed hour-by-hour study schedule
    4. Generates a full practice quiz with complete MC options + answer key
    5. Saves everything to a downloadable .txt file

TOOLS:
  analyze_topic          — identify key concepts, exam areas, hard parts
  search_web             — find current study resources (DuckDuckGo)
  build_study_schedule   — detailed hourly plan
  generate_practice_quiz — full quiz with all MC options + answer key
  save_study_plan        — write study pack to file
"""

import anthropic
import json
from datetime import datetime
from pathlib import Path

client = anthropic.Anthropic()


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """You are a CIS Exam Prep Agent for AACC (Anne Arundel Community College) students.

When a student tells you about an upcoming exam or topic they need to study:

STEP 1: Call analyze_topic to understand the full scope of the topic.

STEP 2: Call search_web to find current, high-quality study resources for this topic.
        Search for: '[topic] study guide community college', '[topic] practice exam',
        '[topic] common exam questions'. Use 2-3 targeted searches.

STEP 3: Call build_study_schedule using concepts from Step 1 and resources from Step 2.

STEP 4: Call generate_practice_quiz for a complete, well-formatted practice exam.

STEP 5: Call save_study_plan to write everything to a file the student can keep.

RULES:
- Always use all five tools in order. Do not skip any.
- Start working immediately. Do not ask clarifying questions first.
- Make reasonable assumptions (default to 1.5 hours/day if not stated).
- Incorporate specific resources found in Step 2 into the study schedule and tips.
- After saving the file, provide a short encouraging summary (under 150 words):
  tell the student the filename, what the plan covers, and one specific tip."""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "analyze_topic",
        "description": (
            "Analyze a CIS exam topic. Identify key concepts a student must know, "
            "the types of questions that appear on exams, the areas students find hardest, "
            "and what prerequisite knowledge is assumed. ALWAYS call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The CIS topic (e.g. 'subnetting', 'SQL JOINs', 'OSI model', 'cybersecurity')"
                },
                "student_concerns": {
                    "type": "string",
                    "description": "Specific areas the student mentioned struggling with, if any"
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for current, high-quality study resources on a CIS topic. "
            "Use this to find: study guides, practice exam questions, official documentation, "
            "tutorial resources, and exam prep tips. Call this 2-3 times with specific queries. "
            "Only search for educational content directly relevant to CIS exam preparation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query (e.g. 'subnetting practice problems with solutions', 'OSI model exam questions community college')"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 8)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "build_study_schedule",
        "description": (
            "Create a highly detailed, hour-by-hour study schedule. "
            "Specify exactly what to do each hour — concrete tasks, not generic advice. "
            "Call this after analyze_topic and search_web."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The CIS topic being studied"
                },
                "days_available": {
                    "type": "integer",
                    "description": "Number of days until the exam"
                },
                "key_concepts": {
                    "type": "string",
                    "description": "Key concepts from analyze_topic"
                },
                "web_resources": {
                    "type": "string",
                    "description": "Relevant resources found by search_web to incorporate"
                },
                "hours_per_day": {
                    "type": "number",
                    "description": "Study hours available per day. Default: 1.5"
                }
            },
            "required": ["topic", "days_available", "key_concepts"]
        }
    },
    {
        "name": "generate_practice_quiz",
        "description": (
            "Generate a complete, properly formatted practice quiz. "
            "Multiple choice questions must have all four complete answer options. "
            "Answer key must explain why each answer is correct. "
            "Call this after build_study_schedule."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The CIS topic to test"
                },
                "num_questions": {
                    "type": "integer",
                    "description": "Number of questions to generate (8-12 recommended)"
                },
                "concepts_to_test": {
                    "type": "string",
                    "description": "Specific concepts from analyze_topic to focus on"
                }
            },
            "required": ["topic", "num_questions"]
        }
    },
    {
        "name": "save_study_plan",
        "description": (
            "Save the complete study pack — topic analysis, schedule, quiz, and resources — "
            "to a .txt file. ALWAYS call this last."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Short descriptive filename without extension (e.g. 'subnetting_3day_plan')"
                },
                "topic": {"type": "string"},
                "key_concepts_summary": {
                    "type": "string",
                    "description": "Key concepts and exam areas from analyze_topic"
                },
                "study_schedule": {
                    "type": "string",
                    "description": "Full hour-by-hour schedule from build_study_schedule"
                },
                "practice_quiz": {
                    "type": "string",
                    "description": "Complete quiz with answer key from generate_practice_quiz"
                },
                "resources_and_tips": {
                    "type": "string",
                    "description": "Web resources found and additional study tips"
                }
            },
            "required": ["filename", "topic", "study_schedule", "practice_quiz"]
        }
    }
]


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_topic(topic: str, student_concerns: str = "") -> str:
    """Worker call: deep-dive topic analysis with CIS curriculum expert persona."""
    print(f"   [analyze_topic] {topic}")
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        system=(
            "You are a CIS curriculum expert at a community college. "
            "Analyze the exam topic thoroughly. Return a plain-text breakdown with:\n\n"
            "KEY CONCEPTS (6-8 core ideas the student must understand deeply)\n"
            "COMMON EXAM QUESTION TYPES (4-6 specific question patterns for this topic)\n"
            "HARDEST PARTS (3 areas where students most commonly lose points, and why)\n"
            "PREREQUISITE KNOWLEDGE (what background is assumed)\n"
            "MEMORY AIDS (mnemonics or frameworks to remember key facts)\n\n"
            "Be specific and practical. No generic advice."
        ),
        messages=[{"role": "user", "content": f"Topic: {topic}\nStudent concerns: {student_concerns or 'None stated'}"}]
    )
    return r.content[0].text


def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo (no API key required).
    Returns formatted results with titles, URLs, and summaries.
    """
    print(f"   [search_web] {query!r}")
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
            return "No results found for this query. Try different search terms."
        return f"Search results for: {query!r}\n\n" + "\n\n---\n\n".join(results)
    except ImportError:
        return (
            "Web search unavailable — duckduckgo_search not installed.\n"
            "Run: pip install duckduckgo-search\n"
            "Continuing without web search results."
        )
    except Exception as e:
        return f"Search error ({e}). Continuing without web results for this query."


def build_study_schedule(
    topic: str,
    days_available: int,
    key_concepts: str,
    web_resources: str = "",
    hours_per_day: float = 1.5
) -> str:
    """Worker call: detailed hour-by-hour study schedule."""
    print(f"   [build_study_schedule] {days_available} days @ {hours_per_day}h/day")
    resource_note = f"\n\nWeb resources to incorporate:\n{web_resources}" if web_resources else ""
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4000,
        system=(
            "You are a CIS study coach. Create a highly detailed, hour-by-hour study schedule.\n\n"
            "For EACH DAY use exactly this format:\n\n"
            "============================================================\n"
            "DAY [N]: [Descriptive Title] — [X] hours total\n"
            "============================================================\n\n"
            "HOUR 1 ([HH:MM]-[HH:MM]): [Topic/Activity Title]\n"
            "  * WHAT TO DO: [Specific task — never say 'review notes'. Say exactly what to do.]\n"
            "  * HOW: [Exact method, e.g. 'Write flashcards for each term with definition on back']\n"
            "  * RESOURCE: [Specific resource to use, e.g. website URL, textbook chapter, tool]\n"
            "  * GOAL: [Exactly what you should be able to do by end of this hour]\n\n"
            "HOUR 2 ([HH:MM]-[HH:MM]): [Topic/Activity Title]\n"
            "  [same structure]\n\n"
            "[Continue for all hours]\n\n"
            "END-OF-DAY CHECKPOINT:\n"
            "  Self-test: [Specific thing to quiz yourself on]\n"
            "  You are ready for tomorrow if: [Clear measurable criterion]\n\n"
            "COMMON MISTAKE TO AVOID TODAY: [Most frequent error for this part of the topic]\n\n"
            "----\n\n"
            "Be concrete and specific in every field. Never use generic advice."
        ),
        messages=[{"role": "user", "content": (
            f"Topic: {topic}\nDays: {days_available}\nHours/day: {hours_per_day}\n\n"
            f"Key concepts to cover:\n{key_concepts}{resource_note}"
        )}]
    )
    return r.content[0].text


def generate_practice_quiz(
    topic: str,
    num_questions: int,
    concepts_to_test: str = ""
) -> str:
    """Worker call: full practice quiz with complete MC options and detailed answer key."""
    print(f"   [generate_practice_quiz] {num_questions} questions on: {topic}")
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=8000,
        system=(
            "You are a CIS exam question writer for a community college.\n\n"
            "CRITICAL: Follow this EXACT format. Do not deviate.\n\n"
            "Start with:\n"
            "============================================================\n"
            "PRACTICE QUIZ: [TOPIC IN CAPS]\n"
            "[N] Questions | Mix: Multiple Choice, Short Answer, Scenario\n"
            "============================================================\n\n"
            "For every MULTIPLE CHOICE question use EXACTLY this layout"
            " (leave blank lines between questions):\n\n"
            "Q[N]. [Complete, specific question text ending with a question mark?]\n\n"
            "   A) [Complete answer option as a full phrase or sentence]\n"
            "   B) [Complete answer option as a full phrase or sentence]\n"
            "   C) [Complete answer option as a full phrase or sentence]\n"
            "   D) [Complete answer option as a full phrase or sentence]\n\n"
            "For SHORT ANSWER:\n\n"
            "Q[N]. [Complete question]\n\n"
            "For SCENARIO (include at least 2):\n\n"
            "Q[N]. SCENARIO: [2-3 sentence realistic situation.]\n"
            "      Question: [Specific question about the scenario?]\n\n"
            "After ALL questions, add the separator and answer key:\n\n"
            "============================================================\n"
            "ANSWER KEY\n"
            "============================================================\n\n"
            "For each MULTIPLE CHOICE:\n"
            "Q[N].\n"
            "  Correct answer: [Letter]) [Full text of the correct option]\n"
            "  Explanation: [2-3 sentences explaining the underlying concept "
            "and why this answer is correct]\n"
            "  Most common wrong answer: [Letter]) — Students choose this because "
            "[specific reason], but it is wrong because [specific reason]\n\n"
            "For each SHORT ANSWER:\n"
            "Q[N].\n"
            "  Model answer: [Complete 4-6 sentence answer explaining the concept "
            "fully. Not bullet points.]\n\n"
            "For each SCENARIO:\n"
            "Q[N].\n"
            "  Answer: [Full explanation of the correct approach, the reasoning, "
            "and what would happen with a wrong approach]\n\n"
            "MANDATORY RULES:\n"
            "- At least 50% of questions must be multiple choice.\n"
            "- All four MC options must be plausible — no 'All of the above' "
            "or 'None of the above'.\n"
            "- Every MC option must be a complete phrase or sentence, never a single word.\n"
            "- The answer key must explain concepts, not just label the correct letter.\n"
            "- Leave a blank line between every question in the questions section.\n"
            "- Every question in the answer key must be numbered Q[N] matching the quiz."
        ),
        messages=[{"role": "user", "content": (
            f"Topic: {topic}\n"
            f"Number of questions: {num_questions}\n"
            f"Concepts to focus on: {concepts_to_test or 'all key concepts for this topic'}"
        )}]
    )
    return r.content[0].text


def save_study_plan(
    filename: str,
    topic: str,
    study_schedule: str,
    practice_quiz: str,
    key_concepts_summary: str = "",
    resources_and_tips: str = ""
) -> str:
    """Write the complete study pack to a .txt file."""
    output_dir = Path("study_plans")
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = output_dir / f"{filename}_{ts}.txt"
    divider = "=" * 65
    thin    = "-" * 65

    sections = [
        divider,
        "AACC CIS - PERSONALISED STUDY PLAN",
        f"Topic:   {topic}",
        f"Created: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        "Tool:    StudyBot Agent - Exam Prep Workflow",
        divider, "",
        "KEY CONCEPTS & TOPIC BREAKDOWN", thin,
        key_concepts_summary or "(Concepts are woven into the study schedule below.)", "",
        divider, "STUDY SCHEDULE", thin,
        study_schedule, "",
        divider, "PRACTICE QUIZ", thin,
        practice_quiz, "",
    ]
    if resources_and_tips:
        sections += [divider, "RESOURCES & ADDITIONAL TIPS", thin, resources_and_tips, ""]
    sections += [
        divider,
        "StudyBot Agent - AACC Dept. of Management Information Systems & Applied AI",
        divider,
    ]

    content = "\n".join(sections)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = len(content.encode()) / 1024
    print(f"   [save_study_plan] Saved: {filepath} ({size_kb:.1f} KB)")
    return f"Saved -> {filepath}  ({size_kb:.1f} KB, {len(content):,} characters)"


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, inputs: dict) -> str:
    dispatch = {
        "analyze_topic":          analyze_topic,
        "search_web":             search_web,
        "build_study_schedule":   build_study_schedule,
        "generate_practice_quiz": generate_practice_quiz,
        "save_study_plan":        save_study_plan,
    }
    if name not in dispatch:
        return f"Error: unknown tool '{name}'"
    return dispatch[name](**inputs)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_LABELS = {
    "analyze_topic":          "🔍 Analyzing topic...",
    "search_web":             "🌐 Searching the web...",
    "build_study_schedule":   "📅 Building hour-by-hour schedule...",
    "generate_practice_quiz": "✏️  Generating full quiz with answer key...",
    "save_study_plan":        "💾 Saving study plan...",
}


def run_agent(user_input: str) -> None:
    """
    Main agentic loop. Runs until Claude signals end_turn.
    Prints progress at each tool call.
    """
    print(f"\n{'=' * 65}")
    print("  StudyBot Agent — Starting...")
    print(f"{'=' * 65}\n")

    messages = [{"role": "user", "content": user_input}]
    max_steps = 25  # Increased to allow multiple search calls

    for step in range(max_steps):
        print(f"[Step {step + 1}] Calling Claude...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        print(f"[Step {step + 1}] Stop reason: {response.stop_reason}")

        # Agent finished
        if response.stop_reason == "end_turn":
            final = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\n{'=' * 65}")
            print("  StudyBot Agent — Complete")
            print(f"{'=' * 65}")
            if final:
                print(final)
            print()
            return

        # Agent wants to use tools
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    label = TOOL_LABELS.get(block.name, f"⚙️  {block.name}...")
                    print(f"\n  {label}")

                    # Show input preview
                    preview = json.dumps(block.input, indent=2)
                    if len(preview) > 250:
                        preview = preview[:250] + "\n  ..."
                    print(f"  Input: {preview.replace(chr(10), chr(10) + '  ')}")

                    result = execute_tool(block.name, block.input)

                    result_preview = result[:200] + ("..." if len(result) > 200 else "")
                    print(f"  Result: {result_preview}")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"Unexpected stop reason: {response.stop_reason}. Stopping.")
            return

    print(f"\nAgent stopped after {max_steps} steps (safety limit).")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nStudyBot Exam Prep Agent")
    print("=" * 65)
    print("Examples:")
    print('  "I have a networking exam in 3 days, struggling with subnetting"')
    print('  "Quiz on SQL JOINs tomorrow — never used LEFT JOIN before"')
    print('  "Need to learn cybersecurity definitions for Friday exam"\n')

    user_input = input("You: ").strip()
    if user_input:
        run_agent(user_input)
    else:
        print("No input — exiting.")
