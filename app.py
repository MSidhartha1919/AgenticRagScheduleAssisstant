
import os, re, json, uuid, threading
from time import sleep
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import gradio as gr

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "schedule.json"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(APP_DIR / "chroma_db")))
COLLECTION_NAME = "schedule_events"
MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]
MAX_TOOL_ROUNDS = 6
TODAY = date.today()
WINDOW_END = TODAY + timedelta(days=29)

# ---------- Date/time helpers ----------

def fmt_date(d: date) -> str:
    return d.strftime("%A, %B %d, %Y").replace(" 0", " ")

def parse_date(value: Optional[str], reference: Optional[date] = None) -> date:
    if not value:
        raise ValueError("Date is required.")
    ref = reference or date.today()
    s = value.strip().lower()
    if s == "today":
        return ref
    if s == "tomorrow":
        return ref + timedelta(days=1)
    if s == "yesterday":
        return ref - timedelta(days=1)

    weekdays = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
                "friday":4,"saturday":5,"sunday":6}
    if s.startswith("next "):
        s2 = s[5:]
        if s2 in weekdays:
            delta = (weekdays[s2] - ref.weekday()) % 7
            delta = 7 if delta == 0 else delta
            return ref + timedelta(days=delta)
    if s in weekdays:
        delta = (weekdays[s] - ref.weekday()) % 7
        return ref + timedelta(days=delta)

    for fmt in ("%Y-%m-%d","%d-%m-%Y","%d/%m/%Y","%m/%d/%Y",
                "%B %d %Y","%b %d %Y","%B %d","%b %d",
                "%d %B %Y","%d %b %Y","%d %B","%d %b"):
        try:
            d = datetime.strptime(s, fmt).date()
            if "%Y" not in fmt:
                d = d.replace(year=ref.year)
                if d < ref:
                    d = d.replace(year=ref.year + 1)
            return d
        except ValueError:
            pass
    raise ValueError(f"Could not understand date: {value}")

def parse_time(value: Optional[str]) -> time:
    if not value:
        raise ValueError("Time is required.")
    s = value.strip().lower().replace(".", "")
    for fmt in ("%I:%M %p","%I %p","%H:%M","%H"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"Could not understand time: {value}")

def in_window(d: date) -> bool:
    return TODAY <= d <= WINDOW_END

def window_text() -> str:
    return f"{TODAY.isoformat()} through {WINDOW_END.isoformat()}"

# ---------- Schedule storage ----------

def load_events() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))

def save_events(events: List[Dict[str, Any]]) -> None:
    DATA_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")

def event_dt(e):
    return datetime.fromisoformat(e["start"])

def make_event(title, d, t, duration_minutes=60, event_type="meeting",
               description="", event_id=None):
    start = datetime.combine(d, t)
    end = start + timedelta(minutes=duration_minutes)
    return {
        "id": event_id or str(uuid.uuid4())[:8],
        "title": title,
        "type": event_type,
        "date": d.isoformat(),
        "start": start.isoformat(timespec="minutes"),
        "end": end.isoformat(timespec="minutes"),
        "description": description,
    }

def overlap(a, b):
    return datetime.fromisoformat(a["start"]) < datetime.fromisoformat(b["end"]) and \
           datetime.fromisoformat(b["start"]) < datetime.fromisoformat(a["end"])

# ---------- ChromaDB RAG ----------

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "30-day schedule events"}
)

def event_text(e):
    return (f"{e['title']} | {e['type']} | {e['date']} | "
            f"{e['start']} to {e['end']} | {e.get('description','')}")

def rebuild_index():
    events = load_events()
    ids = [e["id"] for e in events]
    if ids:
        collection.upsert(
            ids=ids,
            documents=[event_text(e) for e in events],
            metadatas=[{
                "event_id": e["id"],
                "date": e["date"],
                "type": e["type"],
                "title": e["title"]
            } for e in events]
        )
    return len(events)

def retrieve_schedule(query="", date_query="", time_query="", top_k=8):
    events = load_events()
    candidates = events[:]

    if date_query:
        d = parse_date(date_query)
        candidates = [e for e in candidates if e["date"] == d.isoformat()]

    if time_query:
        tq = time_query.strip().lower()
        if tq in {"morning","afternoon","evening"}:
            ranges = {
                "morning": (time(6,0), time(12,0)),
                "afternoon": (time(12,0), time(17,0)),
                "evening": (time(17,0), time(22,0))
            }
            lo, hi = ranges[tq]
            candidates = [
                e for e in candidates
                if lo <= datetime.fromisoformat(e["start"]).time() < hi
            ]
        else:
            try:
                target = parse_time(time_query)
                candidates.sort(key=lambda e: abs(
                    (datetime.combine(date.today(), datetime.fromisoformat(e["start"]).time()) -
                     datetime.combine(date.today(), target)).total_seconds()
                ))
            except Exception:
                pass

    if query:
        # Chroma semantic retrieval over schedule documents.
        try:
            res = collection.query(query_texts=[query], n_results=min(top_k, max(1, len(events))))
            ids = set(res.get("ids", [[]])[0])
            semantic = [e for e in events if e["id"] in ids]
            if date_query or time_query:
                semantic = [e for e in semantic if e in candidates]
            if semantic:
                candidates = semantic
            else:
                # Simple lexical fallback.
                q = query.lower()
                candidates = [e for e in candidates if q in event_text(e).lower()] or candidates
        except Exception:
            q = query.lower()
            candidates = [e for e in candidates if q in event_text(e).lower()] or candidates

    candidates.sort(key=lambda e: e["start"])
    return candidates[:max(1, min(int(top_k or 8), 20))]

def get_schedule(date_query="", time_query="", query="", top_k=8):
    try:
        if date_query:
            d = parse_date(date_query)
            if not in_window(d):
                return {"ok": False, "error": f"Schedule only covers the next 30 days: {window_text()}."}
        events = retrieve_schedule(query, date_query, time_query, top_k)
        return {"ok": True, "count": len(events), "events": events, "window": window_text()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

# ---------- Update tool ----------

def update_schedule(action, title="", date_query="", time_query="", new_date="",
                     new_time="", event_type="meeting", description="",
                     event_id="", duration_minutes=60):
    try:
        events = load_events()
        action = action.lower().strip()

        if action == "add":
            d = parse_date(date_query)
            t = parse_time(time_query)
            if not in_window(d):
                return {"ok": False, "error": f"Date must be within the next 30 days: {window_text()}."}
            e = make_event(title or "Untitled event", d, t, int(duration_minutes or 60),
                           event_type or "meeting", description)
            conflicts = [x for x in events if overlap(e, x)]
            if conflicts:
                return {"ok": False, "action":"add", "reason":"conflict", "conflicts":conflicts}
            events.append(e)
            save_events(events); rebuild_index()
            return {"ok": True, "action":"add", "event":e}

        # Locate existing event.
        target = None
        if event_id:
            target = next((e for e in events if e["id"] == event_id), None)
        elif title:
            matches = [e for e in events if title.lower() in e["title"].lower()]
            if date_query:
                d = parse_date(date_query)
                matches = [e for e in matches if e["date"] == d.isoformat()]
            if time_query:
                try:
                    t = parse_time(time_query)
                    matches = [e for e in matches if datetime.fromisoformat(e["start"]).time() == t]
                except Exception:
                    pass
            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                return {"ok":False,"reason":"multiple_matches","matches":matches}

        if not target:
            return {"ok":False,"reason":"not_found","error":"No unique matching event was found."}

        if action in {"delete","remove"}:
            events = [e for e in events if e["id"] != target["id"]]
            save_events(events); rebuild_index()
            return {"ok":True,"action":"delete","event":target}

        if action in {"update","move"}:
            nd = parse_date(new_date) if new_date else date.fromisoformat(target["date"])
            nt = parse_time(new_time) if new_time else datetime.fromisoformat(target["start"]).time()
            if not in_window(nd):
                return {"ok":False,"error":f"New date must be within the next 30 days: {window_text()}."}
            old_start = datetime.fromisoformat(target["start"])
            old_end = datetime.fromisoformat(target["end"])
            duration = int((old_end-old_start).total_seconds()/60)
            replacement = make_event(
                title or target["title"], nd, nt, duration,
                event_type or target["type"], description or target.get("description",""),
                event_id=target["id"]
            )
            conflicts = [x for x in events if x["id"] != target["id"] and overlap(replacement, x)]
            if conflicts:
                return {"ok":False,"action":"update","reason":"conflict","conflicts":conflicts}
            events = [replacement if e["id"] == target["id"] else e for e in events]
            save_events(events); rebuild_index()
            return {"ok":True,"action":"update","old_event":target,"event":replacement}

        return {"ok":False,"error":"Action must be add, update/move, delete, or remove."}
    except Exception as exc:
        return {"ok":False,"error":str(exc)}

# ---------- Exactly two agent tools ----------

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_schedule",
        description="Retrieve relevant schedule entries by date, time, or semantic query. Use for schedule questions, availability, and finding an event before moving it.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "date_query": types.Schema(type="STRING", description="Natural date such as today, tomorrow, Friday, or 2026-10-01."),
                "time_query": types.Schema(type="STRING", description="Time/range such as 3 PM, morning, afternoon, or evening."),
                "query": types.Schema(type="STRING", description="Semantic description of the event or request."),
                "top_k": types.Schema(type="INTEGER", description="Maximum results, 1-20.")
            }
        )
    ),
    types.FunctionDeclaration(
        name="update_schedule",
        description="Add, update/move, or remove a schedule entry. Never claim success unless ok=true.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "action": types.Schema(type="STRING", description="add, update, move, delete, or remove."),
                "title": types.Schema(type="STRING"),
                "date_query": types.Schema(type="STRING"),
                "time_query": types.Schema(type="STRING"),
                "new_date": types.Schema(type="STRING"),
                "new_time": types.Schema(type="STRING"),
                "event_type": types.Schema(type="STRING"),
                "description": types.Schema(type="STRING"),
                "event_id": types.Schema(type="STRING"),
                "duration_minutes": types.Schema(type="INTEGER")
            },
            required=["action"]
        )
    )
]

SYSTEM_PROMPT = f"""
You are an Agentic RAG Schedule Assistant.
The schedule window is {window_text()}.
Use exactly two tools:
1) get_schedule for retrieval.
2) update_schedule for changes.

Rules:
- For schedule questions, use get_schedule.
- For availability questions, retrieve the requested date/time and determine whether events overlap that period.
- For moving an existing event, retrieve it first if its identity is not already unambiguous.
- For add/update/delete, use update_schedule.
- Never invent schedule events.
- Never claim a change succeeded unless the tool returned ok=true.
- If a tool reports a conflict or ambiguity, explain it and ask for the missing detail.
- Always return a useful natural-language answer.
"""

def fallback_answer(message: str) -> str:
    """Guaranteed local fallback that reads the current schedule."""
    m = message.lower().strip()
    try:
        if any(x in m for x in ["what do i have", "schedule", "scheduled", "calendar", "free", "busy", "events"]):
            d = "tomorrow" if "tomorrow" in m else None
            if "today" in m: d = "today"
            if d:
                r = get_schedule(date_query=d, time_query=("afternoon" if "afternoon" in m else "morning" if "morning" in m else "evening" if "evening" in m else None), query=message)
                if r.get("ok"):
                    if not r["events"]:
                        return f"You have no events scheduled for {d}."
                    return "Here is your schedule for " + d + ":\n" + "\n".join(
                        f"- {e['start'][11:16]}–{e['end'][11:16]}: {e['title']} ({e['type']})"
                        for e in r["events"]
                    )
            # Try any weekday mentioned.
            weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            for wd in weekdays:
                if wd in m:
                    r = get_schedule(date_query=wd, time_query=("afternoon" if "afternoon" in m else "morning" if "morning" in m else "evening" if "evening" in m else None), query=message)
                    if r.get("ok"):
                        if not r["events"]:
                            return f"You have no matching events on {wd.title()}."
                        return "Here is your schedule for " + wd.title() + ":\n" + "\n".join(
                            f"- {e['start'][11:16]}–{e['end'][11:16]}: {e['title']} ({e['type']})"
                            for e in r["events"]
                        )
            # Return all matching events rather than a generic error.
            r = get_schedule(query=message)
            if r.get("ok") and r.get("events"):
                return "Matching schedule entries:\n" + "\n".join(
                    f"- {e['start'][0:10]} {e['start'][11:16]}–{e['end'][11:16]}: {e['title']} ({e['type']})"
                    for e in r["events"]
                )
        return ("I can manage your schedule. Try: "
                "\"What do I have scheduled tomorrow?\", "
                "\"Am I free Friday afternoon?\", "
                "\"Add a meeting tomorrow at 3 PM called Project Sync\", or "
                "\"Move my meeting from 2 PM to 4 PM\".")
    except Exception as exc:
        return f"I couldn't complete that request safely. Your schedule was not changed. Details: {exc}"

def run_agent(user_message: str) -> str:
    """
    Main agent function. Every user request enters here.
    Gemini chooses between the two schedule tools; local fallback keeps
    the scheduler useful during temporary Gemini 503/429 errors.
    """
    if not user_message or not user_message.strip():
        return "Please enter a schedule request."

    if genai is None or not os.getenv("GEMINI_API_KEY"):
        return fallback_answer(user_message)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    last_error = None

    for model_name in MODEL_NAMES:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
                    )
                )
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                upper = last_error.upper()
                transient = any(x in upper for x in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "TIMEOUT", "DEADLINE"])
                if transient and attempt == 0:
                    sleep(2)
                    continue
                break

        if last_error:
            continue

        candidate = response.candidates[0]
        parts = candidate.content.parts
        function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        if not function_calls:
            text_parts = [p.text for p in parts if getattr(p, "text", None)]
            answer = "\n".join(text_parts).strip()
            return answer or fallback_answer(user_message)

        contents.append(candidate.content)

        for fc in function_calls:
            args = dict(fc.args or {})
            if fc.name == "get_schedule":
                result = get_schedule(**args)
            elif fc.name == "update_schedule":
                result = update_schedule(**args)
            else:
                result = {"ok": False, "error": f"Unknown tool: {fc.name}"}

            call_id = getattr(fc, "id", None)
            try:
                part = types.Part.from_function_response(
                    name=fc.name,
                    response=result,
                    call_id=call_id,
                )
            except TypeError:
                part = types.Part.from_function_response(
                    name=fc.name,
                    response=result,
                )
            contents.append(types.Content(role="tool", parts=[part]))

        for final_attempt in range(2):
            try:
                final_response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
                    )
                )
                final_parts = final_response.candidates[0].content.parts
                answer = "\n".join(p.text for p in final_parts if getattr(p, "text", None)).strip()
                if answer:
                    return answer
            except Exception as exc:
                last_error = str(exc)
                upper = last_error.upper()
                if final_attempt == 0 and any(x in upper for x in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "TIMEOUT"]):
                    sleep(2)
                    continue
                break

        return fallback_answer(user_message)

    return fallback_answer(user_message)

def chat_fn(message, history):
    return run_agent(message)

# Seed only if schedule.json is missing/empty.
if not DATA_FILE.exists() or not load_events():
    seed = []
    for offset, title, hh, mm, dur, typ, desc in [
        (1,"Project Planning Meeting",10,0,60,"meeting","Weekly product planning."),
        (3,"AI RAG Workshop",14,0,120,"workshop","Hands-on retrieval augmented generation workshop."),
        (5,"Dentist Appointment",11,30,60,"appointment","Routine dental appointment."),
        (7,"Prepare Project Report",9,0,90,"task","Finish report and submit draft."),
        (10,"Team Stand-up",10,0,30,"meeting","Engineering team stand-up."),
        (13,"Cloud Computing Workshop",15,0,120,"workshop","Cloud architecture and deployment."),
        (17,"Client Review Meeting",14,0,60,"meeting","Review project milestone."),
        (20,"Submit Assignment",17,0,30,"task","Submit final assignment."),
        (24,"Doctor Appointment",10,30,60,"appointment","General appointment."),
        (28,"Monthly Planning",16,0,60,"meeting","Plan the next month.")
    ]:
        d = TODAY + timedelta(days=offset)
        seed.append(make_event(title,d,time(hh,mm),dur,typ,desc))
    save_events(seed)

# Build the Chroma index in the background so Render can detect the web port promptly.
threading.Thread(target=rebuild_index, name="chroma-index-init", daemon=True).start()

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🤖 Agentic RAG Schedule Assistant",
    description=(
        f"Manages the next 30 days ({window_text()}) using ChromaDB RAG + "
        f"Gemini tool calling. Exactly two agent tools: get_schedule and update_schedule. "
        f"Every input returns a non-empty response."
    ),
    examples=[
        "What do I have scheduled tomorrow?",
        "Am I free Friday afternoon?",
        "Show me my workshops.",
        "Add a meeting tomorrow at 3 PM called Project Sync.",
        "Move my Project Planning Meeting from 10 AM to 4 PM.",
        "Delete my dentist appointment."
    ]
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT","7860"))
    )
