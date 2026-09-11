"""MathMark — automated math solution marking system.

Flask + PostgreSQL (Neon) + Claude Opus 4.7 vision.
Public deployment on Render.
"""
import os, json, base64, time, io, threading, hashlib
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response, abort, g
import psycopg
from psycopg.rows import dict_row
import anthropic

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB per upload

# ────────────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_conn():
    if 'conn' not in g:
        g.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    return g.conn

@app.teardown_appcontext
def close_conn(exc):
    conn = g.pop('conn', None)
    if conn is not None:
        conn.close()

SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  statement TEXT NOT NULL,
  category TEXT,
  difficulty TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS submissions (
  id SERIAL PRIMARY KEY,
  problem_id INT REFERENCES problems(id) ON DELETE CASCADE,
  author_name TEXT NOT NULL,
  image_data BYTEA NOT NULL,
  image_mimetype TEXT NOT NULL,
  image_size INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS markings (
  id SERIAL PRIMARY KEY,
  submission_id INT REFERENCES submissions(id) ON DELETE CASCADE UNIQUE,
  model TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  score INT,
  verdict TEXT,
  feedback_json JSONB,
  raw_response TEXT,
  latency_ms INT,
  cost_usd NUMERIC(10,4),
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subs_problem ON submissions(problem_id);
CREATE INDEX IF NOT EXISTS idx_subs_author  ON submissions(author_name);
CREATE INDEX IF NOT EXISTS idx_marks_sub    ON markings(submission_id);
CREATE INDEX IF NOT EXISTS idx_marks_status ON markings(status);
"""

def init_db():
    conn = psycopg.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()
    conn.close()

# ────────────────────────────────────────────────────────
# LLM marking (Claude Opus OR Gemini 2.5 Pro)
# ────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY","")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY","")
MODEL = os.environ.get("MARK_MODEL", "gemini-2.5-pro")

MARKING_PROMPT = """Sei un esperto insegnante di matematica. Devi valutare la soluzione manoscritta di uno studente al problema qui sotto.

**Problema**:
{problem}

**Istruzioni**:
1. Leggi attentamente il problema.
2. Analizza l'immagine della soluzione manoscritta dello studente.
3. Identifica ogni passaggio della soluzione dello studente.
4. Verifica la correttezza matematica di ciascun passaggio.
5. Se la calligrafia è illeggibile in punti chiave, dillo esplicitamente.
6. Rispondi SOLO con un oggetto JSON valido secondo lo schema richiesto, senza testo aggiuntivo prima o dopo.

**Schema JSON richiesto** (rispondi SOLO con questo JSON, nessun testo aggiuntivo):
```json
{{
  "score": <int 0-100>,
  "verdict": "<uno tra: corretta, parzialmente corretta, errata, illeggibile>",
  "steps_correct": <int>,
  "steps_total": <int>,
  "strengths": [<lista breve di punti positivi in italiano, max 3>],
  "errors": [<lista di errori con posizione nella soluzione, max 5>],
  "suggestions": [<lista di suggerimenti costruttivi in italiano, max 3>],
  "final_answer_correct": <true|false|null se non presente>,
  "confidence": <"alta"|"media"|"bassa" — quanto sei sicuro della lettura della calligrafia>,
  "overall_feedback": "<2-3 frasi in italiano, tono incoraggiante ma onesto>"
}}
```

Ricorda: rispondi SOLO con il JSON, senza wrappers markdown, senza spiegazioni aggiuntive."""

def _parse_json(raw: str):
    """Extract first {...} block from raw model output, stripping markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"): cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        import re
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m: return json.loads(m.group(0))
        raise

def mark_solution_claude(problem_text, image_data, mimetype):
    if not ANTHROPIC_KEY:
        return {"error":"ANTHROPIC_API_KEY not set"}, 0, None, 0
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    t0 = time.time()
    b64 = base64.standard_b64encode(image_data).decode("ascii")
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=2000, temperature=0.2,
            messages=[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":mimetype,"data":b64}},
                {"type":"text","text": MARKING_PROMPT.format(problem=problem_text)},
            ]}])
        latency = int((time.time()-t0)*1000)
        raw = "".join(b.text for b in msg.content if hasattr(b,"text"))
        data = _parse_json(raw)
        cost = (msg.usage.input_tokens * 15 + msg.usage.output_tokens * 75) / 1_000_000
        return data, latency, raw, cost
    except Exception as e:
        return {"error": str(e)}, int((time.time()-t0)*1000), None, 0

def mark_solution_gemini(problem_text, image_data, mimetype):
    if not GEMINI_KEY:
        return {"error":"GEMINI_API_KEY not set"}, 0, None, 0
    import urllib.request
    t0 = time.time()
    b64 = base64.standard_b64encode(image_data).decode("ascii")
    body = json.dumps({
        "contents":[{"parts":[
            {"inline_data":{"mime_type":mimetype,"data":b64}},
            {"text": MARKING_PROMPT.format(problem=problem_text)},
        ]}],
        "generationConfig":{"temperature":0.2,"maxOutputTokens":2000,"responseMimeType":"application/json"}
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        latency = int((time.time()-t0)*1000)
        raw = d["candidates"][0]["content"]["parts"][0]["text"]
        data = _parse_json(raw)
        usage = d.get("usageMetadata", {})
        # Gemini 2.5 Pro pricing ~$1.25/1M in, $5/1M out (Sep 2026)
        cost = (usage.get("promptTokenCount",0) * 1.25 + usage.get("candidatesTokenCount",0) * 5) / 1_000_000
        return data, latency, raw, cost
    except Exception as e:
        return {"error": str(e)}, int((time.time()-t0)*1000), None, 0

def mark_solution(problem_text: str, image_data: bytes, mimetype: str):
    """Dispatch to Claude or Gemini based on MODEL env."""
    if MODEL.startswith("gemini"):
        return mark_solution_gemini(problem_text, image_data, mimetype)
    return mark_solution_claude(problem_text, image_data, mimetype)

# ────────────────────────────────────────────────────────
# Async marking worker
# ────────────────────────────────────────────────────────
_worker_lock = threading.Lock()

def process_submission(submission_id: int):
    """Run marking in background thread."""
    try:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.image_data, s.image_mimetype, p.statement
                FROM submissions s JOIN problems p ON p.id = s.problem_id
                WHERE s.id = %s
            """, (submission_id,))
            row = cur.fetchone()
            if not row: return
            image = bytes(row["image_data"])
            mimetype = row["image_mimetype"]
            problem = row["statement"]

        data, latency_ms, raw, cost = mark_solution(problem, image, mimetype)

        with conn.cursor() as cur:
            if data.get("error"):
                cur.execute("""
                    UPDATE markings SET status='failed', error=%s, raw_response=%s, latency_ms=%s, updated_at=NOW()
                    WHERE submission_id = %s
                """, (data["error"], raw, latency_ms, submission_id))
            else:
                cur.execute("""
                    UPDATE markings
                    SET status='done', score=%s, verdict=%s, feedback_json=%s, raw_response=%s,
                        latency_ms=%s, cost_usd=%s, updated_at=NOW()
                    WHERE submission_id = %s
                """, (data.get("score"), data.get("verdict"), json.dumps(data), raw,
                      latency_ms, cost, submission_id))
            conn.commit()
        conn.close()
    except Exception as e:
        try:
            conn = psycopg.connect(DATABASE_URL)
            with conn.cursor() as cur:
                cur.execute("UPDATE markings SET status='failed', error=%s WHERE submission_id=%s", (str(e), submission_id))
                conn.commit()
            conn.close()
        except: pass

def kick_worker(submission_id):
    t = threading.Thread(target=process_submission, args=(submission_id,), daemon=True)
    t.start()

# ────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────
@app.route("/")
def home():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.id, s.author_name, s.created_at, p.title as problem_title,
                   m.status, m.score, m.verdict
            FROM submissions s
            JOIN problems p ON p.id = s.problem_id
            LEFT JOIN markings m ON m.submission_id = s.id
            ORDER BY s.id DESC
            LIMIT 30
        """)
        subs = cur.fetchall()
        cur.execute("SELECT COUNT(*) c FROM submissions"); n_subs = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM problems"); n_probs = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(DISTINCT author_name) c FROM submissions"); n_auth = cur.fetchone()["c"]
        cur.execute("SELECT AVG(score)::int a FROM markings WHERE status='done'"); avg = cur.fetchone()["a"]
    return render_template("home.html", subs=subs, n_subs=n_subs, n_probs=n_probs, n_auth=n_auth, avg=avg)

@app.route("/new", methods=["GET","POST"])
def new():
    conn = get_conn()
    if request.method == "POST":
        author = request.form.get("author_name","").strip()
        problem_id = request.form.get("problem_id","").strip()
        problem_title = request.form.get("problem_title","").strip()
        problem_statement = request.form.get("problem_statement","").strip()
        problem_category = request.form.get("problem_category","").strip() or None
        problem_difficulty = request.form.get("problem_difficulty","").strip() or None
        image = request.files.get("image")

        if not author: return "Nome autore mancante", 400
        if not image or not image.filename: return "Immagine mancante", 400

        image_bytes = image.read()
        if len(image_bytes) > 20*1024*1024: return "Immagine troppo grande (max 20MB)", 400
        mimetype = image.mimetype or "image/jpeg"
        if mimetype not in ("image/jpeg","image/png","image/webp","image/gif"):
            mimetype = "image/jpeg"

        with conn.cursor() as cur:
            if problem_id:
                cur.execute("SELECT id FROM problems WHERE id=%s", (int(problem_id),))
                if not cur.fetchone(): return "Problema non trovato", 400
                pid = int(problem_id)
            else:
                if not problem_statement: return "Testo del problema mancante", 400
                cur.execute("""INSERT INTO problems (title, statement, category, difficulty)
                               VALUES (%s,%s,%s,%s) RETURNING id""",
                            (problem_title or problem_statement[:80],
                             problem_statement, problem_category, problem_difficulty))
                pid = cur.fetchone()["id"]

            cur.execute("""INSERT INTO submissions (problem_id, author_name, image_data, image_mimetype, image_size)
                           VALUES (%s,%s,%s,%s,%s) RETURNING id""",
                        (pid, author, image_bytes, mimetype, len(image_bytes)))
            sid = cur.fetchone()["id"]
            cur.execute("""INSERT INTO markings (submission_id, model, status)
                           VALUES (%s,%s,'pending')""", (sid, MODEL))
            conn.commit()

        kick_worker(sid)
        return redirect(url_for("submission_detail", sid=sid))

    with conn.cursor() as cur:
        cur.execute("SELECT id, title, statement, category, difficulty FROM problems ORDER BY id DESC LIMIT 200")
        problems = cur.fetchall()
        cur.execute("SELECT DISTINCT author_name FROM submissions ORDER BY author_name LIMIT 50")
        authors = [r["author_name"] for r in cur.fetchall()]
    return render_template("new.html", problems=problems, authors=authors)

@app.route("/submission/<int:sid>")
def submission_detail(sid):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT s.*, p.title as problem_title, p.statement as problem_statement,
                              p.category, p.difficulty
                       FROM submissions s JOIN problems p ON p.id=s.problem_id
                       WHERE s.id=%s""", (sid,))
        sub = cur.fetchone()
        if not sub: abort(404)
        cur.execute("SELECT * FROM markings WHERE submission_id=%s", (sid,))
        mark = cur.fetchone()
    return render_template("submission.html", sub=sub, mark=mark)

@app.route("/submission/<int:sid>/image")
def submission_image(sid):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT image_data, image_mimetype FROM submissions WHERE id=%s", (sid,))
        row = cur.fetchone()
    if not row: abort(404)
    return Response(bytes(row["image_data"]), mimetype=row["image_mimetype"])

@app.route("/submission/<int:sid>/status")
def submission_status(sid):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT status, score, verdict FROM markings WHERE submission_id=%s", (sid,))
        row = cur.fetchone()
    return jsonify(row or {"status":"unknown"})

@app.route("/problems")
def problems_index():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.id, p.title, p.category, p.difficulty, p.created_at,
                   COUNT(s.id) as sub_count,
                   AVG(m.score)::int as avg_score
            FROM problems p
            LEFT JOIN submissions s ON s.problem_id = p.id
            LEFT JOIN markings m ON m.submission_id = s.id AND m.status='done'
            GROUP BY p.id ORDER BY p.id DESC
        """)
        problems = cur.fetchall()
    return render_template("problems.html", problems=problems)

@app.route("/problem/<int:pid>")
def problem_detail(pid):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM problems WHERE id=%s", (pid,))
        problem = cur.fetchone()
        if not problem: abort(404)
        cur.execute("""SELECT s.id, s.author_name, s.created_at,
                              m.status, m.score, m.verdict
                       FROM submissions s LEFT JOIN markings m ON m.submission_id=s.id
                       WHERE s.problem_id=%s ORDER BY s.id DESC""", (pid,))
        subs = cur.fetchall()
    return render_template("problem.html", problem=problem, subs=subs)

@app.route("/author/<name>")
def author_detail(name):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT s.id, s.created_at, p.title as problem_title,
                              m.status, m.score, m.verdict
                       FROM submissions s JOIN problems p ON p.id=s.problem_id
                       LEFT JOIN markings m ON m.submission_id=s.id
                       WHERE s.author_name=%s ORDER BY s.id DESC""", (name,))
        subs = cur.fetchall()
        cur.execute("""SELECT COUNT(*) c, AVG(m.score)::int avg
                       FROM submissions s LEFT JOIN markings m ON m.submission_id=s.id
                       WHERE s.author_name=%s AND m.status='done'""", (name,))
        stats = cur.fetchone()
    return render_template("author.html", name=name, subs=subs, stats=stats)

@app.route("/api/problems")
def api_problems():
    q = request.args.get("q","").strip().lower()
    conn = get_conn()
    with conn.cursor() as cur:
        if q:
            cur.execute("""SELECT id, title, statement FROM problems
                          WHERE LOWER(title) LIKE %s OR LOWER(statement) LIKE %s
                          ORDER BY id DESC LIMIT 20""", (f"%{q}%", f"%{q}%"))
        else:
            cur.execute("SELECT id, title, statement FROM problems ORDER BY id DESC LIMIT 20")
        return jsonify([dict(r) for r in cur.fetchall()])

@app.route("/api/mark_status/<int:sid>")
def api_mark_status(sid):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT status, score, verdict, feedback_json FROM markings WHERE submission_id=%s", (sid,))
        row = cur.fetchone()
    if not row: return jsonify({"status":"unknown"}), 404
    result = {"status": row["status"], "score": row["score"], "verdict": row["verdict"]}
    if row["feedback_json"]: result["feedback"] = row["feedback_json"]
    return jsonify(result)

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.now(timezone.utc).isoformat()})

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
else:
    # init on import for gunicorn
    try:
        init_db()
    except Exception as e:
        print(f"init_db failed: {e}")
