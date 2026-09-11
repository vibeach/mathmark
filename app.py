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
from PIL import Image, ImageOps
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

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
  language TEXT NOT NULL DEFAULT 'it',
  created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'it';

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

SUPPORTED_LANGS = {"it", "en", "ru"}

_LANG_SPEC = {
    "it": {
        "verdicts": ["corretta", "parzialmente corretta", "errata", "illeggibile"],
        "confidences": ["alta", "media", "bassa"],
        "instr": ("Sei un esperto insegnante di matematica. Valuta la soluzione manoscritta "
                  "dello studente. Rispondi in italiano. La calligrafia potrebbe essere in "
                  "qualsiasi lingua o notazione — leggi con attenzione."),
    },
    "en": {
        "verdicts": ["correct", "partially correct", "wrong", "illegible"],
        "confidences": ["high", "medium", "low"],
        "instr": ("You are an expert math teacher. Grade the student's handwritten solution. "
                  "Respond in English. The handwriting may be in any language or notation — "
                  "read carefully."),
    },
    "ru": {
        "verdicts": ["правильно", "частично правильно", "неверно", "нечитаемо"],
        "confidences": ["высокая", "средняя", "низкая"],
        "instr": ("Вы — опытный учитель математики. Оцените рукописное решение ученика. "
                  "Отвечайте на русском языке. Почерк может быть на любом языке или в любой "
                  "нотации — читайте внимательно."),
    },
}

def build_marking_prompt(problem: str, lang: str) -> str:
    spec = _LANG_SPEC.get(lang, _LANG_SPEC["it"])
    verdicts = " | ".join(spec["verdicts"])
    confidences = " | ".join(spec["confidences"])
    return f"""{spec['instr']}

**Problem** / **Problema** / **Задача**:
{problem}

Return a single JSON object. `verdict_code` MUST be one of the canonical English codes
`correct` / `partial` / `wrong` / `illegible`. The `verdict` field is the same meaning
localized in the target language. All free-text fields (`strengths`, `errors`,
`suggestions`, `overall_feedback`) MUST be written in the target language.

Schema:
{{
  "score": <int 0-100>,
  "verdict_code": "correct" | "partial" | "wrong" | "illegible",
  "verdict": "<one of: {verdicts}>",
  "steps_correct": <int>,
  "steps_total": <int>,
  "strengths": [<up to 3 short strings, target language>],
  "errors": [
    {{"step": "<the exact step or expression where the error is, verbatim from the image>",
      "explanation": "<one short sentence in the target language>"}},
    ...up to 5
  ],
  "suggestions": [<up to 3 short strings, target language>],
  "final_answer_correct": <true | false | null>,
  "confidence": "<one of: {confidences}>",
  "overall_feedback": "<2-3 sentences, encouraging but honest, target language>"
}}

Respond with ONLY the JSON object, no markdown fences, no extra text."""

GEMINI_ACCEPTED = ("image/jpeg","image/png","image/webp","image/gif")

def normalize_image(data: bytes, claimed_mime: str | None):
    """Ensure the image is decodable + in a Gemini-accepted mime.

    - Convert HEIC/HEIF and anything unrecognized to JPEG.
    - Rotate per EXIF, strip metadata, cap max side at 2000px.
    - Return (bytes, mime).
    """
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        fmt = (img.format or "").upper()
        max_side = 2000
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        if fmt in ("JPEG","PNG","WEBP","GIF"):
            out = io.BytesIO()
            mime = {"JPEG":"image/jpeg","PNG":"image/png","WEBP":"image/webp","GIF":"image/gif"}[fmt]
            save_kwargs = {"quality": 88, "optimize": True} if fmt == "JPEG" else {}
            if img.mode not in ("RGB","RGBA","L") and fmt != "PNG":
                img = img.convert("RGB")
            img.save(out, fmt, **save_kwargs)
            return out.getvalue(), mime
        # Anything else (HEIC/HEIF/TIFF/BMP/...) → JPEG
        if img.mode != "RGB": img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, "JPEG", quality=88, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        mime = claimed_mime if claimed_mime in GEMINI_ACCEPTED else "image/jpeg"
        return data, mime

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

def mark_solution_claude(problem_text, image_data, mimetype, lang="it"):
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
                {"type":"text","text": build_marking_prompt(problem_text, lang)},
            ]}])
        latency = int((time.time()-t0)*1000)
        raw = "".join(b.text for b in msg.content if hasattr(b,"text"))
        data = _parse_json(raw)
        cost = (msg.usage.input_tokens * 15 + msg.usage.output_tokens * 75) / 1_000_000
        return data, latency, raw, cost
    except Exception as e:
        return {"error": str(e)}, int((time.time()-t0)*1000), None, 0

def mark_solution_gemini(problem_text, image_data, mimetype, lang="it"):
    if not GEMINI_KEY:
        return {"error":"GEMINI_API_KEY not set"}, 0, None, 0
    import urllib.request
    t0 = time.time()
    b64 = base64.standard_b64encode(image_data).decode("ascii")
    body = json.dumps({
        "contents":[{"parts":[
            {"inline_data":{"mime_type":mimetype,"data":b64}},
            {"text": build_marking_prompt(problem_text, lang)},
        ]}],
        "generationConfig":{
            "temperature":0.2,
            "maxOutputTokens":8192,
            "responseMimeType":"application/json",
            "thinkingConfig":{"thinkingBudget":2048},
        },
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        latency = int((time.time()-t0)*1000)
        cand = (d.get("candidates") or [{}])[0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        raw = "".join(p.get("text","") for p in parts if isinstance(p, dict))
        if not raw:
            block = d.get("promptFeedback", {}).get("blockReason")
            return {"error": f"empty response (finishReason={finish}, block={block})"}, latency, json.dumps(d)[:2000], 0
        data = _parse_json(raw)
        usage = d.get("usageMetadata", {})
        cost = (usage.get("promptTokenCount",0) * 1.25 + usage.get("candidatesTokenCount",0) * 5) / 1_000_000
        return data, latency, raw, cost
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}, int((time.time()-t0)*1000), None, 0

def mark_solution(problem_text: str, image_data: bytes, mimetype: str, lang: str = "it"):
    """Dispatch to Claude or Gemini based on MODEL env."""
    if MODEL.startswith("gemini"):
        return mark_solution_gemini(problem_text, image_data, mimetype, lang)
    return mark_solution_claude(problem_text, image_data, mimetype, lang)

# ────────────────────────────────────────────────────────
# UI i18n
# ────────────────────────────────────────────────────────
UI_LANGS = ("en", "ru")
UI_DEFAULT = "en"

TEXT = {
    "en": {
        "html_lang": "en",
        "brand_tail": "Mark",
        "nav_recent": "Recent",
        "nav_problems": "Problems",
        "nav_new": "＋ New",
        "footer": "MathMark · automated math solution grading",
        "home_title": "Recent submissions",
        "home_sub": "Handwritten math solutions, graded automatically.",
        "stat_submissions": "Submissions",
        "stat_problems": "Problems",
        "stat_authors": "Authors",
        "stat_avg": "Avg. score",
        "table_id": "#",
        "table_author": "Author",
        "table_problem": "Problem",
        "table_date": "Date",
        "table_score": "Score",
        "table_verdict": "Verdict",
        "table_status": "Status",
        "no_subs": "No submissions yet.",
        "new_h1": "New submission",
        "new_sub": "Photograph the solution, pick the problem (or create a new one), enter the author's name.",
        "step_author": "1 · Author",
        "author_label": "Name of the person who wrote the solution",
        "author_ph": "e.g. Anna Ivanova",
        "step_problem": "2 · Problem",
        "reuse_problem": "Reuse problem",
        "new_problem": "New problem",
        "search_existing": "Search existing problems",
        "search_ph": "Search by title or text…",
        "no_problems": "No problems found.",
        "title_label": "Problem title (optional)",
        "title_ph": "e.g. Quadratic equation #1",
        "statement_label": "Problem statement",
        "statement_ph": "e.g. Solve: $x^2 - 5x + 6 = 0$\n\nHint: use inline LaTeX with $...$ and block with $$...$$",
        "category_label": "Category (optional)",
        "category_ph": "algebra, geometry, calculus…",
        "difficulty_label": "Difficulty (optional)",
        "diff_none": "—",
        "diff_easy": "easy",
        "diff_medium": "medium",
        "diff_hard": "hard",
        "diff_veryhard": "very hard",
        "step_photo": "3 · Solution photo",
        "photo_label": "Take or upload a photo",
        "photo_hint": "Tip: make sure the writing is clear and well lit. Max 20 MB (jpg, png, webp).",
        "step_lang": "4 · Grading language",
        "lang_label": "Language of feedback",
        "submit": "➤ Submit for grading",
        "submitting": "⏳ Uploading + analysing…",
        "err_author_missing": "Author name missing",
        "err_image_missing": "Image missing",
        "err_image_too_big": "Image too large (max 20 MB)",
        "err_problem_not_found": "Problem not found",
        "err_statement_missing": "Problem statement missing",
        "sub_title": "Submission",
        "sub_author": "Author",
        "sub_problem": "Problem",
        "sub_submitted": "Submitted",
        "sub_image": "Handwritten solution",
        "sub_marking": "Grading",
        "pending_hint": "Grading in progress — this page will refresh automatically.",
        "mark_score": "Score",
        "mark_confidence": "Reading confidence:",
        "mark_steps": "Correct steps:",
        "mark_final": "Final answer:",
        "mark_final_ok": "✓ correct",
        "mark_final_bad": "✗ wrong",
        "mark_strengths": "Strengths",
        "mark_errors": "Errors found",
        "mark_suggestions": "Suggestions",
        "mark_model": "Model:",
        "mark_failed": "Grading failed. Error:",
        "problems_title": "Problem catalogue",
        "problems_sub": "All problems previously submitted, with their attempt counts and average score.",
        "problem_id": "#",
        "problem_title_col": "Title",
        "problem_category": "Category",
        "problem_difficulty": "Difficulty",
        "problem_subs": "Submissions",
        "problem_avg": "Avg.",
        "problem_created": "Created",
        "problem_statement_h": "Problem statement",
        "problem_subs_h": "Submitted solutions",
        "author_stats_c": "Graded solutions:",
        "author_stats_avg": "Average score:",
        "author_history_h": "Solution history",
        "verdict_correct": "correct",
        "verdict_partial": "partially correct",
        "verdict_wrong": "wrong",
        "verdict_illegible": "illegible",
        "nav_submissions": "All",
        "view_all_subs": "View all submissions →",
        "subs_title": "All submissions",
        "subs_sub": "Every submission ever graded. Click any row to open the full AI review.",
        "subs_search_ph": "Search by title, statement or author…",
        "subs_filter_author": "Author",
        "subs_filter_verdict": "Verdict",
        "subs_filter_status": "Status",
        "subs_filter_lang": "Feedback language",
        "subs_filter_all": "All",
        "subs_reset": "Reset filters",
        "subs_apply": "Apply",
        "subs_page_of": "Page {page} of {pages}",
        "subs_total": "{n} submissions",
        "subs_prev": "← Previous",
        "subs_next": "Next →",
        "st_pending": "pending",
        "st_done": "done",
        "st_failed": "failed",
        "regrade_btn": "Re-grade",
        "regrade_confirm": "Re-run grading on this solution?",
        "regrade_running": "Re-grading…",
    },
    "ru": {
        "html_lang": "ru",
        "brand_tail": "Mark",
        "nav_recent": "Последние",
        "nav_problems": "Задачи",
        "nav_new": "＋ Новое",
        "footer": "MathMark · автоматическая проверка математических решений",
        "home_title": "Последние работы",
        "home_sub": "Рукописные решения по математике, оценённые автоматически.",
        "stat_submissions": "Работ",
        "stat_problems": "Задач",
        "stat_authors": "Авторов",
        "stat_avg": "Ср. балл",
        "table_id": "№",
        "table_author": "Автор",
        "table_problem": "Задача",
        "table_date": "Дата",
        "table_score": "Балл",
        "table_verdict": "Вердикт",
        "table_status": "Статус",
        "no_subs": "Пока нет работ.",
        "new_h1": "Новая работа",
        "new_sub": "Сфотографируйте решение, выберите задачу (или создайте новую), введите имя автора.",
        "step_author": "1 · Автор",
        "author_label": "Имя автора решения",
        "author_ph": "Напр. Анна Иванова",
        "step_problem": "2 · Задача",
        "reuse_problem": "Использовать существующую",
        "new_problem": "Новая задача",
        "search_existing": "Поиск среди существующих задач",
        "search_ph": "Поиск по названию или тексту…",
        "no_problems": "Задачи не найдены.",
        "title_label": "Название задачи (необязательно)",
        "title_ph": "Напр. Квадратное уравнение №1",
        "statement_label": "Условие задачи",
        "statement_ph": "Напр.: Решите $x^2 - 5x + 6 = 0$\n\nПодсказка: используйте LaTeX через $...$ и $$...$$",
        "category_label": "Раздел (необязательно)",
        "category_ph": "алгебра, геометрия, анализ…",
        "difficulty_label": "Сложность (необязательно)",
        "diff_none": "—",
        "diff_easy": "лёгкая",
        "diff_medium": "средняя",
        "diff_hard": "сложная",
        "diff_veryhard": "очень сложная",
        "step_photo": "3 · Фото решения",
        "photo_label": "Сделайте или загрузите фото",
        "photo_hint": "Совет: убедитесь, что запись чёткая и хорошо освещена. Макс. 20 МБ (jpg, png, webp).",
        "step_lang": "4 · Язык проверки",
        "lang_label": "Язык обратной связи",
        "submit": "➤ Отправить на проверку",
        "submitting": "⏳ Загрузка и проверка…",
        "err_author_missing": "Не указано имя автора",
        "err_image_missing": "Не приложено изображение",
        "err_image_too_big": "Файл слишком большой (макс. 20 МБ)",
        "err_problem_not_found": "Задача не найдена",
        "err_statement_missing": "Отсутствует условие задачи",
        "sub_title": "Работа",
        "sub_author": "Автор",
        "sub_problem": "Задача",
        "sub_submitted": "Отправлено",
        "sub_image": "Рукописное решение",
        "sub_marking": "Проверка",
        "pending_hint": "Идёт проверка — страница обновится автоматически.",
        "mark_score": "Балл",
        "mark_confidence": "Уверенность чтения:",
        "mark_steps": "Верных шагов:",
        "mark_final": "Итоговый ответ:",
        "mark_final_ok": "✓ верный",
        "mark_final_bad": "✗ неверный",
        "mark_strengths": "Сильные стороны",
        "mark_errors": "Найденные ошибки",
        "mark_suggestions": "Рекомендации",
        "mark_model": "Модель:",
        "mark_failed": "Проверка не удалась. Ошибка:",
        "problems_title": "Каталог задач",
        "problems_sub": "Все задачи, отправленные ранее, с числом попыток и средним баллом.",
        "problem_id": "№",
        "problem_title_col": "Название",
        "problem_category": "Раздел",
        "problem_difficulty": "Сложность",
        "problem_subs": "Работ",
        "problem_avg": "Ср.",
        "problem_created": "Создано",
        "problem_statement_h": "Условие задачи",
        "problem_subs_h": "Отправленные решения",
        "author_stats_c": "Проверено решений:",
        "author_stats_avg": "Средний балл:",
        "author_history_h": "История решений",
        "verdict_correct": "правильно",
        "verdict_partial": "частично правильно",
        "verdict_wrong": "неверно",
        "verdict_illegible": "нечитаемо",
        "nav_submissions": "Все",
        "view_all_subs": "Все работы →",
        "subs_title": "Все работы",
        "subs_sub": "Все отправленные работы. Нажмите на строку, чтобы открыть полный разбор от ИИ.",
        "subs_search_ph": "Поиск по названию, условию или автору…",
        "subs_filter_author": "Автор",
        "subs_filter_verdict": "Вердикт",
        "subs_filter_status": "Статус",
        "subs_filter_lang": "Язык проверки",
        "subs_filter_all": "Все",
        "subs_reset": "Сбросить фильтры",
        "subs_apply": "Применить",
        "subs_page_of": "Стр. {page} из {pages}",
        "subs_total": "{n} работ",
        "subs_prev": "← Назад",
        "subs_next": "Вперёд →",
        "st_pending": "в ожидании",
        "st_done": "готово",
        "st_failed": "ошибка",
        "regrade_btn": "Проверить снова",
        "regrade_confirm": "Запустить проверку заново?",
        "regrade_running": "Проверка…",
    },
}

def current_lang():
    q = (request.args.get("lang") or "").lower()
    if q in UI_LANGS:
        return q
    c = (request.cookies.get("lang") or "").lower()
    if c in UI_LANGS:
        return c
    return UI_DEFAULT

_VERDICT_ALIAS = {
    # Italian legacy
    "corretta": "correct", "parzialmente": "partial", "errata": "wrong", "illeggibile": "illegible",
    # English
    "correct": "correct", "partial": "partial", "partially": "partial", "wrong": "wrong",
    "incorrect": "wrong", "illegible": "illegible",
    # Russian
    "правильно": "correct", "верно": "correct", "частично": "partial",
    "неверно": "wrong", "неправильно": "wrong", "нечитаемо": "illegible",
}
def verdict_class(v):
    if not v: return ""
    first = v.strip().split()[0].lower()
    return _VERDICT_ALIAS.get(first, "")

@app.context_processor
def inject_i18n():
    lang = current_lang()
    return {"lang": lang, "t": TEXT[lang], "UI_LANGS": UI_LANGS, "verdict_class": verdict_class}

@app.route("/lang/<code>")
def set_lang(code):
    code = code.lower()
    if code not in UI_LANGS:
        abort(404)
    nxt = request.args.get("next") or url_for("home")
    resp = redirect(nxt)
    resp.set_cookie("lang", code, max_age=60*60*24*365, samesite="Lax")
    return resp

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
                SELECT s.image_data, s.image_mimetype, s.language, p.statement
                FROM submissions s JOIN problems p ON p.id = s.problem_id
                WHERE s.id = %s
            """, (submission_id,))
            row = cur.fetchone()
            if not row: return
            image = bytes(row["image_data"])
            mimetype = row["image_mimetype"]
            problem = row["statement"]
            lang = row["language"] or "en"

        data, latency_ms, raw, cost = mark_solution(problem, image, mimetype, lang)

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
    ui_lang = current_lang()
    tt = TEXT[ui_lang]
    if request.method == "POST":
        author = request.form.get("author_name","").strip()
        problem_id = request.form.get("problem_id","").strip()
        problem_title = request.form.get("problem_title","").strip()
        problem_statement = request.form.get("problem_statement","").strip()
        problem_category = request.form.get("problem_category","").strip() or None
        problem_difficulty = request.form.get("problem_difficulty","").strip() or None
        lang = (request.form.get("language","").strip().lower() or ui_lang)
        if lang not in SUPPORTED_LANGS: lang = ui_lang if ui_lang in SUPPORTED_LANGS else "en"
        image = request.files.get("image")

        if not author: return tt["err_author_missing"], 400
        if not image or not image.filename: return tt["err_image_missing"], 400

        image_bytes = image.read()
        if len(image_bytes) > 20*1024*1024: return tt["err_image_too_big"], 400
        image_bytes, mimetype = normalize_image(image_bytes, image.mimetype)

        with conn.cursor() as cur:
            if problem_id:
                cur.execute("SELECT id FROM problems WHERE id=%s", (int(problem_id),))
                if not cur.fetchone(): return tt["err_problem_not_found"], 400
                pid = int(problem_id)
            else:
                if not problem_statement: return tt["err_statement_missing"], 400
                cur.execute("""INSERT INTO problems (title, statement, category, difficulty)
                               VALUES (%s,%s,%s,%s) RETURNING id""",
                            (problem_title or problem_statement[:80],
                             problem_statement, problem_category, problem_difficulty))
                pid = cur.fetchone()["id"]

            cur.execute("""INSERT INTO submissions (problem_id, author_name, image_data, image_mimetype, image_size, language)
                           VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (pid, author, image_bytes, mimetype, len(image_bytes), lang))
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

@app.route("/submissions")
def submissions_index():
    conn = get_conn()
    q = request.args.get("q","").strip()
    author = request.args.get("author","").strip()
    verdict = request.args.get("verdict","").strip()  # canonical: correct|partial|wrong|illegible
    status = request.args.get("status","").strip()    # pending|done|failed
    lang_f = request.args.get("submission_lang","").strip()
    page = max(1, int(request.args.get("page", 1) or 1))
    per_page = 25
    where = ["1=1"]
    params = []
    if q:
        where.append("(LOWER(p.title) LIKE %s OR LOWER(p.statement) LIKE %s OR LOWER(s.author_name) LIKE %s)")
        needle = f"%{q.lower()}%"
        params += [needle, needle, needle]
    if author:
        where.append("s.author_name = %s")
        params.append(author)
    if verdict:
        where.append("""
          CASE
            WHEN m.feedback_json ? 'verdict_code' THEN m.feedback_json->>'verdict_code'
            WHEN LOWER(SPLIT_PART(COALESCE(m.verdict,''),' ',1)) IN ('corretta','correct','правильно','верно') THEN 'correct'
            WHEN LOWER(SPLIT_PART(COALESCE(m.verdict,''),' ',1)) IN ('parzialmente','partial','partially','частично') THEN 'partial'
            WHEN LOWER(SPLIT_PART(COALESCE(m.verdict,''),' ',1)) IN ('errata','wrong','incorrect','неверно','неправильно') THEN 'wrong'
            WHEN LOWER(SPLIT_PART(COALESCE(m.verdict,''),' ',1)) IN ('illeggibile','illegible','нечитаемо') THEN 'illegible'
            ELSE ''
          END = %s
        """)
        params.append(verdict)
    if status:
        where.append("COALESCE(m.status,'pending') = %s")
        params.append(status)
    if lang_f:
        where.append("COALESCE(s.language,'en') = %s")
        params.append(lang_f)
    where_sql = " AND ".join(where)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT COUNT(*) c FROM submissions s
            JOIN problems p ON p.id=s.problem_id
            LEFT JOIN markings m ON m.submission_id=s.id
            WHERE {where_sql}
        """, params)
        total = cur.fetchone()["c"]
        cur.execute(f"""
            SELECT s.id, s.author_name, s.created_at, s.language,
                   p.title AS problem_title,
                   m.status, m.score, m.verdict, m.feedback_json
            FROM submissions s
            JOIN problems p ON p.id=s.problem_id
            LEFT JOIN markings m ON m.submission_id=s.id
            WHERE {where_sql}
            ORDER BY s.id DESC
            LIMIT %s OFFSET %s
        """, params + [per_page, (page-1)*per_page])
        subs = cur.fetchall()
        cur.execute("SELECT DISTINCT author_name FROM submissions ORDER BY author_name")
        authors = [r["author_name"] for r in cur.fetchall()]
    pages = max(1, (total + per_page - 1) // per_page)
    return render_template("submissions.html",
        subs=subs, authors=authors,
        q=q, author=author, verdict=verdict, status=status, submission_lang=lang_f,
        page=page, pages=pages, total=total)

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

@app.route("/submission/<int:sid>/regrade", methods=["POST"])
def submission_regrade(sid):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM submissions WHERE id=%s", (sid,))
        if not cur.fetchone(): abort(404)
        cur.execute("SELECT status FROM markings WHERE submission_id=%s", (sid,))
        m = cur.fetchone()
        if m and m["status"] == "pending":
            return jsonify({"ok": False, "reason": "already pending"}), 409
        cur.execute("""INSERT INTO markings (submission_id, model, status)
                       VALUES (%s,%s,'pending')
                       ON CONFLICT (submission_id) DO UPDATE
                       SET status='pending', error=NULL, score=NULL, verdict=NULL,
                           feedback_json=NULL, raw_response=NULL, latency_ms=NULL,
                           cost_usd=NULL, updated_at=NOW(), model=EXCLUDED.model""",
                    (sid, MODEL))
        conn.commit()
    kick_worker(sid)
    return jsonify({"ok": True})

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
