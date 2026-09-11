# MathMark

Automated math solution marking system. Upload a photo of a handwritten math solution, get an AI-generated score + detailed feedback.

Live: https://mathmark.onrender.com/ (once deployed)

## Stack
- Flask + PostgreSQL (Neon)
- Claude Opus 4.7 vision for marking
- Deployed on Render

## Env vars
- DATABASE_URL
- ANTHROPIC_API_KEY  
- MARK_MODEL (default: claude-opus-4-7)
