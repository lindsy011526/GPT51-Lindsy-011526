# AgentQuest — GUDID & MedDist Chronicles SKILL Spec

This document describes the skills, agents, data flows, and model usage
for the **AgentQuest — GUDID & Medical Device Distribution Chronicles**
Streamlit app deployed on Hugging Face Spaces.

The system focuses on:

- Multi‑agent LLM workflows
- GUDID / medical device–style data exploration
- **Medical device distribution chain analysis** from packing lists
- Note organization and AI “magics”
- Editable `agents.yaml` + `SKILL.md` for rapid customization

---

## 1. System Overview / 系統總覽

- **Frontend**: Streamlit app (`app.py`)
- **Config**: 
  - `agents.yaml` — All agent prompt templates and painter styles.
  - `SKILL.md` — This skill / capability definition.
- **Backends (LLM providers)**:
  - OpenAI (e.g. `gpt-4o-mini`, `gpt-4.1-mini`)
  - Google Gemini (e.g. `gemini-2.5-flash`, `gemini-2.5-flash-lite`)
  - Anthropic (e.g. `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`)
  - Grok / xAI (e.g. `grok-4-fast-reasoning`, `grok-3-mini`)

- **Key UI Areas / 主要介面分區**:
  1. Mission Control & Agent Workbench
  2. Med Device Distribution Lab (裝箱單 / 配銷分析實驗室)
  3. AI Note Keeper & Magics
  4. Agents & Skills Editor

- **UI Enhancements (WOW UI)**:
  - Language: English / 繁體中文
  - Theme: light / dark
  - 20 painter‑style labels (Van Gogh, Monet, …) with **style wheel (jackpot)**.
  - WOW status bar: LLM call count, last provider, last model, last‑call status.

---

## 2. LLM Routing & API Keys

### 2.1 Supported Models

Global model dropdown supports (and is used by default):

- `gpt-4o-mini`
- `gpt-4.1-mini`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `claude-3-5-sonnet-20241022`
- `claude-3-haiku-20240307`
- `grok-4-fast-reasoning`
- `grok-3-mini`

Each **agent run** can override:

- Model (via the global dropdown)
- `max_tokens` (global numeric control, default 12,000)

### 2.2 API Key Handling / 金鑰處理

Environment variables (preferred, never shown in UI):

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROK_API_KEY`

If an env var is missing, the sidebar shows a **password input** for that provider:

- Values entered are stored in `st.session_state` only.
- Keys originating from environment variables are never displayed.

### 2.3 LLM Call Logic (High‑level)

`call_llm(prompt, model, max_tokens)`:

1. Route by model prefix:
   - `gemini-*` → Google Generative AI
   - `gpt-*` → OpenAI ChatCompletion
   - `claude-*` / `anthropic-*` → Anthropic Messages API
   - `grok-*` → Grok / xAI chat API
2. On failure, fall back to:
   - Gemini (`gemini-2.5-flash`)
   - Then OpenAI (`gpt-4o-mini`)
3. If still no provider available, returns `[mock response] ...`.

Session metadata:

- `last_llm_provider`
- `last_llm_model`
- `last_llm_success`
- `run_count` (total agent / LLM calls)

These feed the WOW status bar.

---

## 3. Agents & Prompt System / 代理與提示系統

### 3.1 Agent Definition (`agents.yaml`)

Each agent entry in `agents.yaml` uses:

```yaml
agents_map:
  AgentName:
    description: "Short human description"
    prompt_template: |
      Some instructions...
      {{input}}
