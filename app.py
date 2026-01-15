import os
import io
import random
import yaml
import json
from typing import Dict, Any, Optional, Tuple

import streamlit as st

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Data / viz libs
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px, go = None, None

st.set_page_config(page_title="AgentQuest — GUDID & MedDist Chronicles", layout="wide")

BASE_DIR = os.path.dirname(__file__)

# ---------- Utility: file helpers ----------

def load_agents() -> Dict[str, Any]:
    path = os.path.join(BASE_DIR, "agents.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def read_text_file(relpath: str) -> str:
    path = os.path.join(BASE_DIR, relpath)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def write_text_file(relpath: str, content: str) -> None:
    path = os.path.join(BASE_DIR, relpath)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_i18n() -> Dict[str, Dict[str, str]]:
    path = os.path.join(BASE_DIR, "i18n.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "en": {
                "theme": "Theme",
                "style_picker": "Visual Style",
                "choose_style": "Choose painter style",
                "jackpot": "Spin Style Wheel",
                "jackpot_spun": "Wheel result",
                "subtitle": "Multi-agent workbench for GUDID & medical device distribution intelligence",
                "missions": "Missions",
                "select_mission": "Select a mission",
                "open_briefing": "Open Mission Briefing",
                "workbench": "Agent Workbench",
                "intel_prompt": "Working text / data",
                "chain_agents": "Chain agents (legacy pipeline)",
                "running_pipeline": "Running pipeline...",
                "pipeline_complete": "Pipeline finished.",
                "selected_style": "Selected painter style",
            },
            "zh-TW": {
                "theme": "主題",
                "style_picker": "視覺風格",
                "choose_style": "選擇畫家風格",
                "jackpot": "風格轉盤",
                "jackpot_spun": "轉盤結果",
                "subtitle": "多代理工作台：GUDID 與醫療器材流通智慧分析",
                "missions": "任務",
                "select_mission": "選擇任務",
                "open_briefing": "開啟任務簡報",
                "workbench": "代理工作台",
                "intel_prompt": "工作文字 / 資料",
                "chain_agents": "串接代理（舊版管線）",
                "running_pipeline": "正在執行管線...",
                "pipeline_complete": "管線完成。",
                "selected_style": "目前畫家風格",
            },
        }

AGENTS = load_agents()
I18N = load_i18n()

def t(key: str, lang: str = "en") -> str:
    return I18N.get(lang, {}).get(key, key)

# ---------- LLM routing ----------

def get_api_keys() -> Dict[str, Optional[str]]:
    return {
        "openai": os.environ.get("OPENAI_API_KEY") or st.session_state.get("openai_key"),
        "gemini": os.environ.get("GEMINI_API_KEY") or st.session_state.get("gemini_key"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_key"),
        "grok": os.environ.get("GROK_API_KEY") or st.session_state.get("grok_key"),
    }

def call_llm(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Unified LLM caller.
    Supports OpenAI (gpt-*), Gemini (gemini-*), Anthropic (claude/anthropic-*), Grok (grok-*).
    Falls back to Gemini / OpenAI if provider not available.
    """
    if max_tokens is None:
        max_tokens = st.session_state.get("max_tokens", 12000)

    if model is None:
        model = st.session_state.get("llm_model", "gpt-4o-mini")

    keys = get_api_keys()
    last_provider = "unknown"
    success = False
    response_text = ""

    # ---- Gemini ----
    if model.startswith("gemini") and keys["gemini"]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=keys["gemini"])
            gen_model = model
            # for some HF setups, model names are slightly different – leave as-is
            resp = genai.generate_text(model=gen_model, prompt=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "text") and resp.text:
                response_text = resp.text
            elif getattr(resp, "candidates", None):
                response_text = resp.candidates[0].content
            else:
                response_text = str(resp)
            last_provider = "gemini"
            success = True
        except Exception as e:
            response_text = f"[Gemini error] {e}"

    # ---- OpenAI (gpt-*) ----
    if (not success) and model.startswith("gpt") and keys["openai"]:
        try:
            import openai
            openai.api_key = keys["openai"]
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            if resp and hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if isinstance(choice, dict):
                    msg = choice.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        response_text = msg["content"]
                else:
                    msg = getattr(choice, "message", None)
                    if msg and hasattr(msg, "content"):
                        response_text = msg.content
            if not response_text:
                response_text = str(resp)
            last_provider = "openai"
            success = True
        except Exception as e:
            response_text = f"[OpenAI error] {e}"

    # ---- Anthropic (claude / anthropic- prefix) ----
    if (not success) and (model.startswith("claude") or model.startswith("anthropic")) and keys["anthropic"]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=keys["anthropic"])
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # anthropic response: resp.content is a list of content blocks
            if getattr(resp, "content", None):
                blocks = resp.content
                texts = []
                for b in blocks:
                    # text blocks usually have type="text" and attribute "text"
                    if hasattr(b, "text"):
                        texts.append(b.text)
                    elif isinstance(b, dict) and "text" in b:
                        texts.append(b["text"])
                response_text = "\n".join(texts) if texts else str(resp)
            else:
                response_text = str(resp)
            last_provider = "anthropic"
            success = True
        except Exception as e:
            response_text = f"[Anthropic error] {e}"

    # ---- Grok (xAI) ----
    if (not success) and model.startswith("grok") and keys["grok"]:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {keys['grok']}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            r = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if data.get("choices"):
                msg = data["choices"][0]["message"]["content"]
                response_text = msg
            else:
                response_text = str(data)
            last_provider = "grok"
            success = True
        except Exception as e:
            response_text = f"[Grok error] {e}"

    # ---- Fallback: Gemini then OpenAI default models ----
    if not success and keys["gemini"]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=keys["gemini"])
            resp = genai.generate_text(model="gemini-2.5-flash", prompt=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "text") and resp.text:
                response_text = resp.text
            else:
                response_text = str(resp)
            last_provider = "gemini-fallback"
            success = True
        except Exception as e:
            response_text = response_text or f"[Gemini fallback error] {e}"

    if not success and keys["openai"]:
        try:
            import openai
            openai.api_key = keys["openai"]
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            if resp and hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if isinstance(choice, dict):
                    msg = choice.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        response_text = msg["content"]
                else:
                    msg = getattr(choice, "message", None)
                    if msg and hasattr(msg, "content"):
                        response_text = msg.content
            if not response_text:
                response_text = str(resp)
            last_provider = "openai-fallback"
            success = True
        except Exception as e:
            response_text = response_text or f"[OpenAI fallback error] {e}"

    if not success and not response_text:
        response_text = "[mock response] " + prompt[:400]

    # Update WOW status in session
    st.session_state["last_llm_provider"] = last_provider
    st.session_state["last_llm_model"] = model
    st.session_state["last_llm_success"] = success
    st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1

    return response_text

def execute_agent(
    agent_cfg: Dict[str, Any],
    input_text: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    user_prompt_override: Optional[str] = None,
) -> str:
    """Render prompt_template and call LLM."""
    template = agent_cfg.get("prompt_template", "{{input}}")
    prompt = template.replace("{{input}}", input_text)

    # Optional override: {{user_prompt}} or appended block
    if user_prompt_override:
        if "{{user_prompt}}" in prompt:
            prompt = prompt.replace("{{user_prompt}}", user_prompt_override)
        else:
            prompt = f"{user_prompt_override}\n\n---\n\n{prompt}"

    return call_llm(prompt, model=model, max_tokens=max_tokens)

# ---------- Painter styles ----------

DEFAULT_PAINTER_STYLES = [
    "梵谷星夜風 (Van Gogh – Starry Night)",
    "莫內印象日出風 (Monet – Impression, Sunrise)",
    "畢卡索立體派風 (Picasso – Cubism)",
    "達文西寫實光影風 (Da Vinci – Chiaroscuro)",
    "高更塔希提色塊風 (Gauguin – Tahiti Colors)",
    "克林姆金箔裝飾風 (Klimt – Golden Deco)",
    "康丁斯基抽象幾何風 (Kandinsky – Abstract Geometry)",
    "馬蒂斯野獸派鮮彩風 (Matisse – Fauvism)",
    "安迪沃荷普普藝術風 (Warhol – Pop Art)",
    "霍克尼數位線條風 (Hockney – Digital Lines)",
    "透納浪漫雲海風 (Turner – Romantic Clouds)",
    "修拉點描派風 (Seurat – Pointillism)",
    "米羅夢境符號風 (Miró – Dream Symbols)",
    "達利超現實融化風 (Dalí – Surreal Melting)",
    "羅斯科色域冥想風 (Rothko – Color Field)",
    "梵谷向日葵暖色風 (Van Gogh – Sunflowers)",
    "馬格利特超現實日常風 (Magritte – Surreal Everyday)",
    "懷斯寫實詩意風 (Wyeth – Poetic Realism)",
    "葛飾北齋浮世繪風 (Hokusai – Ukiyo-e)",
    "齊白石寫意水墨風 (Qi Baishi – Ink Expressionism)",
]

# ---------- Medical device packing list parser & analysis ----------

def load_packing_list(
    upload_file,
    pasted_text: str,
) -> Optional[pd.DataFrame]:
    if pd is None:
        st.error("pandas 未安裝，無法進行資料表解析。請在 requirements 中加入 pandas。")
        return None

    if upload_file is not None:
        name = upload_file.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(upload_file)
            elif name.endswith(".json"):
                df = pd.read_json(upload_file)
            else:
                # try CSV by default
                df = pd.read_csv(upload_file)
            return df
        except Exception as e:
            st.error(f"上傳檔案解析失敗：{e}")
            return None

    pasted_text = (pasted_text or "").strip()
    if not pasted_text:
        return None

    # Try CSV from pasted text
    try:
        df = pd.read_csv(io.StringIO(pasted_text))
        return df
    except Exception:
        pass

    # Try JSON from pasted text
    try:
        df = pd.read_json(io.StringIO(pasted_text))
        return df
    except Exception:
        pass

    st.error("無法從貼上的內容解析出 CSV 或 JSON 格式的裝箱單。")
    return None

def build_packing_summary_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Try to cast Numbers to numeric
    if "Numbers" in df.columns:
        df["Numbers"] = pd.to_numeric(df["Numbers"], errors="coerce").fillna(0)
    else:
        df["Numbers"] = 1

    # Table 1: By Supplier + DeviceCategory
    group_cols_1 = [c for c in ["Suppliername", "DeviceCategory"] if c in df.columns]
    if group_cols_1:
        t1 = (
            df.groupby(group_cols_1, dropna=False)
            .agg(
                total_units=("Numbers", "sum"),
                shipment_count=("Numbers", "size"),
            )
            .reset_index()
            .sort_values("total_units", ascending=False)
        )
    else:
        t1 = df.head(50)

    # Table 2: By Customer + DeviceName
    group_cols_2 = [c for c in ["customer", "DeviceName"] if c in df.columns]
    if group_cols_2:
        t2 = (
            df.groupby(group_cols_2, dropna=False)
            .agg(
                total_units=("Numbers", "sum"),
                distinct_models=("ModelNum", pd.Series.nunique) if "ModelNum" in df.columns else ("Numbers", "size"),
            )
            .reset_index()
            .sort_values("total_units", ascending=False)
        )
    else:
        t2 = df.head(50)

    # Table 3: By licenseID + DeviceName
    group_cols_3 = [c for c in ["licenseID", "DeviceName"] if c in df.columns]
    if group_cols_3:
        t3 = (
            df.groupby(group_cols_3, dropna=False)
            .agg(
                total_units=("Numbers", "sum"),
                unique_customers=("customer", pd.Series.nunique) if "customer" in df.columns else ("Numbers", "size"),
            )
            .reset_index()
            .sort_values("total_units", ascending=False)
        )
    else:
        t3 = df.head(50)

    return t1, t2, t3

def build_packing_figures(df: pd.DataFrame) -> Dict[str, Any]:
    figs: Dict[str, Any] = {}
    if px is None or go is None:
        return figs

    df = df.copy()
    if "Numbers" not in df.columns:
        df["Numbers"] = 1
    df["Numbers"] = pd.to_numeric(df["Numbers"], errors="coerce").fillna(0)

    # 1. Units by Supplier (distribution bar chart)
    if "Suppliername" in df.columns:
        t_sup = df.groupby("Suppliername", dropna=False)["Numbers"].sum().reset_index()
        figs["units_by_supplier"] = px.bar(
            t_sup,
            x="Suppliername",
            y="Numbers",
            title="各供應商出貨數量 (Units by Supplier)",
        )

    # 2. Units by Customer
    if "customer" in df.columns:
        t_cus = df.groupby("customer", dropna=False)["Numbers"].sum().reset_index()
        figs["units_by_customer"] = px.bar(
            t_cus,
            x="customer",
            y="Numbers",
            title="各醫院 / 客戶收貨數量 (Units by Customer)",
        )

    # 3. Category share (pie)
    if "DeviceCategory" in df.columns:
        t_cat = df.groupby("DeviceCategory", dropna=False)["Numbers"].sum().reset_index()
        figs["category_share"] = px.pie(
            t_cat,
            names="DeviceCategory",
            values="Numbers",
            title="器材分類佔比 (Device Category Share)",
        )

    # 4. License vs volume (bar)
    if "licenseID" in df.columns:
        t_lic = df.groupby("licenseID", dropna=False)["Numbers"].sum().reset_index()
        figs["license_volume"] = px.bar(
            t_lic,
            x="licenseID",
            y="Numbers",
            title="許可證字號 vs 出貨數量 (License vs Volume)",
        )

    # 5. Supply chain flow (Sankey: Supplier -> Customer)
    if "Suppliername" in df.columns and "customer" in df.columns:
        flow = df.groupby(["Suppliername", "customer"], dropna=False)["Numbers"].sum().reset_index()
        suppliers = list(flow["Suppliername"].astype(str).unique())
        customers = list(flow["customer"].astype(str).unique())
        nodes = suppliers + customers
        node_index = {name: i for i, name in enumerate(nodes)}

        sources = flow["Suppliername"].astype(str).map(node_index).tolist()
        targets = flow["customer"].astype(str).map(node_index).tolist()
        values = flow["Numbers"].tolist()

        sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=nodes,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                    ),
                )
            ]
        )
        sankey.update_layout(title_text="醫療器材供應鏈流向圖 (Supplier → Customer Supply Chain Flow)", font_size=10)
        figs["supply_chain_flow"] = sankey

    return figs

# ---------- Main UI ----------

def sidebar_controls():
    st.sidebar.title("AgentQuest — Settings")

    # Language & theme
    lang = st.sidebar.selectbox(
        "Language / 語言",
        options=["en", "zh-TW"],
        format_func=lambda x: "English" if x == "en" else "繁體中文",
        key="lang",
    )

    theme = st.sidebar.selectbox(
        t("theme", lang),
        ["light", "dark"],
        index=0,
        key="ui_theme",
    )

    # Painter styles
    styles = AGENTS.get("styles") or DEFAULT_PAINTER_STYLES
    st.sidebar.markdown("---")
    st.sidebar.subheader(t("style_picker", lang))
    selected_style = st.sidebar.selectbox(
        t("choose_style", lang),
        options=styles,
        index=0,
        key="painter_style",
    )
    if st.sidebar.button(t("jackpot", lang), key="style_jackpot"):
        selected_style = random.choice(styles)
        st.session_state["painter_style"] = selected_style
        st.sidebar.success(f"{t('jackpot_spun', lang)}: {selected_style}")

    # Model & tokens
    st.sidebar.markdown("---")
    st.sidebar.subheader("LLM Model & Tokens")

    model_options = [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "grok-4-fast-reasoning",
        "grok-3-mini",
    ]
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = model_options[0]
    st.sidebar.selectbox(
        "LLM Model",
        options=model_options,
        index=model_options.index(st.session_state["llm_model"]),
        key="llm_model",
    )

    max_tokens = st.sidebar.number_input(
        "max_tokens (default 12000)",
        min_value=256,
        max_value=128000,
        value=st.session_state.get("max_tokens", 12000),
        step=256,
        key="max_tokens",
    )

    # API keys
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Keys")

    env_openai = os.environ.get("OPENAI_API_KEY")
    env_gemini = os.environ.get("GEMINI_API_KEY")
    env_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    env_grok = os.environ.get("GROK_API_KEY")

    if "openai_key" not in st.session_state:
        st.session_state["openai_key"] = None
    if "gemini_key" not in st.session_state:
        st.session_state["gemini_key"] = None
    if "anthropic_key" not in st.session_state:
        st.session_state["anthropic_key"] = None
    if "grok_key" not in st.session_state:
        st.session_state["grok_key"] = None

    if env_openai:
        st.sidebar.info("OpenAI API key 由環境變數提供，將直接使用。")
    else:
        st.session_state["openai_key"] = st.sidebar.text_input(
            "OpenAI API key（只在本次 Session 使用）",
            type="password",
            key="openai_key_input",
        )

    if env_gemini:
        st.sidebar.info("Gemini API key 由環境變數提供，將直接使用。")
    else:
        st.session_state["gemini_key"] = st.sidebar.text_input(
            "Gemini API key（只在本次 Session 使用）",
            type="password",
            key="gemini_key_input",
        )

    if env_anthropic:
        st.sidebar.info("Anthropic API key 由環境變數提供，將直接使用。")
    else:
        st.sidebar.text_input(
            "Anthropic API key（只在本次 Session 使用）",
            type="password",
            key="anthropic_key",
        )

    if env_grok:
        st.sidebar.info("Grok / xAI API key 由環境變數提供，將直接使用。")
    else:
        st.sidebar.text_input(
            "Grok / xAI API key（只在本次 Session 使用）",
            type="password",
            key="grok_key",
        )

    st.sidebar.markdown("---")
    st.sidebar.write(f"{t('selected_style', lang)}: {st.session_state.get('painter_style', selected_style)}")

    return lang, theme, selected_style

def wow_status_bar(lang: str):
    run_count = st.session_state.get("run_count", 0)
    last_provider = st.session_state.get("last_llm_provider", "—")
    last_model = st.session_state.get("last_llm_model", "—")
    success = st.session_state.get("last_llm_success", None)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Agent Runs / LLM Calls", run_count)
    with col2:
        st.metric("Last Provider", last_provider)
    with col3:
        st.metric("Last Model", last_model)
    with col4:
        status_label = "OK" if success else ("—" if success is None else "Error")
        st.metric("Last Call Status", status_label)

def mission_and_workbench_tab(lang: str):
    st.subheader(t("missions", lang))

    missions = AGENTS.get("sample_missions", [])
    mission = st.selectbox(
        t("select_mission", lang),
        options=missions,
        key="mission_select",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button(t("open_briefing", lang), key="btn_open_briefing"):
            agent_cfg = AGENTS.get("agents_map", {}).get("MissionBriefingGeneratorAgent")
            if not agent_cfg:
                st.error("MissionBriefingGeneratorAgent 不存在於 agents.yaml")
            else:
                briefing = execute_agent(agent_cfg, mission)
                st.text_area("Mission Briefing", value=briefing, height=300)

    with col2:
        st.subheader(t("workbench", lang))

        if "workbench_input" not in st.session_state:
            st.session_state["workbench_input"] = "Paste intel / data here (CSV, JSON, text)..."

        workbench_text = st.text_area(
            t("intel_prompt", lang),
            value=st.session_state["workbench_input"],
            height=200,
            key="workbench_text",
        )
        st.session_state["workbench_input"] = workbench_text

        agents_map = AGENTS.get("agents_map", {})
        agent_names = list(agents_map.keys())
        selected_agent = st.selectbox(
            "Select Agent (單一執行)",
            options=agent_names,
            key="single_agent_select",
        )

        user_extra = st.text_area("Optional extra instructions / 額外說明 (user_prompt)", height=100, key="single_agent_extra")

        display_mode = st.radio(
            "Output display mode / 輸出顯示方式",
            options=["text", "markdown"],
            index=0,
            horizontal=True,
            key="single_agent_display_mode",
        )

        if st.button("Run Agent / 執行代理", key="btn_run_single_agent"):
            cfg = agents_map.get(selected_agent)
            if not cfg:
                st.error("Agent config not found.")
            else:
                out = execute_agent(
                    cfg,
                    input_text=workbench_text,
                    model=st.session_state.get("llm_model"),
                    max_tokens=st.session_state.get("max_tokens", 12000),
                    user_prompt_override=user_extra.strip() or None,
                )
                st.session_state["last_agent_output"] = out

        edited_output = st.text_area(
            "Agent Output (editable, 可編輯後作為下一個代理輸入)",
            value=st.session_state.get("last_agent_output", ""),
            height=250,
            key="single_agent_output",
        )

        if st.button("Use above output as new workbench input / 以此輸出作為新輸入", key="btn_promote_output"):
            st.session_state["workbench_input"] = edited_output
            st.success("已將上述輸出設定為新的工作內容。請在上方編輯區查看。")

        # Legacy pipeline (keep original feature)
        st.markdown("---")
        st.caption("Legacy multi-agent pipeline (保留原始功能)")
        selected_agents = st.multiselect(
            t("chain_agents", lang),
            options=agent_names,
            default=agent_names[:2] if len(agent_names) >= 2 else agent_names,
            key="multi_agent_chain",
        )
        if st.button(t("run_analysis", lang), key="btn_run_pipeline"):
            st.info(t("running_pipeline", lang))
            current = workbench_text
            for name in selected_agents:
                cfg = agents_map.get(name)
                if not cfg:
                    st.warning(f"Agent {name} not found in agents.yaml")
                    continue
                st.write(f"-> {name}: {cfg.get('description', '')}")
                try:
                    current = execute_agent(
                        cfg,
                        current,
                        model=st.session_state.get("llm_model"),
                        max_tokens=st.session_state.get("max_tokens", 12000),
                    )
                except Exception as e:
                    current = f"[agent-error] {e}"
                st.text_area(f"Output — {name}", value=current, height=150)
            st.success(t("pipeline_complete", lang))

def medical_distribution_tab(lang: str):
    st.subheader("Medical Device Distribution Chain Analyzer / 醫療器材流通鏈分析")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 1. 上傳或貼上裝箱單 (CSV / JSON)")
        upload = st.file_uploader("Upload packing list", type=["csv", "json"], key="packing_upload")
        pasted = st.text_area("Or paste packing list CSV / JSON here", height=150, key="packing_paste")

        if st.button("Parse & Analyze / 解析並分析", key="btn_parse_packing"):
            df = load_packing_list(upload, pasted)
            st.session_state["packing_df"] = df
            if df is not None:
                st.success(f"已載入裝箱單，共 {len(df)} 筆紀錄。")

    df = st.session_state.get("packing_df", None)

    if df is not None:
        st.markdown("#### 2. 資料預覽")
        st.dataframe(df.head(50))

        st.markdown("#### 3. 分析總結表格 (3 tables)")
        t1, t2, t3 = build_packing_summary_tables(df)

        st.markdown("**Table 1 — 供應商 × 器材分類 (Supplier × DeviceCategory)**")
        st.dataframe(t1)

        st.markdown("**Table 2 — 客戶 × 器材名稱 (Customer × DeviceName)**")
        st.dataframe(t2)

        st.markdown("**Table 3 — 許可證 × 器材名稱 (LicenseID × DeviceName)**")
        st.dataframe(t3)

        st.markdown("#### 4. 視覺化圖表 (5 graphs, including distribution & supply chain)")
        figs = build_packing_figures(df)
        if not figs:
            st.info("Plotly 未安裝或沒有可用欄位，無法繪製圖表。請在 requirements 中加入 plotly。")
        else:
            if "units_by_supplier" in figs:
                st.plotly_chart(figs["units_by_supplier"], use_container_width=True)
            if "units_by_customer" in figs:
                st.plotly_chart(figs["units_by_customer"], use_container_width=True)
            if "category_share" in figs:
                st.plotly_chart(figs["category_share"], use_container_width=True)
            if "license_volume" in figs:
                st.plotly_chart(figs["license_volume"], use_container_width=True)
            if "supply_chain_flow" in figs:
                st.plotly_chart(figs["supply_chain_flow"], use_container_width=True)

        st.markdown("#### 5. LLM / Agents on Packing List (在裝箱單上執行 LLM / 代理)")

        # Prepare a compact textual representation for LLM
        numeric_preview = df.head(50)
        preview_md = numeric_preview.to_markdown(index=False)
        default_prompt = (
            "請根據以下醫療器材裝箱單資料，進行供應鏈與流通風險分析，並以要點條列方式輸出：\n\n"
            "```markdown\n" + preview_md + "\n```"
        )

        analysis_input = st.text_area(
            "Analysis prompt (editable)", value=default_prompt, height=200, key="packing_analysis_prompt"
        )

        agents_map = AGENTS.get("agents_map", {})
        agent_names = list(agents_map.keys())
        selected_agent = st.selectbox(
            "Select Agent from agents.yaml (optional，留空則直接呼叫 LLM)",
            options=["<Direct LLM>"] + agent_names,
            key="packing_agent_select",
        )

        display_mode = st.radio(
            "Output display mode / 顯示方式",
            options=["text", "markdown"],
            index=1,
            horizontal=True,
            key="packing_display_mode",
        )

        if st.button("Run on packing list / 在裝箱單上執行", key="btn_run_packing_agent"):
            if selected_agent == "<Direct LLM>":
                out = call_llm(
                    analysis_input,
                    model=st.session_state.get("llm_model"),
                    max_tokens=st.session_state.get("max_tokens", 12000),
                )
            else:
                cfg = agents_map.get(selected_agent)
                if not cfg:
                    st.error("Agent config not found.")
                    out = ""
                else:
                    out = execute_agent(
                        cfg,
                        input_text=analysis_input,
                        model=st.session_state.get("llm_model"),
                        max_tokens=st.session_state.get("max_tokens", 12000),
                    )

            st.session_state["packing_llm_output"] = out

        output_text = st.session_state.get("packing_llm_output", "")
        if display_mode == "markdown":
            st.markdown(output_text, unsafe_allow_html=True)
        else:
            st.text_area("Output", value=output_text, height=300, key="packing_llm_output_view")

def note_keeper_tab():
    st.subheader("AI Note Keeper")

    nk_col1, nk_col2 = st.columns([1, 1])

    with nk_col1:
        uploaded = st.file_uploader("Upload text or PDF (optional)", type=["pdf", "txt", "md"], key="note_uploader")
        note_text = st.text_area("Or paste note / markdown here", height=200, key="note_text")
        if uploaded is not None:
            if uploaded.type == "application/pdf" and PdfReader is not None:
                try:
                    reader = PdfReader(uploaded)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    note_text = "\n".join(pages)
                    st.session_state["note_text"] = note_text
                    st.success("Extracted text from PDF")
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")
            else:
                try:
                    text = uploaded.read().decode("utf-8", errors="ignore")
                    note_text = text
                    st.session_state["note_text"] = note_text
                    st.success("Loaded text file")
                except Exception as e:
                    st.error(f"Text file reading failed: {e}")

        if st.button("Organize Note into Markdown"):
            agent_cfg = AGENTS.get("agents_map", {}).get("NoteOrganizerAgent")
            input_text = st.session_state.get("note_text") or note_text
            if not input_text:
                st.warning("Please paste or upload some note text first.")
            elif not agent_cfg:
                st.error("NoteOrganizerAgent not found in agents.yaml")
            else:
                organized = execute_agent(
                    agent_cfg,
                    input_text,
                    model=st.session_state.get("llm_model"),
                    max_tokens=st.session_state.get("max_tokens", 12000),
                )
                st.session_state["organized_note"] = organized

    with nk_col2:
        organized_note = st.session_state.get("organized_note", "")
        st.subheader("Organized Note (editable)")
        edited = st.text_area("Edit markdown", value=organized_note, height=300, key="edited_note")

        # AI Magics
        st.subheader("AI Magics")
        magic = st.selectbox(
            "Choose an AI Magic",
            options=[
                "AI Wordgraph",
                "AI Keywords",
                "AI Summarize",
                "AI Expand",
                "AI ToneAdjust",
                "AI QnA",
            ],
            key="ai_magic",
        )
        kw_input = st.text_input("Keywords (comma-separated)", key="ai_magic_kw")
        kw_color = st.color_picker("Keyword color", value="#FF7F50", key="ai_magic_color")

        if st.button("Apply Magic", key="btn_apply_magic"):
            agent_name_map = {
                "AI Wordgraph": "AIWordgraphAgent",
                "AI Keywords": "AIKeywordsAgent",
                "AI Summarize": "AISummarizeAgent",
                "AI Expand": "AIExpandAgent",
                "AI ToneAdjust": "AIToneAdjustAgent",
                "AI QnA": "AIQnAAgent",
            }
            agent_name = agent_name_map[magic]
            agent_cfg = AGENTS.get("agents_map", {}).get(agent_name)
            if agent_cfg is None:
                st.error("Agent not found in manifest")
            else:
                payload = edited
                if magic == "AI Keywords":
                    payload = f"KEYWORDS: {kw_input}\nCOLOR: {kw_color}\n\n{edited}"
                out = execute_agent(
                    agent_cfg,
                    payload,
                    model=st.session_state.get("llm_model"),
                    max_tokens=st.session_state.get("max_tokens", 12000),
                )

                if magic == "AI Keywords" and kw_input:
                    kws = [k.strip() for k in kw_input.split(",") if k.strip()]
                    highlighted = out
                    for k in kws:
                        highlighted = highlighted.replace(
                            k, f"<span style='color:{kw_color}; font-weight:700'>{k}</span>"
                        )
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.text_area("Magic Output", value=out, height=300, key="magic_output")

def agents_and_skill_tab():
    st.subheader("Agents & SKILL Editor")

    ed_col1, ed_col2 = st.columns(2)

    # Agents.yaml editor
    with ed_col1:
        st.markdown("### agents.yaml")
        agents_raw = read_text_file("agents.yaml")
        uploaded_agents = st.file_uploader("Upload agents.yaml (replace)", type=["yaml", "yml"], key="upload_agents")
        if uploaded_agents is not None:
            try:
                data = uploaded_agents.read().decode("utf-8")
                st.session_state["agents_edit"] = data
                st.success("Uploaded agents.yaml into editor")
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
        agents_edit = st.text_area(
            "Edit agents.yaml",
            value=st.session_state.get("agents_edit", agents_raw),
            height=300,
            key="agents_edit_area",
        )
        if st.button("Save agents.yaml", key="btn_save_agents"):
            try:
                write_text_file("agents.yaml", agents_edit)
                st.success("Saved agents.yaml")
                global AGENTS
                AGENTS = load_agents()
            except Exception as e:
                st.error(f"Failed to save agents.yaml: {e}")
        st.download_button("Download agents.yaml", data=agents_edit, file_name="agents.yaml", mime="text/yaml")

    # SKILL.md editor
    with ed_col2:
        st.markdown("### SKILL.md")
        skill_raw = read_text_file("SKILL.md")
        uploaded_skill = st.file_uploader("Upload SKILL.md (replace)", type=["md"], key="upload_skill")
        if uploaded_skill is not None:
            try:
                data = uploaded_skill.read().decode("utf-8")
                st.session_state["skill_edit"] = data
                st.success("Uploaded SKILL.md into editor")
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
        skill_edit = st.text_area(
            "Edit SKILL.md",
            value=st.session_state.get("skill_edit", skill_raw),
            height=300,
            key="skill_edit_area",
        )
        if st.button("Save SKILL.md", key="btn_save_skill"):
            try:
                write_text_file("SKILL.md", skill_edit)
                st.success("Saved SKILL.md")
            except Exception as e:
                st.error(f"Failed to save SKILL.md: {e}")
        st.download_button("Download SKILL.md", data=skill_edit, file_name="SKILL.md", mime="text/markdown")

def main():
    lang, theme, painter_style = sidebar_controls()

    st.title("AgentQuest — GUDID & MedDist Chronicles")
    st.caption(t("subtitle", lang))

    wow_status_bar(lang)

    tabs = st.tabs(
        [
            "Mission Control & Agent Workbench",
            "Med Device Distribution Lab",
            "AI Note Keeper & Magics",
            "Agents & Skills",
        ]
    )

    with tabs[0]:
        mission_and_workbench_tab(lang)

    with tabs[1]:
        medical_distribution_tab(lang)

    with tabs[2]:
        note_keeper_tab()

    with tabs[3]:
        agents_and_skill_tab()


if __name__ == "__main__":
    main()
