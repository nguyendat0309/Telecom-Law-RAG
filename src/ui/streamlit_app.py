"""
Streamlit Chat UI — Giao diện chatbot gọi vào FastAPI backend.
"""

import json
import os
import streamlit as st
import requests
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Chatbot Luật Viễn thông",
    page_icon="🏛️",
    layout="centered",
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .main-header h1 { font-size: 1.8rem; margin: 0; }
    .main-header p { color: #666; font-size: 0.9rem; margin: 0.3rem 0 0; }
    .source-badge {
        display: inline-block;
        background: #f0f4ff;
        border: 1px solid #c0d0f0;
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        color: #3366aa;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏛️ Chatbot Luật Viễn thông</h1>
    <p>Hỏi đáp về Luật Viễn thông Việt Nam (Luật số 24/2023/QH15)</p>
</div>
""", unsafe_allow_html=True)


def check_api():
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=3)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None

with st.sidebar:
    st.markdown("### ⚙️ Hệ thống")
    health = check_api()
    if health:
        st.success("API: ✅ Online")
        st.info(f"🖥️ Device: {health.get('device', '?')}")
        st.info(f"🤖 Ollama: {health.get('ollama', '?')}")
        st.info(f"📦 VectorDB: {health.get('vectorstore', '?')}")
    else:
        st.error("❌ API offline! Bật API trước:")
        st.code("uvicorn src.api.main:app --port 8000", language="bash")

    st.divider()

    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💡 Câu hỏi mẫu")
    examples = [
        "Viễn thông là gì?",
        "Điều kiện cấp phép viễn thông?",
        "Quyền và nghĩa vụ của doanh nghiệp viễn thông?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            sources_html = " ".join(f'<span class="source-badge">📎 {s}</span>' for s in msg["sources"])
            st.markdown(sources_html, unsafe_allow_html=True)
        if msg.get("time"):
            st.caption(f"⏱️ {msg['time']:.1f}s")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
else:
    question = st.chat_input("Nhập câu hỏi về Luật Viễn thông...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        text_placeholder = st.empty()
        start = time.time()
        full_text = ""
        sources = []

        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]

            status_placeholder.markdown("🔍 *Đang xử lý...*")
            resp = requests.post(
                f"{API_URL}/api/chat/stream",
                json={"question": question, "history": history[-6:]},
                stream=True,
                timeout=300,
            )

            if resp.ok:
                first_token = True
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not decoded.startswith("data: "):
                        continue
                    data = decoded[6:]
                    if data == "[DONE]":
                        break
                    elif data.startswith("[SOURCES]"):
                        sources_raw = json.loads(data[9:])
                        sources = [s["source"] for s in sources_raw]
                    else:
                        if first_token:
                            status_placeholder.empty()
                            first_token = False
                        full_text += json.loads(data)
                        text_placeholder.markdown(full_text + "▌")

                api_time = round(time.time() - start, 1)
                text_placeholder.markdown(full_text)

                if sources:
                    sources_html = " ".join(f'<span class="source-badge">📎 {s}</span>' for s in sources)
                    st.markdown(sources_html, unsafe_allow_html=True)

                st.caption(f"⏱️ {api_time:.1f}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                    "sources": sources,
                    "time": api_time,
                })
            else:
                status_placeholder.empty()
                st.error(f"❌ API error: {resp.status_code}")

        except requests.exceptions.ConnectionError:
            status_placeholder.empty()
            st.error("❌ Không kết nối được API. Hãy bật API trước!")
            st.code("uvicorn src.api.main:app --port 8000")
        except requests.exceptions.Timeout:
            status_placeholder.empty()
            st.error("⏰ Timeout — LLM đang xử lý quá lâu.")
