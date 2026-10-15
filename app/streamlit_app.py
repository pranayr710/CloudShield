"""CloudShield: live predictions from the three final models (Streamlit).

    streamlit run app/streamlit_app.py

Open a task directly with ?track=classify, ?track=cluster or ?track=regress (the project website
embeds this page that way).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
st.set_page_config(page_title="CloudShield · Live models", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the trained models…")
def models():
    import predict
    return predict


M = models()

TRACKS = {
    "classify": ("Classify a flow", "TON_IoT · Bagging (tuned)"),
    "cluster": ("Find its cluster", "TON_IoT · K-Means"),
    "regress": ("Forecast reply size", "UNSW-NB15 · Random Forest"),
}
PROTOS = ["tcp", "udp", "icmp"]
SERVICES = ["-", "dns", "http", "ftp", "ssl", "gssapi", "dce_rpc", "smb", "smb;gssapi"]  # every service in TON_IoT
STATES = ["S0", "SF", "REJ", "OTH", "SH", "S1", "S3", "SHR", "RSTR", "RSTRH", "RSTO", "S2", "RSTOS0"]  # every state in TON_IoT
U_PROTOS = ["tcp", "udp", "arp", "unas", "ospf", "sctp"]
U_SERVICES = ["-", "dns", "http", "ftp", "ftp-data", "smtp", "ssh", "pop3", "snmp", "ssl", "dhcp", "irc", "radius"]
U_STATES = ["FIN", "INT", "CON", "REQ", "RST", "ACC", "CLO"]

# one real flow per traffic type from the held-out test machines, which the classifier gets right
TON_PRESETS = {f"{M.name(c)} (real test flow)": tuple(p["values"]) + (p["dst_internal"],) for c, p in M.CLS_PRESETS.items()}
TON_KEYS = ["proto", "service", "conn_state", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
            "src_port", "dst_internal"]
FIRST = next((k for k in TON_PRESETS if k.startswith("Normal")), next(iter(TON_PRESETS)))
REG_PRESETS = {
    "Small exchange": ("tcp", "-", "FIN", 0.45, 10, 6, 738, 254, 252, 0),
    "Busy download": ("tcp", "-", "FIN", 1.85, 28, 72, 1320, 62, 252, 2),
    "Large transfer": ("tcp", "-", "FIN", 4.2, 300, 800, 16000, 31, 29, 14),
    "One-way UDP": ("udp", "dns", "INT", 0.0, 2, 0, 114, 254, 0, 0),
}
REG_KEYS = ["r_proto", "r_service", "r_state", "dur", "spkts", "dpkts", "sbytes", "sttl", "dttl", "dloss"]

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; max-width: 1200px; }
      h1, h2, h3 { font-family: "Arial Black", Inter, sans-serif; letter-spacing: -0.02em; }
      .cs-card { border-radius: 14px; padding: 18px 20px; background: #15130f; color: #efeae0; }
      .cs-eyebrow { font: 600 0.72rem ui-monospace, Consolas, monospace; letter-spacing: 0.18em; text-transform: uppercase; opacity: 0.7; }
      .cs-big { font: 900 2.4rem "Arial Black", Inter, sans-serif; line-height: 1.05; margin: 4px 0; text-transform: uppercase; }
      .cs-note { opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply(keys, values):
    for k, v in zip(keys, values):
        st.session_state[k] = v


def card(eyebrow: str, big: str, note: str, color: str) -> None:
    st.markdown(
        f'<div class="cs-card"><div class="cs-eyebrow">{eyebrow}</div>'
        f'<div class="cs-big" style="color:{color}">{big}</div><div class="cs-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- header and task picker
default = st.query_params.get("track", "classify")
default = default if default in TRACKS else "classify"
embedded = st.query_params.get("embed") == "true"  # inside the project website, which shows its own title and task cards
if not embedded:
    st.title("CloudShield · live models")
st.caption(
    f"Three trained models from the CloudShield project. Classification and clustering use TON_IoT network flows; "
    f"regression uses UNSW-NB15. The classifier was tested on machines it never saw in training "
    f"(accuracy {M.CLS_SCORES['acc']:.2%}, weighted F1 {M.CLS_SCORES['f1w']:.3f})."
)
if not embedded:
    import results_pages as RP
    page = st.sidebar.radio("Pages", RP.PAGES)
    st.sidebar.caption("23CSE301 Machine Learning · CloudShield")
    if page != "Live predictions":
        RP.render(page)
        st.stop()

if embedded:
    track = default
else:
    track = st.segmented_control("Task", list(TRACKS), default=default, format_func=lambda k: TRACKS[k][0],
                                 label_visibility="collapsed") or default
    st.caption(TRACKS[track][1])

left, right = st.columns([6, 5], gap="large")

# ---------------------------------------------------------------- inputs
with left:
    if track in ("classify", "cluster"):
        if "proto" not in st.session_state:
            apply(TON_KEYS, TON_PRESETS[FIRST])
        preset = st.selectbox("Start from an example", ["–"] + list(TON_PRESETS))
        if preset != "–" and st.session_state.get("_ton_preset") != preset:
            apply(TON_KEYS, TON_PRESETS[preset])
            st.session_state["_ton_preset"] = preset
        with st.form("ton"):
            a, b, c = st.columns(3)
            a.selectbox("Protocol", PROTOS, key="proto")
            b.selectbox("Service ('-' for none)", SERVICES, key="service")
            c.selectbox("Connection state", STATES, key="conn_state")
            a, b = st.columns(2)
            a.number_input("Destination port", min_value=0, max_value=65535, step=1, key="dst_port")
            b.number_input("Duration (s)", min_value=0.0, step=0.001, format="%.3f", key="duration")
            a, b = st.columns(2)
            a.number_input("Bytes sent", min_value=0, step=1, key="src_bytes")
            b.number_input("Bytes received", min_value=0, step=1, key="dst_bytes")
            a, b = st.columns(2)
            a.number_input("Packets sent", min_value=0, step=1, key="src_pkts")
            b.number_input("Packets received", min_value=0, step=1, key="dst_pkts")
            a, b = st.columns(2)
            a.number_input("Source port", min_value=0, max_value=65535, step=1, key="src_port")
            b.checkbox("Destination inside the network (192.168.1.x)", key="dst_internal")
            st.caption("Other TON_IoT columns (DNS, SSL and HTTP details) are set to their 'not applicable' values. "
                       "The clustering model does not use the source port or the destination flag.")
            go = st.form_submit_button(TRACKS[track][0], type="primary")
    else:
        if "r_proto" not in st.session_state:
            apply(REG_KEYS, REG_PRESETS["Busy download"])
        preset = st.selectbox("Start from an example", ["–"] + list(REG_PRESETS))
        if preset != "–" and st.session_state.get("_reg_preset") != preset:
            apply(REG_KEYS, REG_PRESETS[preset])
            st.session_state["_reg_preset"] = preset
        with st.form("reg"):
            a, b, c = st.columns(3)
            a.selectbox("Protocol", U_PROTOS, key="r_proto")
            b.selectbox("Service", U_SERVICES, key="r_service")
            c.selectbox("State", U_STATES, key="r_state")
            a, b = st.columns(2)
            a.number_input("Duration (s)", min_value=0.0, step=0.01, format="%.2f", key="dur")
            b.number_input("Bytes sent", min_value=0, step=1, key="sbytes")
            a, b, c = st.columns(3)
            a.number_input("Packets sent", min_value=0, step=1, key="spkts")
            b.number_input("Packets received", min_value=0, step=1, key="dpkts")
            c.number_input("Packets lost back", min_value=0, step=1, key="dloss")
            a, b = st.columns(2)
            a.number_input("Source TTL", min_value=0, max_value=255, step=1, key="sttl")
            b.number_input("Destination TTL", min_value=0, max_value=255, step=1, key="dttl")
            st.caption("Other UNSW-NB15 columns take their typical values from the training data.")
            go = st.form_submit_button(TRACKS[track][0], type="primary")

# ---------------------------------------------------------------- results
with right:
    s = st.session_state
    if track == "classify":
        out = M.classify(*(s[k] for k in TON_KEYS)) if go or "_cls" not in s else s["_cls"]
        s["_cls"] = out
        color = "#5fd39a" if out["prediction"] == "normal" else "#ff7a50"
        card("Verdict" if out["prediction"] == "normal" else "Attack detected", out["label"],
             f"{out['confidence']:.1%} confidence. {out['note']}", color)
        st.dataframe(
            pd.DataFrame({"Class": [M.name(k) for k in out["probabilities"]], "Probability": list(out["probabilities"].values())}),
            hide_index=True, use_container_width=True,
            column_config={"Probability": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0)},
        )
    elif track == "cluster":
        out = M.cluster(*(s[k] for k in TON_KEYS)) if go or "_clu" not in s else s["_clu"]
        s["_clu"] = out
        card("Behaviour group", f"Cluster {out['cluster']} of {out['k']}",
             f"{out['size']:,} training flows fell in this cluster.", "#ffb547")
        st.markdown("**What that cluster held during training**")
        st.dataframe(
            pd.DataFrame({"Traffic type": [M.name(k) for k in out["mix"]], "Share": list(out["mix"].values())}),
            hide_index=True, use_container_width=True,
            column_config={"Share": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0)},
        )
        st.markdown("**Distance to each cluster centre** (shorter is closer)")
        st.bar_chart(pd.DataFrame({"distance": out["distances"]}, index=[f"Cluster {i + 1}" for i in range(out["k"])]),
                     horizontal=True, height=220)
    else:
        out = M.regress(*(s[k] for k in REG_KEYS)) if go or "_reg" not in s else s["_reg"]
        s["_reg"] = out
        card("Predicted reply size", f"{out['bytes']:,.0f} bytes",
             f"The trees agree on {out['low']:,.0f} to {out['high']:,.0f} bytes (10th to 90th percentile).", "#5fd39a")
        st.caption("Packets received is by far the strongest signal; try changing it.")
