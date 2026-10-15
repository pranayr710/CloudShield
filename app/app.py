"""CloudShield: live predictions from the three final models (Gradio).

Tabs and API endpoints (the project website calls these through the Gradio client):
    /classify  TON_IoT flow -> traffic type (Bagging, tuned)
    /cluster   TON_IoT flow -> behaviour cluster (K-Means)
    /regress   UNSW-NB15 flow -> bytes the reply will carry (Random Forest)

Run locally:  python app.py   (from this folder)
"""
from __future__ import annotations

import gradio as gr

import predict as M
from predict import name

PROTOS = ["tcp", "udp", "icmp"]
SERVICES = ["-", "dns", "http", "ftp", "ssl", "gssapi", "dce_rpc", "smb", "smb;gssapi"]  # every service in TON_IoT
STATES = ["S0", "SF", "REJ", "OTH", "SH", "S1", "S3", "SHR", "RSTR", "RSTRH", "RSTO", "S2", "RSTOS0"]  # every state in TON_IoT
UNSW_PROTOS = ["tcp", "udp", "arp", "unas", "ospf", "sctp"]
UNSW_SERVICES = ["-", "dns", "http", "ftp", "ftp-data", "smtp", "ssh", "pop3", "snmp", "ssl", "dhcp", "irc", "radius"]
UNSW_STATES = ["FIN", "INT", "CON", "REQ", "RST", "ACC", "CLO"]
CLS = {"test_scores": M.CLS_SCORES}


def card(title: str, line: str, alert: bool) -> str:
    bg, border, ink = ("#fdeee8", "#e5481c", "#8a2a0e") if alert else ("#e8f3ef", "#0f6e5a", "#0b4f41")
    return (f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:14px 16px">'
            f'<div style="color:{ink};font-weight:700;font-size:1.1rem">{title}</div>'
            f'<div style="color:{ink};margin-top:4px">{line}</div></div>')


def classify(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts, src_port=49152, dst_internal=True):
    out = M.classify(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts, src_port, dst_internal)
    top = out["prediction"]
    html = card("Normal traffic" if top == "normal" else f"Attack: {out['label']}",
                f"{out['confidence']:.1%} confidence. {out['note']}", top != "normal")
    return html, {name(k): v for k, v in out["probabilities"].items()}, out


def cluster(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts, src_port=49152, dst_internal=True):
    out = M.cluster(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts)
    mix = ", ".join(f"{name(k)} {v:.0%}" for k, v in list(out["mix"].items())[:3])
    html = card(f"Cluster {out['cluster']} of {out['k']}", f"Flows in this cluster during training: {mix}.", False)
    return html, {name(k): v for k, v in out["mix"].items()}, out


def regress(proto, service, state, dur, spkts, dpkts, sbytes, sttl, dttl, dloss):
    out = M.regress(proto, service, state, dur, spkts, dpkts, sbytes, sttl, dttl, dloss)
    html = card(f"About {out['bytes']:,.0f} bytes back",
                f"The trees agree on {out['low']:,.0f} to {out['high']:,.0f} bytes (10th to 90th percentile).", False)
    return html, out


# ---------------------------------------------------------------- interface
# real test-set flows, one per traffic type, that the model classifies correctly (chosen in build_models.py)
TON_EXAMPLES = [p["values"][:9] + [p["values"][9], p["dst_internal"]] for p in M.CLS_PRESETS.values()]


def ton_inputs():
    with gr.Row():
        proto = gr.Dropdown(PROTOS, value="tcp", label="Protocol")
        service = gr.Dropdown(SERVICES, value="http", label="Service ('-' for none)")
        state = gr.Dropdown(STATES, value="SF", label="Connection state")
    with gr.Row():
        port = gr.Number(80, label="Destination port", precision=0)
        dur = gr.Number(0.22, label="Duration (s)")
    with gr.Row():
        sb = gr.Number(320, label="Bytes sent", precision=0)
        db = gr.Number(1920, label="Bytes received", precision=0)
    with gr.Row():
        sp = gr.Number(5, label="Packets sent", precision=0)
        dp = gr.Number(6, label="Packets received", precision=0)
    with gr.Row():
        sport = gr.Number(49152, label="Source port", precision=0)
        internal = gr.Checkbox(True, label="Destination inside the network (192.168.1.x)")
    return [proto, service, state, port, dur, sb, db, sp, dp, sport, internal]


with gr.Blocks(title="CloudShield · live models") as demo:
    gr.Markdown(
        f"# CloudShield · live models\n"
        f"Three trained models from the CloudShield project. Classification and clustering use **TON_IoT** network flows; "
        f"regression uses **UNSW-NB15**. Classification was evaluated on machines unseen in training "
        f"(accuracy {CLS['test_scores']['acc']:.2%}, weighted F1 {CLS['test_scores']['f1w']:.3f})."
    )
    with gr.Tab("Classify a flow"):
        with gr.Row():
            with gr.Column():
                c_in = ton_inputs()
                c_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                c_html, c_label, c_json = gr.HTML(), gr.Label(num_top_classes=5, label="Class probabilities"), gr.JSON(label="API result")
        c_btn.click(classify, c_in, [c_html, c_label, c_json], api_name="classify")
        gr.Examples(TON_EXAMPLES, c_in)

    with gr.Tab("Find its cluster"):
        with gr.Row():
            with gr.Column():
                k_in = ton_inputs()
                k_btn = gr.Button("Assign cluster", variant="primary")
            with gr.Column():
                k_html, k_label, k_json = gr.HTML(), gr.Label(label="Traffic types in that cluster (training data)"), gr.JSON(label="API result")
        k_btn.click(cluster, k_in, [k_html, k_label, k_json], api_name="cluster")
        gr.Examples(TON_EXAMPLES, k_in)

    with gr.Tab("Forecast reply size"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    r_proto = gr.Dropdown(UNSW_PROTOS, value="tcp", label="Protocol")
                    r_service = gr.Dropdown(UNSW_SERVICES, value="-", label="Service")
                    r_state = gr.Dropdown(UNSW_STATES, value="FIN", label="State")
                with gr.Row():
                    r_dur = gr.Number(1.85, label="Duration (s)")
                    r_sbytes = gr.Number(1320, label="Bytes sent", precision=0)
                with gr.Row():
                    r_spkts = gr.Number(28, label="Packets sent", precision=0)
                    r_dpkts = gr.Number(72, label="Packets received", precision=0)
                with gr.Row():
                    r_sttl = gr.Number(62, label="Source TTL", precision=0)
                    r_dttl = gr.Number(252, label="Destination TTL", precision=0)
                    r_dloss = gr.Number(0, label="Packets lost on the way back", precision=0)
                r_btn = gr.Button("Forecast", variant="primary")
            with gr.Column():
                r_html, r_json = gr.HTML(), gr.JSON(label="API result")
        r_in = [r_proto, r_service, r_state, r_dur, r_spkts, r_dpkts, r_sbytes, r_sttl, r_dttl, r_dloss]
        r_btn.click(regress, r_in, [r_html, r_json], api_name="regress")

if __name__ == "__main__":
    demo.launch()
