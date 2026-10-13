---
title: CloudShield - Live Models
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.9.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# CloudShield: live models

The three final models of the CloudShield project (23CSE301 Machine Learning capstone), each behind an input form and an API endpoint.

| Tab | API endpoint | Dataset | Model | Test result |
|---|---|---|---|---|
| Classify a flow | `/classify` | TON_IoT (network) | Gradient Boosting, tuned (100 trees, depth 5, learning rate 0.05, subsample 0.8) | accuracy 0.8365, weighted F1 0.8742, macro F1 0.5345 |
| Find its cluster | `/cluster` | TON_IoT (network) | K-Means, k = 6 (elbow) | silhouette 0.474 |
| Forecast reply size | `/regress` | UNSW-NB15 | Random Forest (200 trees) | R² 0.9994 on `log1p(dbytes)` |

The classifier was evaluated on machines it never saw in training: the flows were split with `StratifiedGroupKFold` grouped by source IP, so every test flow comes from an unseen machine.

## Inputs

- **Classify / cluster:** protocol, service, connection state, destination port, duration, bytes and packets in each direction. The remaining TON_IoT columns (DNS, SSL, HTTP details) are set to their "not applicable" values, and IP-level byte counts add a 40-byte header per packet.
- **Forecast:** protocol, service, state, duration, bytes and packets sent, packets received, packets lost on the way back, and both TTLs. The remaining UNSW-NB15 columns take their typical (median or most common) training values.

## Calling the API

```python
from gradio_client import Client

client = Client("pranayraparla/Ml_Capstone")
html, probabilities, result = client.predict(
    proto="udp", service="dns", conn_state="SF", dst_port=53, duration=0.004,
    src_bytes=74, dst_bytes=90, src_pkts=2, dst_pkts=2, api_name="/classify",
)
print(result["prediction"], result["confidence"])
```

```js
import { Client } from "@gradio/client";
const client = await Client.connect("pranayraparla/Ml_Capstone");
const res = await client.predict("/regress", { proto: "tcp", service: "-", state: "FIN", dur: 1.85, spkts: 28, dpkts: 72, sbytes: 1320, sttl: 62, dttl: 252, dloss: 2 });
```

## Files

| File | Contents |
|---|---|
| `app.py` | Gradio interface and the three prediction functions |
| `pipelines.py` | Preprocessing copied from the project notebooks, so inputs are transformed exactly as in training |
| `models/*.joblib` | Fitted models with their preprocessing (built by `app/build_models.py` in the project repository) |

Datasets: TON_IoT (Moustafa, 2021, *Sustainable Cities and Society* 72) and UNSW-NB15 (Moustafa and Slay, 2015, MilCIS).
