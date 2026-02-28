# dashboard.py
import json as _json
import os
import webbrowser

from .sentinel_config import OUTPUT_HTML


def render_html(
    accuracy,
    auc,
    report,
    tp,
    tn,
    fp,
    fn,
    total_samples,
    total_bots,
    total_humans,
    bot_pct,
    feat_names,
    feat_scores,
    sample_users,
    roc_fpr,
    roc_tpr,
    test_size,
    train_size,
    n_features,
):
    h = report.get("Human", {})
    b = report.get("Bot", {})
    m = report.get("macro avg", {})
    w = report.get("weighted avg", {})

    h_prec = round(h.get("precision", 0), 4)
    h_rec = round(h.get("recall", 0), 4)
    h_f1 = round(h.get("f1-score", 0), 4)
    h_sup = int(h.get("support", 0))

    b_prec = round(b.get("precision", 0), 4)
    b_rec = round(b.get("recall", 0), 4)
    b_f1 = round(b.get("f1-score", 0), 4)
    b_sup = int(b.get("support", 0))

    m_prec = round(m.get("precision", 0), 4)
    m_rec = round(m.get("recall", 0), 4)
    m_f1 = round(m.get("f1-score", 0), 4)

    w_prec = round(w.get("precision", 0), 4)
    w_rec = round(w.get("recall", 0), 4)
    w_f1 = round(w.get("f1-score", 0), 4)

    human_fp_rate = round(fp / (fp + tn) * 100, 1) if (fp + tn) > 0 else 0
    bot_miss_rate = round(fn / (fn + tp) * 100, 1) if (fn + tp) > 0 else 0

    feat_json = _json.dumps(list(zip(feat_names, feat_scores)))
    users_json = _json.dumps(sample_users)
    roc_json = _json.dumps({"fpr": roc_fpr, "tpr": roc_tpr})

    # Keep your original HTML template body here instead of this placeholder.
    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL — Fake Engagement Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
    :root{{
    --bg:#030712;--surface:#0d1117;--surface2:#161b22;--border:#21262d;
    --accent:#00ff9d;--accent2:#ff4d6d;--accent3:#a78bfa;--accent4:#38bdf8;
    --warn:#ff9900;--text:#e6edf3;--dim:#7d8590;
    --bot:#ff4d6d;--human:#00ff9d;
    }}
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;overflow-x:hidden;}}
    body::before{{content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(0,255,157,.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,255,157,.025) 1px,transparent 1px);
    background-size:44px 44px;pointer-events:none;z-index:0;}}
    .scanline{{position:fixed;top:-120px;left:0;right:0;height:120px;
    background:linear-gradient(transparent,rgba(0,255,157,.025),transparent);
    animation:scan 7s linear infinite;pointer-events:none;z-index:0;}}
    @keyframes scan{{to{{top:100vh;}}}}
    .wrap{{position:relative;z-index:1;max-width:1440px;margin:0 auto;padding:0 28px;}}

    /* HEADER */
    header{{padding:36px 0 28px;border-bottom:1px solid var(--border);margin-bottom:44px;}}
    .hdr-inner{{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;}}
    .brand{{font-size:34px;font-weight:800;letter-spacing:-1.5px;}}
    .brand span{{color:var(--accent);}}
    .tagline{{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-top:5px;}}
    .live{{display:inline-flex;align-items:center;gap:7px;background:rgba(0,255,157,.08);
    border:1px solid rgba(0,255,157,.25);padding:5px 12px;
    font-family:'Space Mono',monospace;font-size:10px;color:var(--accent);letter-spacing:1px;}}
    .dot{{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:blink 1.4s ease-in-out infinite;}}
    @keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:.3;}}}}
    .hdr-meta{{font-family:'Space Mono',monospace;font-size:11px;color:var(--dim);text-align:right;line-height:1.9;}}

    /* METRICS STRIP */
    .strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border);
    border:1px solid var(--border);margin-bottom:44px;animation:up .6s ease both;}}
    .mcell{{background:var(--surface);padding:26px 22px;position:relative;overflow:hidden;
    transition:background .2s;cursor:default;}}
    .mcell:hover{{background:var(--surface2);}}
    .mcell::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:var(--c,var(--accent));transform:scaleX(0);transform-origin:left;transition:transform .4s;}}
    .mcell:hover::after{{transform:scaleX(1);}}
    .mlabel{{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;
    text-transform:uppercase;color:var(--dim);margin-bottom:10px;}}
    .mval{{font-size:38px;font-weight:800;line-height:1;color:var(--c,var(--accent));letter-spacing:-2px;}}
    .msub{{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);margin-top:7px;}}

    /* PANEL */
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:22px;}}
    .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px;margin-bottom:22px;}}
    .full{{grid-column:1/-1;}}
    .panel{{background:var(--surface);border:1px solid var(--border);padding:26px;
    animation:up .7s ease both;}}
    .ph{{display:flex;align-items:center;justify-content:space-between;
    margin-bottom:22px;padding-bottom:14px;border-bottom:1px solid var(--border);}}
    .pt{{font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;}}
    .pbadge{{font-family:'Space Mono',monospace;font-size:9px;padding:3px 8px;border:1px solid;letter-spacing:1px;}}

    /* TABLE */
    .rtable{{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:12px;}}
    .rtable th{{color:var(--dim);font-size:9px;letter-spacing:1px;text-transform:uppercase;
    text-align:right;padding:7px 10px;border-bottom:1px solid var(--border);}}
    .rtable th:first-child{{text-align:left;}}
    .rtable td{{padding:11px 10px;text-align:right;border-bottom:1px solid rgba(33,38,45,.5);}}
    .rtable td:first-child{{text-align:left;font-weight:700;}}
    .rtable tr:hover td{{background:rgba(255,255,255,.02);}}
    .ch{{color:var(--human);}} .cb{{color:var(--bot);}} .cm{{color:var(--accent4);}} .cw{{color:var(--accent3);}}

    /* FEATURE BARS */
    .frow{{display:flex;align-items:center;gap:10px;margin-bottom:9px;animation:slidein .5s ease both;}}
    @keyframes slidein{{from{{opacity:0;transform:translateX(-18px);}}to{{opacity:1;transform:translateX(0);}}}}
    .fname{{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);
    width:200px;flex-shrink:0;text-align:right;}}
    .fwrap{{flex:1;height:18px;background:rgba(255,255,255,.04);border:1px solid var(--border);
    position:relative;overflow:hidden;}}
    .fbar{{height:100%;transition:width 1.1s cubic-bezier(.16,1,.3,1);
    display:flex;align-items:center;justify-content:flex-end;padding-right:6px;}}
    .fbar::after{{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.1));}}
    .fscore{{font-family:'Space Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);position:relative;z-index:1;}}

    /* CONFUSION MATRIX */
    .cmwrap{{display:flex;align-items:flex-start;gap:28px;flex-wrap:wrap;}}
    .cmatrix{{display:grid;grid-template-columns:auto 1fr 1fr;grid-template-rows:auto 1fr 1fr;gap:2px;}}
    .cmlbl{{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);
    display:flex;align-items:center;justify-content:center;padding:8px;letter-spacing:1px;text-transform:uppercase;}}
    .cmcell{{width:106px;height:86px;display:flex;flex-direction:column;align-items:center;
    justify-content:center;border:1px solid var(--border);transition:transform .2s;cursor:default;}}
    .cmcell:hover{{transform:scale(1.04);z-index:1;position:relative;}}
    .cmnum{{font-size:30px;font-weight:800;letter-spacing:-1.5px;}}
    .cmsub{{font-family:'Space Mono',monospace;font-size:9px;color:rgba(255,255,255,.35);margin-top:3px;}}
    .tp{{background:rgba(0,255,157,.1);}} .tp .cmnum{{color:var(--human);}}
    .tn{{background:rgba(0,255,157,.06);}} .tn .cmnum{{color:var(--accent);}}
    .fp{{background:rgba(255,77,109,.1);}} .fp .cmnum{{color:var(--bot);}}
    .fn{{background:rgba(255,77,109,.06);}} .fn .cmnum{{color:var(--warn);}}
    .cmlegend{{flex:1;min-width:180px;}}
    .legrow{{display:flex;align-items:flex-start;gap:10px;margin-bottom:14px;}}
    .legsq{{width:22px;height:22px;flex-shrink:0;border:1px solid;}}
    .legtxt{{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);line-height:1.7;}}

    /* USER CARDS */
    .ugrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px;}}
    .ucard{{border:1px solid var(--border);padding:18px 15px;position:relative;overflow:hidden;
    transition:transform .2s,border-color .2s;cursor:pointer;animation:up .8s ease both;}}
    .ucard:hover{{transform:translateY(-4px);}}
    .ucard::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--uc);}}
    .uhandle{{font-family:'Space Mono',monospace;font-size:11px;color:var(--text);
    margin-bottom:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
    .uhandle span{{color:var(--dim);}}
    .vbadge{{display:inline-block;font-family:'Space Mono',monospace;font-size:9px;
    letter-spacing:1px;padding:3px 8px;border:1px solid;margin-bottom:13px;}}
    .vbot{{color:var(--bot);border-color:var(--bot);background:rgba(255,77,109,.08);}}
    .vhuman{{color:var(--human);border-color:var(--human);background:rgba(0,255,157,.08);}}
    .gwrap{{height:5px;background:rgba(255,255,255,.06);margin-bottom:7px;overflow:hidden;}}
    .gfill{{height:100%;transition:width 1.3s cubic-bezier(.16,1,.3,1);}}
    .uprob{{font-size:26px;font-weight:800;letter-spacing:-1px;margin-bottom:2px;}}
    .uplabel{{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);}}
    .umeta{{margin-top:10px;font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);line-height:1.9;}}
    .correct-tick{{position:absolute;top:10px;right:10px;font-size:10px;}}

    /* ROC CHART */
    .roc-wrap{{position:relative;}}

    /* ANOMALY */
    .agrid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
    .aitem{{border:1px solid var(--border);padding:18px;position:relative;}}
    .atitle{{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
    display:flex;align-items:center;gap:7px;margin-bottom:7px;}}
    .adot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}
    .adesc{{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);line-height:1.75;}}
    .ascore{{position:absolute;top:14px;right:14px;font-family:'Space Mono',monospace;font-size:16px;font-weight:700;}}

    /* ARCH */
    .arch{{display:flex;align-items:center;overflow-x:auto;padding-bottom:8px;gap:0;}}
    .anode{{flex-shrink:0;background:var(--surface2);border:1px solid var(--border);
    padding:14px 18px;text-align:center;min-width:118px;}}
    .ant{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
    color:var(--accent3);margin-bottom:5px;}}
    .and{{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);line-height:1.7;}}
    .aarrow{{color:var(--dim);font-size:16px;padding:0 5px;flex-shrink:0;}}

    /* FOOTER */
    footer{{border-top:1px solid var(--border);margin-top:44px;padding:22px 0;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;}}
    .fl{{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);}}
    .fr{{display:flex;gap:18px;font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);letter-spacing:1px;}}

    @keyframes up{{from{{opacity:0;transform:translateY(22px);}}to{{opacity:1;transform:translateY(0);}}}}
    @media(max-width:900px){{
    .grid2,.grid3{{grid-template-columns:1fr;}}
    .full{{grid-column:1;}}
    .strip{{grid-template-columns:repeat(2,1fr);}}
    .agrid{{grid-template-columns:1fr;}}
    .cmwrap{{flex-direction:column;}}
    }}
    </style>
    </head>
    <body>
    <div class="scanline"></div>
    <div class="wrap">

    <!-- HEADER -->
    <header>
    <div class="hdr-inner">
        <div>
        <div class="brand">SENTI<span>NEL</span></div>
        <div class="tagline">Fake Engagement Detection · Problem Statement 3 · TwiBot-20</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:10px;">
        <div class="live"><div class="dot"></div>RESULTS LOADED</div>
        <div class="hdr-meta">
            Dataset: TwiBot-20 &nbsp;·&nbsp; {total_samples:,} accounts<br>
            Train: {train_size:,} &nbsp;·&nbsp; Test: {test_size:,} samples<br>
            Features: {n_features} behavioral dimensions
        </div>
        </div>
    </div>
    </header>

    <!-- METRICS STRIP -->
    <div class="strip">
    <div class="mcell" style="--c:var(--accent)">
        <div class="mlabel">Accuracy</div>
        <div class="mval" id="m0">0%</div>
        <div class="msub">held-out test set</div>
    </div>
    <div class="mcell" style="--c:var(--accent4)">
        <div class="mlabel">AUC-ROC</div>
        <div class="mval" id="m1">0.00</div>
        <div class="msub">discrimination power</div>
    </div>
    <div class="mcell" style="--c:var(--bot)">
        <div class="mlabel">Bot Recall</div>
        <div class="mval" id="m2">0%</div>
        <div class="msub">bots caught</div>
    </div>
    <div class="mcell" style="--c:var(--human)">
        <div class="mlabel">Human Precision</div>
        <div class="mval" id="m3">0%</div>
        <div class="msub">true human rate</div>
    </div>
    <div class="mcell" style="--c:var(--accent3)">
        <div class="mlabel">Total Accounts</div>
        <div class="mval" id="m4">0</div>
        <div class="msub">{bot_pct:.1f}% bots · {100-bot_pct:.1f}% humans</div>
    </div>
    </div>

    <!-- ROW 1: Classification + Confusion -->
    <div class="grid2">
    <div class="panel">
        <div class="ph">
        <div class="pt">Classification Report</div>
        <div class="pbadge" style="color:var(--accent);border-color:var(--accent)">XGBoost · 300 trees</div>
        </div>
        <table class="rtable">
        <thead><tr>
            <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
        </tr></thead>
        <tbody>
            <tr><td class="ch">Human</td><td>{h_prec}</td><td>{h_rec}</td><td>{h_f1}</td><td>{h_sup:,}</td></tr>
            <tr><td class="cb">Bot</td><td>{b_prec}</td><td>{b_rec}</td><td>{b_f1}</td><td>{b_sup:,}</td></tr>
            <tr style="border-top:1px solid var(--border)">
            <td class="cm">Macro Avg</td><td>{m_prec}</td><td>{m_rec}</td><td>{m_f1}</td><td>{test_size:,}</td>
            </tr>
            <tr>
            <td class="cw">Weighted Avg</td><td>{w_prec}</td><td>{w_rec}</td><td>{w_f1}</td><td>{test_size:,}</td>
            </tr>
        </tbody>
        </table>
        <div style="margin-top:24px;">
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);
            letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;">Metrics Radar</div>
        <canvas id="radarChart" width="310" height="220"></canvas>
        </div>
    </div>

    <div class="panel">
        <div class="ph">
        <div class="pt">Confusion Matrix</div>
        <div class="pbadge" style="color:var(--accent4);border-color:var(--accent4)">TEST · {test_size:,}</div>
        </div>
        <div class="cmwrap">
        <div>
            <div style="font-family:'Space Mono',monospace;font-size:8px;color:var(--dim);
            letter-spacing:1px;text-align:center;margin-bottom:8px;">PREDICTED →</div>
            <div class="cmatrix">
            <div class="cmlbl"></div>
            <div class="cmlbl">Human</div>
            <div class="cmlbl">Bot</div>
            <div class="cmlbl" style="writing-mode:vertical-lr;transform:rotate(180deg)">ACTUAL</div>
            <div class="cmcell tp"><div class="cmnum" id="cm0">0</div><div class="cmsub">True Human</div></div>
            <div class="cmcell fp"><div class="cmnum" id="cm1">0</div><div class="cmsub">False Bot</div></div>
            <div style=""></div>
            <div class="cmcell fn"><div class="cmnum" id="cm2">0</div><div class="cmsub">Missed Bot</div></div>
            <div class="cmcell tn"><div class="cmnum" id="cm3">0</div><div class="cmsub">True Bot</div></div>
            </div>
        </div>
        <div class="cmlegend">
            <div class="legrow">
            <div class="legsq" style="background:rgba(0,255,157,.1);border-color:var(--human);"></div>
            <div class="legtxt"><strong style="color:var(--human)">True Human ({tn:,})</strong><br>
                Correctly kept as human.<br>Precision: {h_prec}</div>
            </div>
            <div class="legrow">
            <div class="legsq" style="background:rgba(0,255,157,.06);border-color:var(--accent);"></div>
            <div class="legtxt"><strong style="color:var(--accent)">True Bot ({tp:,})</strong><br>
                Bots correctly caught.<br>Recall: {b_rec}</div>
            </div>
            <div class="legrow">
            <div class="legsq" style="background:rgba(255,77,109,.1);border-color:var(--bot);"></div>
            <div class="legtxt"><strong style="color:var(--bot)">False Bot ({fp:,})</strong><br>
                Humans flagged as bots.<br>{human_fp_rate}% false alarm rate</div>
            </div>
            <div class="legrow">
            <div class="legsq" style="background:rgba(255,153,0,.08);border-color:var(--warn);"></div>
            <div class="legtxt"><strong style="color:var(--warn)">Missed Bot ({fn:,})</strong><br>
                Bots slipping through.<br>{bot_miss_rate}% escape rate</div>
            </div>
        </div>
        </div>

        <!-- ROC CURVE -->
        <div style="margin-top:22px;">
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);
            letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">
            ROC Curve &nbsp;<span style="color:var(--accent)">AUC = {auc:.4f}</span>
        </div>
        <canvas id="rocChart" width="310" height="170"></canvas>
        </div>
    </div>
    </div>

    <!-- ROW 2: Feature Importance + Anomaly Indicators -->
    <div class="grid2">
    <div class="panel">
        <div class="ph">
        <div class="pt">Feature Importance</div>
        <div class="pbadge" style="color:var(--accent3);border-color:var(--accent3)">{n_features} features</div>
        </div>
        <div id="featBars"></div>
    </div>

    <div class="panel">
        <div class="ph">
        <div class="pt">Behavioral Anomaly Indicators</div>
        <div class="pbadge" style="color:var(--accent2);border-color:var(--accent2)">4 dimensions</div>
        </div>
        <div class="agrid">
        <div class="aitem">
            <div class="atitle"><div class="adot" style="background:#ff4d6d;box-shadow:0 0 5px #ff4d6d;"></div>Timing Regularity</div>
            <div class="ascore" style="color:var(--bot)">HIGH</div>
            <div class="adesc">Bots post at suspiciously fixed intervals. Low std_interval_sec and high burst_ratio (posts &lt;60s apart) signal automated tools.</div>
        </div>
        <div class="aitem">
            <div class="atitle"><div class="adot" style="background:#ff9900;box-shadow:0 0 5px #ff9900;"></div>Engagement Bursts</div>
            <div class="ascore" style="color:var(--warn)">MED</div>
            <div class="adesc">daily_count_cv measures day-to-day volatility. Coordinated campaigns show sudden spikes — low CV in bots vs. high CV in humans.</div>
        </div>
        <div class="aitem">
            <div class="atitle"><div class="adot" style="background:#38bdf8;box-shadow:0 0 5px #38bdf8;"></div>Profile Authenticity</div>
            <div class="ascore" style="color:var(--accent4)">KEY</div>
            <div class="adesc">verified dominates at {feat_scores[0]:.1%} importance. Description length, URL presence, digit-heavy usernames also separate real vs. fake accounts.</div>
        </div>
        <div class="aitem">
            <div class="atitle"><div class="adot" style="background:#a78bfa;box-shadow:0 0 5px #a78bfa;"></div>Network Signals</div>
            <div class="ascore" style="color:var(--accent3)">MOD</div>
            <div class="adesc">follower_following_ratio exposes follow-spam behaviour. listed_count reflects organic curation trust — bots rarely appear in human-made lists.</div>
        </div>
        </div>
    </div>
    </div>

    <!-- USER SCORING -->
    <div class="panel" style="margin-bottom:22px;">
    <div class="ph">
        <div class="pt">Live User Scoring — Sample Accounts</div>
        <div class="pbadge" style="color:var(--accent);border-color:var(--accent)">{len(sample_users)} accounts</div>
    </div>
    <div class="ugrid" id="userGrid"></div>
    </div>

    <!-- PIPELINE -->
    <div class="panel" style="margin-bottom:22px;">
    <div class="ph">
        <div class="pt">Detection Pipeline</div>
        <div class="pbadge" style="color:var(--accent4);border-color:var(--accent4)">End-to-End</div>
    </div>
    <div class="arch">
        <div class="anode" style="border-top:3px solid #38bdf8;">
        <div class="ant">Data Ingestion</div>
        <div class="and">TwiBot-20 JSON<br>{total_samples:,} accounts<br>train/test/dev</div>
        </div>
        <div class="aarrow">→</div>
        <div class="anode" style="border-top:3px solid #a78bfa;">
        <div class="ant">Feature Eng.</div>
        <div class="and">{n_features} behavioral<br>+ profile features<br>Safe type coerce</div>
        </div>
        <div class="aarrow">→</div>
        <div class="anode" style="border-top:3px solid #ff9900;">
        <div class="ant">Timing Analysis</div>
        <div class="and">Intervals, burst<br>Night ratio<br>Entropy scoring</div>
        </div>
        <div class="aarrow">→</div>
        <div class="anode" style="border-top:3px solid #00ff9d;">
        <div class="ant">XGBoost</div>
        <div class="and">300 estimators<br>depth=5, lr=0.05<br>stratified split</div>
        </div>
        <div class="aarrow">→</div>
        <div class="anode" style="border-top:3px solid #ff4d6d;">
        <div class="ant">Evaluation</div>
        <div class="and">{accuracy:.1f}% accuracy<br>{auc:.4f} AUC-ROC<br>Classification rpt</div>
        </div>
        <div class="aarrow">→</div>
        <div class="anode" style="border-top:3px solid #e879f9;">
        <div class="ant">Output Score</div>
        <div class="and">Bot probability<br>Authenticity score<br>Behavioral flags</div>
        </div>
    </div>
    </div>

    <!-- FOOTER -->
    <footer>
    <div class="fl">SENTINEL · Behavior Analytics Hackathon · Problem Statement 3<br>
        Model: XGBoost · Dataset: TwiBot-20 · Features: {n_features} dimensions</div>
    <div class="fr">
        <span>ACC: {accuracy:.1f}%</span>
        <span>AUC: {auc:.4f}</span>
        <span>F1-Bot: {b_f1}</span>
        <span>N: {total_samples:,}</span>
    </div>
    </footer>
    </div>

    <script>
    // ─── DATA FROM PYTHON ─────────────────────────────────────
    const ACCURACY  = {accuracy:.4f};
    const AUC       = {auc:.4f};
    const BOT_REC   = {b_rec:.4f};
    const HUM_PREC  = {h_prec:.4f};
    const TOTAL     = {total_samples};
    const TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp};

    const FEATURES  = {feat_json};
    const USERS     = {users_json};
    const ROC       = {roc_json};

    const RADAR_VALS = [
    {h_prec:.4f}, // Human Precision
    {h_rec:.4f},  // Human Recall
    {b_prec:.4f}, // Bot Precision
    {b_rec:.4f},  // Bot Recall
    {b_f1:.4f},   // F1 Bot
    {auc:.4f},    // AUC-ROC
    ];

    // ─── ANIMATED COUNTERS ───────────────────────────────────
    function animateVal(id, target, decimals, suffix, delay) {{
    setTimeout(() => {{
        const el = document.getElementById(id);
        const dur = 1600, t0 = performance.now();
        function step(now) {{
        const t = Math.min((now - t0) / dur, 1);
        const e = 1 - Math.pow(1 - t, 4);
        const v = target * e;
        el.textContent = decimals > 0
            ? v.toFixed(decimals) + suffix
            : Math.round(v) + suffix;
        if (t < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }}, delay);
    }}

    animateVal('m0', ACCURACY,    1, '%',   300);
    animateVal('m1', AUC,         4, '',    450);
    animateVal('m2', BOT_REC*100, 1, '%',   600);
    animateVal('m3', HUM_PREC*100,1, '%',   750);
    animateVal('m4', TOTAL,       0, '',    900);
    animateVal('cm0', TN, 0, '', 400);
    animateVal('cm1', FP, 0, '', 550);
    animateVal('cm2', FN, 0, '', 700);
    animateVal('cm3', TP, 0, '', 850);

    // ─── FEATURE BARS ────────────────────────────────────────
    const barColors = [
    '#00ff9d','#38bdf8','#38bdf8','#a78bfa','#a78bfa',
    '#a78bfa','#a78bfa','#ff9900','#a78bfa','#ff9900',
    '#a78bfa','#ff9900','#ff9900','#38bdf8','#38bdf8'
    ];
    const maxScore = FEATURES[0][1];
    const fbContainer = document.getElementById('featBars');
    FEATURES.forEach(([name, score], i) => {{
    const pct = (score / maxScore) * 100;
    const d = document.createElement('div');
    d.className = 'frow';
    d.style.animationDelay = `${{i * 0.04}}s`;
    d.innerHTML = `
        <div class="fname">${{name}}</div>
        <div class="fwrap">
        <div class="fbar" id="fb${{i}}"
            style="width:0%;background:${{barColors[i] || '#a78bfa'}}20;border-right:2px solid ${{barColors[i] || '#a78bfa'}};">
            <span class="fscore">${{score.toFixed(4)}}</span>
        </div>
        </div>`;
    fbContainer.appendChild(d);
    setTimeout(() => {{
        document.getElementById(`fb${{i}}`).style.width = pct + '%';
    }}, 500 + i * 55);
    }});

    // ─── USER CARDS ─────────────────────────────────────────
    const ug = document.getElementById('userGrid');
    USERS.forEach((u, i) => {{
    const isBot = u.verdict === 'BOT';
    const color = isBot ? 'var(--bot)' : 'var(--human)';
    const card = document.createElement('div');
    card.className = 'ucard';
    card.style.cssText = `--uc:${{color}};animation-delay:${{0.08*i}}s;`;
    const correctMark = u.correct ? '✓' : '✗';
    const correctColor = u.correct ? 'var(--human)' : 'var(--bot)';
    card.innerHTML = `
        <div class="correct-tick" style="color:${{correctColor}}">${{correctMark}}</div>
        <div class="uhandle"><span>@</span>${{u.handle}}</div>
        <div class="vbadge ${{isBot ? 'vbot' : 'vhuman'}}">${{u.verdict}}</div>
        <div class="gwrap"><div class="gfill" id="g${{i}}" style="width:0%;background:${{color}};"></div></div>
        <div class="uprob" id="up${{i}}" style="color:${{color}}">0%</div>
        <div class="uplabel">Bot Probability</div>
        <div class="umeta">
        Auth: ${{u.authScore.toFixed(1)}}%<br>
        True: ${{u.trueLabel}}<br>
        Burst: ${{u.burst.toFixed(3)}} · CV: ${{u.dailyCV.toFixed(3)}}
        </div>`;
    ug.appendChild(card);
    setTimeout(() => {{
        document.getElementById(`g${{i}}`).style.width = u.botProb + '%';
        animateVal(`up${{i}}`, u.botProb, 2, '%', 0);
    }}, 600 + i * 100);
    }});

    // ─── RADAR CHART ─────────────────────────────────────────
    (function() {{
    const cvs = document.getElementById('radarChart');
    const ctx = cvs.getContext('2d');
    const W = cvs.width, H = cvs.height;
    const cx = W/2, cy = H/2+8, r = Math.min(W,H)*0.33;
    const labels = ['H-Prec','H-Rec','B-Prec','B-Rec','F1-Bot','AUC'];
    const n = labels.length;

    function pt(angle, radius) {{
        return [cx + radius*Math.cos(angle - Math.PI/2),
                cy + radius*Math.sin(angle - Math.PI/2)];
    }}

    let prog = 0;
    function draw(p) {{
        ctx.clearRect(0,0,W,H);
        for (let ring=1; ring<=5; ring++) {{
        const rr = r*ring/5;
        ctx.beginPath();
        for (let i=0; i<=n; i++) {{
            const [x,y] = pt((i/n)*2*Math.PI, rr);
            i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }}
        ctx.closePath();
        ctx.strokeStyle='rgba(255,255,255,.06)'; ctx.lineWidth=1; ctx.stroke();
        }}
        for (let i=0; i<n; i++) {{
        const [x,y] = pt((i/n)*2*Math.PI, r);
        ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(x,y);
        ctx.strokeStyle='rgba(255,255,255,.08)'; ctx.lineWidth=1; ctx.stroke();
        }}
        ctx.beginPath();
        for (let i=0; i<=n; i++) {{
        const angle = (i/n)*2*Math.PI;
        const val = RADAR_VALS[i%n]*p;
        const [x,y] = pt(angle, val*r);
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }}
        ctx.closePath();
        ctx.fillStyle='rgba(0,255,157,.07)'; ctx.fill();
        ctx.strokeStyle='rgba(0,255,157,.65)'; ctx.lineWidth=1.5; ctx.stroke();
        for (let i=0; i<n; i++) {{
        const [x,y] = pt((i/n)*2*Math.PI, RADAR_VALS[i]*p*r);
        ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2);
        ctx.fillStyle='#00ff9d'; ctx.fill();
        }}
        ctx.font='9px Space Mono,monospace';
        ctx.fillStyle='rgba(125,133,144,.85)'; ctx.textAlign='center';
        for (let i=0; i<n; i++) {{
        const [x,y] = pt((i/n)*2*Math.PI, r+20);
        ctx.fillText(labels[i], x, y+4);
        }}
    }}
    function anim() {{
        prog = Math.min(prog+0.03, 1);
        draw(1 - Math.pow(1-prog,3));
        if (prog<1) requestAnimationFrame(anim);
    }}
    setTimeout(anim, 700);
    }})();

    // ─── ROC CURVE ───────────────────────────────────────────
    (function() {{
    const cvs = document.getElementById('rocChart');
    const ctx = cvs.getContext('2d');
    const W = cvs.width, H = cvs.height;
    const pad = {{l:34,r:14,t:10,b:28}};
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

    function toXY(fpr, tpr) {{
        return [pad.l + fpr*pw, pad.t + (1-tpr)*ph];
    }}

    // Axes
    ctx.strokeStyle='rgba(255,255,255,.1)'; ctx.lineWidth=1;
    ctx.strokeRect(pad.l, pad.t, pw, ph);

    // Grid lines
    [.25,.5,.75].forEach(v => {{
        ctx.beginPath();
        const [x1,y1] = toXY(v,0), [x2,y2] = toXY(v,1);
        ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle='rgba(255,255,255,.05)'; ctx.stroke();
        ctx.beginPath();
        const [xa,ya] = toXY(0,v), [xb,yb] = toXY(1,v);
        ctx.moveTo(xa,ya); ctx.lineTo(xb,yb); ctx.stroke();
    }});

    // Diagonal
    ctx.beginPath();
    ctx.setLineDash([4,4]);
    ctx.moveTo(...toXY(0,0)); ctx.lineTo(...toXY(1,1));
    ctx.strokeStyle='rgba(255,255,255,.15)'; ctx.lineWidth=1; ctx.stroke();
    ctx.setLineDash([]);

    // Axis labels
    ctx.font='8px Space Mono,monospace'; ctx.fillStyle='rgba(125,133,144,.7)';
    ctx.textAlign='center';
    ctx.fillText('False Positive Rate', pad.l+pw/2, H-4);
    ctx.save(); ctx.translate(10, pad.t+ph/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('True Positive Rate', 0, 0); ctx.restore();
    [0,0.5,1].forEach(v => {{
        ctx.textAlign='right';
        ctx.fillText(v.toFixed(1), pad.l-4, pad.t+(1-v)*ph+3);
        ctx.textAlign='center';
        ctx.fillText(v.toFixed(1), pad.l+v*pw, pad.t+ph+12);
    }});

    // ROC line (animated draw)
    const pts = ROC.fpr.map((f,i) => toXY(f, ROC.tpr[i]));
    let pIdx = 0;
    function drawROC() {{
        if (pIdx >= pts.length-1) return;
        pIdx = Math.min(pIdx+3, pts.length-1);
        // Draw filled area
        ctx.beginPath();
        ctx.moveTo(...toXY(0,0));
        for (let i=0; i<=pIdx; i++) ctx.lineTo(...pts[i]);
        ctx.lineTo(...toXY(pts[pIdx][0] > pad.l ? (pts[pIdx][0]-pad.l)/pw : 0, 0));
        ctx.closePath();
        ctx.fillStyle='rgba(0,255,157,.05)'; ctx.fill();
        // Draw line
        ctx.beginPath();
        ctx.moveTo(...pts[0]);
        for (let i=1; i<=pIdx; i++) ctx.lineTo(...pts[i]);
        ctx.strokeStyle='rgba(0,255,157,.8)'; ctx.lineWidth=2; ctx.stroke();
        if (pIdx < pts.length-1) requestAnimationFrame(drawROC);
    }}
    setTimeout(drawROC, 900);
    }})();
    </script>
    </body>
    </html>"""


    return html


def save_and_open_dashboard(html):
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print(f" ✓ Dashboard saved: {OUTPUT_HTML}")
    print("=" * 60 + "\n")

    try:
        webbrowser.open("file://" + os.path.abspath(OUTPUT_HTML))
        print(" Opening in browser...")
    except Exception:
        print(f" Open manually: {os.path.abspath(OUTPUT_HTML)}")
