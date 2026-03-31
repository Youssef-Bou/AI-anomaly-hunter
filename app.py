"""
AI Anomaly Hunter — Web Visualizer
Implements the same logic as check_anomaly.py, served as a Flask web app.
Put TRAIN_model.keras in the same folder, then run: python app.py
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = Flask(__name__)

MODEL_PATH = Path("TRAIN_model.keras")
DATA_POINTS = 1000
FLATTEN_WINDOW = 401

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {MODEL_PATH}\n"
            "Bitte fuehre zuerst die Pipeline aus: python TRAIN_HUNT_pipeline.py"
        )
    from tensorflow.keras.models import load_model as _load
    logging.info("Loading model from %s", MODEL_PATH)
    _model = _load(MODEL_PATH)
    return _model


def download_and_preprocess(star_id: str, author: str, quarter) -> np.ndarray:
    import lightkurve as lk
    kwargs = {"author": author}
    if quarter is not None:
        kwargs["quarter"] = quarter
    logging.info("Searching lightcurve: %s (author=%s, quarter=%s)", star_id, author, quarter)
    search = lk.search_lightcurve(star_id, **kwargs)
    if len(search) == 0:
        raise ValueError(f"Keine Lichtkurve fuer '{star_id}' gefunden (author={author}, quarter={quarter}).")
    lc = search.download()
    cleaned = lc.remove_nans().normalize()
    if len(cleaned.flux) < 10:
        raise ValueError(f"Zu wenige Datenpunkte nach NaN-Entfernung: {len(cleaned.flux)}")
    window = FLATTEN_WINDOW
    if window >= len(cleaned.flux):
        window = len(cleaned.flux) - 2
        if window % 2 == 0:
            window -= 1
    if window < 5:
        raise ValueError("Lichtkurve zu kurz zum Glaetten.")
    flattened = cleaned.flatten(window_length=window)
    x_old = np.linspace(0, 1, len(flattened.flux))
    x_new = np.linspace(0, 1, DATA_POINTS)
    y = np.interp(x_new, x_old, flattened.flux.value).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Preprocessed array enthaelt nicht-endliche Werte.")
    return y


def classify(max_dip: float, threshold_planet=0.001, threshold_binary=0.05):
    if max_dip > threshold_binary:
        return "DOPPELSTERN (Eclipsing Binary)", f"Einbruch {max_dip:.2%} — zu tief fuer einen Planeten-Transit.", "binary"
    if max_dip > threshold_planet:
        return "EXOPLANET-KANDIDAT", f"Tiefe {max_dip:.2%} liegt im Bereich eines Planeten-Transits.", "planet"
    return "Wahrscheinlich Rauschen", f"Kein signifikanter Einbruch erkennbar (Tiefe {max_dip:.2%}).", "noise"


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    star_id = (data.get("star_id") or "").strip()
    author = data.get("author", "Kepler").strip() or "Kepler"
    quarter_raw = data.get("quarter", "10")
    threshold_planet = float(data.get("threshold_planet", 0.001))
    threshold_binary = float(data.get("threshold_binary", 0.05))

    try:
        quarter = int(quarter_raw) if str(quarter_raw).strip() not in ("", "0", "none") else None
    except Exception:
        quarter = None

    if not star_id:
        return jsonify({"error": "Bitte eine Stern-ID eingeben."}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        y_true = download_and_preprocess(star_id, author, quarter)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Download-Fehler: {e}"}), 500

    try:
        y_pred = model.predict(y_true.reshape(1, DATA_POINTS, 1), verbose=0).flatten()
    except Exception as e:
        return jsonify({"error": f"Modell-Fehler: {e}"}), 500

    residual = np.abs(y_true - y_pred)
    mse = float(np.mean(np.square(y_true - y_pred)))
    max_dip = float(1.0 - np.min(y_true))
    verdict, reason, verdict_type = classify(max_dip, threshold_planet, threshold_binary)

    return jsonify({
        "star_id": star_id,
        "real": y_true.tolist(),
        "reconstructed": y_pred.tolist(),
        "residual": residual.tolist(),
        "mse": round(mse, 6),
        "max_dip": round(max_dip, 6),
        "max_dip_pct": f"{max_dip:.2%}",
        "verdict": verdict,
        "reason": reason,
        "verdict_type": verdict_type,
        "threshold_planet": threshold_planet,
        "threshold_binary": threshold_binary,
        "data_points": DATA_POINTS,
    })


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Anomaly Hunter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root,[data-theme="light"]{
  --bg:#f4f3ef;--surface:#ffffff;--surface-2:#eeede9;--surface-3:#e7e5e0;
  --border:rgba(0,0,0,0.09);--text:#1a1814;--text-muted:#6b6965;--text-faint:#b0afaa;
  --primary:#0284c7;--primary-glow:rgba(2,132,199,.15);
  --accent:#d97706;--accent-glow:rgba(217,119,6,.12);
  --danger:#dc2626;--danger-glow:rgba(220,38,38,.12);
  --success:#16a34a;--success-glow:rgba(22,163,74,.12);
  --purple:#7c3aed;
  --font-body:'Space Grotesk',sans-serif;--font-mono:'Space Mono',monospace;
  --shadow:0 2px 12px rgba(0,0,0,.07);--shadow-lg:0 8px 32px rgba(0,0,0,.10);
  --radius:10px;--radius-sm:6px;--radius-lg:14px;
}
[data-theme="dark"]{
  --bg:#08090d;--surface:#0d0f14;--surface-2:#12141a;--surface-3:#181a22;
  --border:rgba(255,255,255,.07);--text:#dddbd5;--text-muted:#706e68;--text-faint:#2e2d29;
  --primary:#38bdf8;--primary-glow:rgba(56,189,248,.13);
  --accent:#fbbf24;--accent-glow:rgba(251,191,36,.10);
  --danger:#f87171;--danger-glow:rgba(248,113,113,.12);
  --success:#4ade80;--success-glow:rgba(74,222,128,.10);
  --purple:#c084fc;
  --shadow:0 2px 12px rgba(0,0,0,.35);--shadow-lg:0 8px 32px rgba(0,0,0,.55);
}
html{scroll-behavior:smooth}
body{font-family:var(--font-body);background:var(--bg);color:var(--text);min-height:100dvh;line-height:1.6;}
canvas{display:block;max-width:100%;}
#starfield{position:fixed;inset:0;z-index:0;pointer-events:none;}
.app{position:relative;z-index:1;display:flex;flex-direction:column;min-height:100dvh;}
header{display:flex;align-items:center;justify-content:space-between;padding:.9rem 1.75rem;border-bottom:1px solid var(--border);background:color-mix(in srgb,var(--surface) 80%,transparent);backdrop-filter:blur(16px);position:sticky;top:0;z-index:100;}
.logo{display:flex;align-items:center;gap:.7rem;}
.logo-text{font-size:1rem;font-weight:700;letter-spacing:-.02em;background:linear-gradient(120deg,var(--primary),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.logo-sub{font-size:.65rem;color:var(--text-muted);font-family:var(--font-mono);}
.hdr-right{display:flex;gap:.6rem;align-items:center;}
.btn{display:inline-flex;align-items:center;gap:.35rem;padding:.42rem .85rem;border-radius:var(--radius-sm);font-size:.78rem;font-weight:500;cursor:pointer;border:1px solid var(--border);background:var(--surface-2);color:var(--text);transition:all 160ms ease;font-family:var(--font-body);}
.btn:hover{background:var(--primary-glow);border-color:var(--primary);color:var(--primary);}
.btn-primary{background:var(--primary);color:#000;border-color:var(--primary);font-weight:600;}
.btn-primary:hover{filter:brightness(1.1);color:#000;}
.btn-icon{width:34px;height:34px;padding:0;justify-content:center;}
.btn:disabled{opacity:.4;cursor:not-allowed;}
main{flex:1;padding:1.25rem 1.75rem;display:flex;flex-direction:column;gap:1rem;max-width:1280px;margin:0 auto;width:100%;}
.search-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.25rem 1.5rem;box-shadow:var(--shadow);}
.search-title{font-size:1rem;font-weight:700;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem;}
.search-title svg{color:var(--primary);}
.search-row{display:flex;flex-wrap:wrap;gap:.75rem;align-items:flex-end;}
.field{display:flex;flex-direction:column;gap:.3rem;flex:1;min-width:160px;}
.field label{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);font-weight:600;}
.field input,.field select{padding:.5rem .7rem;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface-2);color:var(--text);font-family:var(--font-mono);font-size:.82rem;transition:border-color 160ms ease;outline:none;}
.field input:focus,.field select:focus{border-color:var(--primary);}
.field input::placeholder{color:var(--text-muted);}
.thresholds{display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.75rem;padding-top:.75rem;border-top:1px solid var(--border);}
.thresholds .field{min-width:140px;flex:0 1 160px;}
.results-grid{display:grid;grid-template-columns:1fr 340px;gap:1rem;}
.chart-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.25rem;box-shadow:var(--shadow);}
.chart-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.85rem;}
.chart-card-title{font-size:.85rem;font-weight:700;color:var(--text);}
.chart-wrap{position:relative;width:100%;}
.chart-label-row{display:flex;gap:.75rem;margin-top:.4rem;}
.chart-label{font-size:.68rem;font-family:var(--font-mono);padding:.18rem .45rem;border-radius:4px;display:flex;align-items:center;gap:.3rem;}
.chart-label.real{background:var(--primary-glow);color:var(--primary);}
.chart-label.recon{background:var(--accent-glow);color:var(--accent);}
.chart-label.residual{background:var(--danger-glow);color:var(--danger);}
.sidebar{display:flex;flex-direction:column;gap:1rem;}
.verdict-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.25rem;box-shadow:var(--shadow);}
.verdict-icon{font-size:2.2rem;margin-bottom:.5rem;}
.verdict-type{font-size:1rem;font-weight:700;margin-bottom:.3rem;}
.verdict-type.planet{color:var(--primary);}
.verdict-type.binary{color:var(--danger);}
.verdict-type.noise{color:var(--success);}
.verdict-reason{font-size:.78rem;color:var(--text-muted);line-height:1.5;}
.verdict-divider{border:none;border-top:1px solid var(--border);margin:.85rem 0;}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:.35rem 0;}
.metric-label{font-size:.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;}
.metric-value{font-size:.88rem;font-family:var(--font-mono);font-weight:600;color:var(--text);}
.metric-value.warn{color:var(--accent);}
.metric-value.danger{color:var(--danger);}
.metric-value.ok{color:var(--success);}
.dip-bar-wrap{margin-top:.5rem;}
.dip-bar-label{font-size:.68rem;color:var(--text-muted);margin-bottom:.3rem;display:flex;justify-content:space-between;}
.dip-bar-track{height:8px;background:var(--surface-3);border-radius:99px;overflow:hidden;}
.dip-bar-fill{height:100%;border-radius:99px;transition:width .6s ease;background:var(--primary);}
.history-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.25rem;box-shadow:var(--shadow);}
.history-title{font-size:.8rem;font-weight:700;margin-bottom:.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;}
.history-list{display:flex;flex-direction:column;gap:.4rem;max-height:200px;overflow-y:auto;}
.history-item{display:flex;justify-content:space-between;align-items:center;padding:.4rem .6rem;border-radius:var(--radius-sm);cursor:pointer;border:1px solid transparent;transition:all 160ms ease;}
.history-item:hover{background:var(--surface-2);border-color:var(--border);}
.history-star{font-size:.78rem;font-family:var(--font-mono);color:var(--text);}
.tag{font-size:.62rem;padding:.15rem .4rem;border-radius:4px;font-weight:600;text-transform:uppercase;}
.tag.planet{background:var(--primary-glow);color:var(--primary);}
.tag.binary{background:var(--danger-glow);color:var(--danger);}
.tag.noise{background:var(--success-glow);color:var(--success);}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 1rem;color:var(--text-muted);text-align:center;gap:.75rem;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);box-shadow:var(--shadow);}
.empty-state svg{color:var(--text-faint);opacity:.5;}
.empty-state p{font-size:.85rem;max-width:32ch;}
.loading-state{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 1rem;gap:1rem;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);}
.spinner{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--primary);border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{font-size:.82rem;color:var(--text-muted);font-family:var(--font-mono);}
.error-banner{background:var(--danger-glow);border:1px solid rgba(248,113,113,.3);border-radius:var(--radius-sm);padding:.65rem 1rem;font-size:.8rem;color:var(--danger);display:none;}
footer{text-align:center;padding:.75rem;border-top:1px solid var(--border);font-size:.65rem;color:var(--text-faint);font-family:var(--font-mono);}
@media(max-width:900px){.results-grid{grid-template-columns:1fr;}}
@media(max-width:600px){main{padding:.75rem}}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.fade-up{animation:fadeUp .35s ease both;}
</style>
</head>
<body>
<canvas id="starfield"></canvas>
<div class="app">
<header>
  <div class="logo">
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-label="Logo">
      <circle cx="16" cy="16" r="14.5" stroke="#38bdf8" stroke-width="1.4"/>
      <circle cx="16" cy="16" r="6.5" stroke="#a855f7" stroke-width="1.2" stroke-dasharray="2 2"/>
      <circle cx="16" cy="16" r="2.2" fill="#fbbf24"/>
      <line x1="16" y1="1.5" x2="16" y2="6.5" stroke="#38bdf8" stroke-width="1.4"/>
      <line x1="16" y1="25.5" x2="16" y2="30.5" stroke="#38bdf8" stroke-width="1.4"/>
      <line x1="1.5" y1="16" x2="6.5" y2="16" stroke="#38bdf8" stroke-width="1.4"/>
      <line x1="25.5" y1="16" x2="30.5" y2="16" stroke="#38bdf8" stroke-width="1.4"/>
      <circle cx="25" cy="7" r="1.3" fill="#fbbf24"/>
      <circle cx="7" cy="25" r="0.9" fill="#38bdf8" opacity=".5"/>
    </svg>
    <div>
      <div class="logo-text">AI Anomaly Hunter</div>
      <div class="logo-sub">Conv. Autoencoder &middot; Kepler / TESS &middot; Live Analyse</div>
    </div>
  </div>
  <div class="hdr-right">
    <button class="btn btn-icon" data-theme-toggle title="Theme wechseln">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    </button>
  </div>
</header>
<main>
  <div class="search-card fade-up">
    <div class="search-title">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      Stern analysieren
    </div>
    <div class="search-row">
      <div class="field" style="flex:2;min-width:200px;">
        <label>Stern-ID</label>
        <input id="starId" type="text" placeholder="z.B. KIC 8435766  oder  TIC 394137592" />
      </div>
      <div class="field">
        <label>Mission / Author</label>
        <select id="author">
          <option value="Kepler">Kepler</option>
          <option value="SPOC">TESS (SPOC)</option>
          <option value="TESS-SPOC">TESS-SPOC</option>
          <option value="QLP">TESS (QLP)</option>
        </select>
      </div>
      <div class="field">
        <label>Quarter / Sector (0 = alle)</label>
        <input id="quarter" type="number" value="10" min="0" max="99" />
      </div>
      <button class="btn btn-primary" id="analyzeBtn" style="height:38px;padding:.42rem 1.2rem;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        Analysieren
      </button>
    </div>
    <div class="thresholds">
      <div class="field">
        <label>Planeten-Schwelle</label>
        <input id="threshPlanet" type="number" value="0.001" step="0.0001" min="0.0001" max="0.05" />
      </div>
      <div class="field">
        <label>Binary-Schwelle</label>
        <input id="threshBinary" type="number" value="0.05" step="0.001" min="0.01" max="0.5" />
      </div>
      <div class="field" style="justify-content:flex-end;padding-bottom:2px;">
        <label style="opacity:0">.</label>
        <span style="font-size:.72rem;color:var(--text-muted);">Werte aus check_anomaly.py Defaults</span>
      </div>
    </div>
    <div id="errorBanner" class="error-banner" style="margin-top:.75rem;"></div>
  </div>
  <div id="resultsArea">
    <div class="empty-state fade-up" id="emptyState">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" stroke-dasharray="2 2"/>
        <line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/>
      </svg>
      <p>Gib eine Stern-ID ein und klicke <strong>Analysieren</strong>.<br>Das Modell laedt die NASA-Lichtkurve und klassifiziert den Stern in Echtzeit.</p>
    </div>
  </div>
  <div class="history-card fade-up" id="historyCard" style="display:none;">
    <div class="history-title">Analyse-Verlauf</div>
    <div class="history-list" id="historyList"></div>
  </div>
</main>
<footer>AI Anomaly Hunter &middot; Conv. Autoencoder &middot; TensorFlow / Lightkurve / Flask</footer>
</div>
<script>
(function(){
  const c=document.getElementById('starfield'),ctx=c.getContext('2d');
  let s=[];
  function resize(){c.width=innerWidth;c.height=innerHeight;}
  function init(){s=[];for(let i=0;i<180;i++)s.push({x:Math.random()*c.width,y:Math.random()*c.height,r:Math.random()*.9+.2,a:Math.random(),da:(Math.random()-.5)*.004});}
  function draw(){ctx.clearRect(0,0,c.width,c.height);s.forEach(p=>{p.a+=p.da;if(p.a<0||p.a>1)p.da*=-1;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=`rgba(170,195,255,${p.a*.35})`;ctx.fill();});requestAnimationFrame(draw);}
  resize();init();draw();
  window.addEventListener('resize',()=>{resize();init();});
})();
(function(){
  const btn=document.querySelector('[data-theme-toggle]'),root=document.documentElement;
  let d='dark';root.setAttribute('data-theme',d);
  btn&&btn.addEventListener('click',()=>{
    d=d==='dark'?'light':'dark';root.setAttribute('data-theme',d);
    btn.innerHTML=d==='dark'
      ?'<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
      :'<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  });
})();
let charts={};
const history=[];
document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);
document.getElementById('starId').addEventListener('keydown', e=>{ if(e.key==='Enter') runAnalysis(); });
async function runAnalysis(){
  const starId=document.getElementById('starId').value.trim();
  if(!starId){showError('Bitte eine Stern-ID eingeben.');return;}
  clearError();showLoading(starId);
  document.getElementById('analyzeBtn').disabled=true;
  try {
    const res=await fetch('/api/analyze',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        star_id:starId,author:document.getElementById('author').value,
        quarter:document.getElementById('quarter').value,
        threshold_planet:parseFloat(document.getElementById('threshPlanet').value),
        threshold_binary:parseFloat(document.getElementById('threshBinary').value),
      })
    });
    const data=await res.json();
    if(!res.ok){showError(data.error||'Unbekannter Fehler.');clearLoading();return;}
    renderResults(data);addHistory(data);
  } catch(e){showError('Server nicht erreichbar. Laeuft app.py?');}
  finally{document.getElementById('analyzeBtn').disabled=false;}
}
function showLoading(starId){
  document.getElementById('emptyState')&&document.getElementById('emptyState').remove();
  document.getElementById('resultsArea').innerHTML=`<div class="loading-state fade-up" id="loadingState"><div class="spinner"></div><div class="loading-text">Lade Lichtkurve fuer <strong>${starId}</strong> von NASA...</div><div class="loading-text" style="font-size:.7rem;opacity:.6;">lightkurve &rarr; preprocess &rarr; model.predict()</div></div>`;
}
function clearLoading(){document.getElementById('loadingState')&&document.getElementById('loadingState').remove();}
function showError(msg){const b=document.getElementById('errorBanner');b.textContent=msg;b.style.display='block';}
function clearError(){document.getElementById('errorBanner').style.display='none';}
function renderResults(d){
  Object.values(charts).forEach(c=>c&&c.destroy&&c.destroy());charts={};
  const icons={planet:'&#127758;',binary:'&#11088;',noise:'&#9989;'};
  const dipColor=d.verdict_type==='planet'?'var(--accent)':d.verdict_type==='binary'?'var(--danger)':'var(--success)';
  const mseColor=d.mse>d.threshold_binary*0.05?'warn':'ok';
  const dipPct=(d.max_dip*100).toFixed(3);
  const barPct=Math.min(d.max_dip/d.threshold_binary*70,100).toFixed(1);
  const barColor=d.verdict_type==='binary'?'var(--danger)':d.verdict_type==='planet'?'var(--primary)':'var(--success)';
  document.getElementById('resultsArea').innerHTML=`
  <div class="results-grid fade-up">
    <div style="display:flex;flex-direction:column;gap:1rem;">
      <div class="chart-card">
        <div class="chart-card-header">
          <div class="chart-card-title">Lichtkurve &#8212; ${d.star_id}</div>
          <div class="chart-label-row">
            <span class="chart-label real">&#9679; NASA-Daten</span>
            <span class="chart-label recon">&#9679; KI-Rekonstruktion</span>
          </div>
        </div>
        <div class="chart-wrap" style="height:220px;"><canvas id="lcChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card-header">
          <div class="chart-card-title">Residuum |Real &#8722; Rekonstruktion|</div>
          <div class="chart-label-row">
            <span class="chart-label residual">&#9679; Abweichung</span>
            <span class="chart-label" style="background:rgba(74,222,128,.08);color:var(--success);">&#8212; Planeten-Schwelle</span>
            <span class="chart-label" style="background:rgba(248,113,113,.08);color:var(--danger);">&#8212; Binary-Schwelle</span>
          </div>
        </div>
        <div class="chart-wrap" style="height:140px;"><canvas id="resChart"></canvas></div>
      </div>
    </div>
    <div class="sidebar">
      <div class="verdict-card fade-up">
        <div class="verdict-icon">${icons[d.verdict_type]||'&#128301;'}</div>
        <div class="verdict-type ${d.verdict_type}">${d.verdict}</div>
        <div class="verdict-reason">${d.reason}</div>
        <hr class="verdict-divider">
        <div class="metric-row"><span class="metric-label">Stern-ID</span><span class="metric-value" style="font-size:.75rem;">${d.star_id}</span></div>
        <div class="metric-row"><span class="metric-label">MSE Score</span><span class="metric-value ${mseColor}">${d.mse}</span></div>
        <div class="metric-row"><span class="metric-label">Max. Dip-Tiefe</span><span class="metric-value" style="color:${dipColor};">${d.max_dip_pct}</span></div>
        <div class="metric-row"><span class="metric-label">Datenpunkte</span><span class="metric-value">${d.data_points}</span></div>
        <div class="dip-bar-wrap">
          <div class="dip-bar-label"><span>Dip-Intensitaet</span><span>${dipPct}%</span></div>
          <div class="dip-bar-track"><div class="dip-bar-fill" style="width:${barPct}%;background:${barColor};"></div></div>
        </div>
        <hr class="verdict-divider">
        <div class="metric-row"><span class="metric-label">Planet-Schwelle</span><span class="metric-value" style="font-size:.75rem;">${(d.threshold_planet*100).toFixed(2)}%</span></div>
        <div class="metric-row"><span class="metric-label">Binary-Schwelle</span><span class="metric-value" style="font-size:.75rem;">${(d.threshold_binary*100).toFixed(1)}%</span></div>
      </div>
    </div>
  </div>`;
  requestAnimationFrame(()=>{buildLcChart(d);buildResChart(d);});
}
function buildLcChart(d){
  const ctx=document.getElementById('lcChart').getContext('2d');
  const labels=Array.from({length:d.real.length},(_,i)=>i);
  charts.lc=new Chart(ctx,{
    type:'line',data:{labels,datasets:[
      {label:'NASA-Daten',data:d.real,borderColor:'rgba(56,189,248,.85)',borderWidth:1.2,pointRadius:0,tension:.2,fill:false},
      {label:'KI-Rekonstruktion',data:d.reconstructed,borderColor:'rgba(251,191,36,.85)',borderWidth:1.4,pointRadius:0,tension:.2,fill:false,borderDash:[5,3]},
    ]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,titleFont:{family:'Space Mono',size:10},bodyFont:{family:'Space Mono',size:10}}},
      scales:{
        x:{ticks:{maxTicksLimit:8,font:{size:9,family:'Space Mono'},color:'#706e68'},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{ticks:{font:{size:9,family:'Space Mono'},color:'#706e68'},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}
function buildResChart(d){
  const ctx=document.getElementById('resChart').getContext('2d');
  const labels=Array.from({length:d.residual.length},(_,i)=>i);
  const barColors=d.residual.map(v=>v>d.threshold_binary?'rgba(248,113,113,.8)':v>d.threshold_planet?'rgba(251,191,36,.7)':'rgba(56,189,248,.3)');
  charts.res=new Chart(ctx,{
    type:'bar',data:{labels,datasets:[{label:'Residuum',data:d.residual,backgroundColor:barColors,borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:400},
      plugins:{legend:{display:false},tooltip:{callbacks:{title:l=>`Punkt ${l[0].label}`,label:l=>`Delta Flux: ${l.raw.toFixed(5)}`},titleFont:{family:'Space Mono',size:10},bodyFont:{family:'Space Mono',size:10}}},
      scales:{x:{display:false},y:{ticks:{maxTicksLimit:4,font:{size:9,family:'Space Mono'},color:'#706e68'},grid:{color:'rgba(255,255,255,0.04)'}}}
    }
  });
}
function addHistory(d){
  history.unshift(d);
  const card=document.getElementById('historyCard');
  const list=document.getElementById('historyList');
  card.style.display='block';
  list.innerHTML=history.slice(0,20).map((h,i)=>`
    <div class="history-item" onclick="replayHistory(${i})">
      <span class="history-star">${h.star_id}</span>
      <div style="display:flex;gap:.4rem;align-items:center;">
        <span style="font-size:.7rem;color:var(--text-muted);font-family:var(--font-mono);">MSE ${h.mse}</span>
        <span class="tag ${h.verdict_type}">${h.verdict_type}</span>
      </div>
    </div>`).join('');
}
function replayHistory(i){renderResults(history[i]);}
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n{'='*55}")
    print(f"  AI Anomaly Hunter --- Web Visualizer")
    print(f"  http://localhost:{port}")
    print(f"  Modell erwartet: {MODEL_PATH.resolve()}")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
