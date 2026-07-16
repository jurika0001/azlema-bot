# -*- coding: utf-8 -*-
"""
Servico de PAPER TRADING — HA-2h 250/30/25 — para hospedar no Render.

SINAIS INTERNOS (threads independentes):
  13s -> heartbeat  : prova de vida, contador, mantem o processo quente
  21s -> preco      : le o preco do ETH e gerencia SL / TP / trailing
  30s -> estado     : grava em disco + checa virada do candle de 2h (nova entrada)

SINAL EXTERNO:
  GET/POST /ping    -> keep-alive vindo de fora (e o que REALMENTE impede o
                       Render free de dormir; os timers internos NAO impedem)
  POST     /sinal   -> recebe um sinal externo em JSON

Preco: endpoint PUBLICO da Binance (somente leitura de mercado, sem chave,
sem conta, sem API de trading). Nao executa ordem em lugar nenhum.
"""
import os
import json
import time
import threading
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request

from estrategia import sinal_atual, TF_MS
from paper import PaperTrader, FEE_LADO, SPREAD_RT, N_MIN, T_MIN

APP_INICIO = time.time()
PRECO_URL = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
KLINES_URL = ("https://api.binance.com/api/v3/klines"
              "?symbol=ETHUSDT&interval=2h&limit=400")

app = Flask(__name__)
pt = PaperTrader()
log_recente = []
_loglock = threading.Lock()


def log(msg):
    linha = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z | {msg}"
    print(linha, flush=True)
    with _loglock:
        log_recente.append(linha)
        del log_recente[:-200]


# --------------------------------------------------- dados de mercado
def preco_agora():
    r = requests.get(PRECO_URL, timeout=8)
    r.raise_for_status()
    return float(r.json()["price"])


def candles_2h():
    """Candles de 2h FECHADOS (descarta o ultimo, que ainda esta aberto)."""
    r = requests.get(KLINES_URL, timeout=12)
    r.raise_for_status()
    k = r.json()[:-1]
    ts = [int(x[0]) for x in k]
    o = [float(x[1]) for x in k]
    h = [float(x[2]) for x in k]
    l = [float(x[3]) for x in k]
    c = [float(x[4]) for x in k]
    return ts, o, h, l, c


# --------------------------------------------------- sinais internos
def sinal_13s():
    """Heartbeat: prova de vida."""
    while True:
        try:
            pt.contadores["hb13"] += 1
            if pt.contadores["hb13"] % 100 == 0:
                log(f"[13s] heartbeat #{pt.contadores['hb13']} | trades={len(pt.trades)}")
        except Exception as e:
            log(f"[13s] erro: {e}")
        time.sleep(13)


def sinal_21s():
    """Preco: gerencia a posicao aberta (SL / TP / trailing)."""
    while True:
        try:
            p = preco_agora()
            pt.contadores["preco21"] += 1
            t = pt.on_preco(p, int(time.time() * 1000))
            if t:
                log(f"[21s] FECHOU {'LONG' if t['dir']==1 else 'SHORT'} "
                    f"{t['motivo']} | entry {t['entry']:.2f} -> saida {t['saida']:.2f} "
                    f"| liquido {t['ret']*100:+.3f}% | total={len(pt.trades)}")
                pt.salvar()
        except Exception as e:
            log(f"[21s] erro: {e}")
        time.sleep(21)


def sinal_30s():
    """Estado: grava em disco e checa a virada do candle de 2h."""
    while True:
        try:
            pt.contadores["estado30"] += 1
            agora_ms = int(time.time() * 1000)
            candle_atual = (agora_ms // TF_MS) * TF_MS
            if pt.ultimo_candle_ts is None or candle_atual > pt.ultimo_candle_ts:
                ts, o, h, l, c = candles_2h()
                if len(c) >= 60:
                    d = sinal_atual(o, h, l, c, ts)
                    p = preco_agora()
                    fechada = pt.on_candle_fechado(d, p, agora_ms)
                    if fechada:
                        log(f"[30s] FECHOU por reversao | liquido {fechada['ret']*100:+.3f}%")
                    nome = {1: "LONG", -1: "SHORT", 0: "nada"}[d]
                    log(f"[30s] candle 2h virou | sinal={nome} | preco={p:.2f}")
            pt.salvar()
        except Exception as e:
            log(f"[30s] erro: {e}")
        time.sleep(30)


# --------------------------------------------------- endpoints
@app.route("/ping", methods=["GET", "POST"])
def ping():
    """SINAL EXTERNO — keep-alive. Aponte um cron externo aqui a cada ~10 min."""
    pt.contadores["externos"] += 1
    return jsonify({"ok": True, "pong": pt.contadores["externos"],
                    "uptime_s": round(time.time() - APP_INICIO, 1),
                    "trades": len(pt.trades)})


@app.route("/sinal", methods=["POST"])
def sinal_externo():
    """SINAL EXTERNO — recebe JSON de fora. Registrado, nunca executado as cegas.

    Nao opera nada com base nisso: conteudo que chega de fora e DADO, nao ordem.
    Se voce quiser que um sinal externo abra trade, me peca para ligar isso
    explicitamente e com validacao de origem.
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}
    pt.contadores["externos"] += 1
    log(f"[externo] sinal recebido (apenas registrado): {json.dumps(payload)[:200]}")
    return jsonify({"ok": True, "recebido": payload, "acao": "registrado_sem_operar"})


@app.route("/status")
def status():
    s = pt.stats()
    s.update({
        "uptime_s": round(time.time() - APP_INICIO, 1),
        "contadores": pt.contadores,
        "ultimo_preco": pt.ultimo_preco,
        "posicao_aberta": None if pt.pos is None else {
            "dir": pt.pos.pdir, "entry": pt.pos.entry,
            "armed": pt.pos.armed, "peak": pt.pos.peak},
        "custos": {"fee_por_lado_pct": FEE_LADO * 100,
                   "spread_ida_volta_pct": SPREAD_RT * 100},
    })
    return jsonify(s)


@app.route("/log")
def ver_log():
    with _loglock:
        return "<pre>" + "\n".join(log_recente[-100:]) + "</pre>"


@app.route("/")
def home():
    s = pt.stats()
    cor = {"PASSOU": "#0a0", "REPROVOU": "#c00", "coletando": "#888"}[s["veredito"]]
    falta = s.get("dias_restantes")
    falta_txt = f"{falta:.0f} dias" if falta else "calculando..."
    return f"""<html><head><meta charset="utf-8"><title>Paper HA-2h</title>
<meta http-equiv="refresh" content="30"></head>
<body style="font-family:system-ui;max-width:760px;margin:40px auto;line-height:1.6">
<h2>Paper Trading — HA-2h 250/30/25 (ETH, 1x)</h2>
<p><b>Veredito:</b> <span style="color:{cor};font-weight:700">{s['veredito']}</span>
&nbsp;|&nbsp; trades: <b>{s['n']}</b> / {N_MIN} &nbsp;|&nbsp; faltam ~{falta_txt}</p>
<table cellpadding="6" style="border-collapse:collapse">
<tr><td>media/trade</td><td><b>{s.get('media_pct',0):+.4f}%</b>
    <span style="color:#888">(cofre: {s['ref_media_pct']:+.4f}%)</span></td></tr>
<tr><td>desvio/trade</td><td>{s.get('desvio_pct',0):.4f}%
    <span style="color:#888">(cofre: {s['ref_desvio_pct']:.4f}%)</span></td></tr>
<tr><td>t-stat</td><td><b>{s.get('tstat',0):.2f}</b> (precisa &ge; {T_MIN})</td></tr>
<tr><td>winrate</td><td>{s.get('winrate',0):.1f}%</td></tr>
<tr><td>retorno 1x</td><td>{s.get('ret_total_1x_pct',0):+.2f}% (DD {s.get('dd_1x_pct',0):.1f}%)</td></tr>
<tr><td>retorno 2x</td><td>{s.get('ret_total_2x_pct',0):+.2f}% (DD {s.get('dd_2x_pct',0):.1f}%)</td></tr>
<tr><td>rodando ha</td><td>{s.get('dias_rodando',0):.2f} dias ({s.get('trades_por_dia',0):.2f} trades/dia)</td></tr>
<tr><td>preco ETH</td><td>{pt.ultimo_preco}</td></tr>
<tr><td>posicao</td><td>{'nenhuma' if pt.pos is None else ('LONG' if pt.pos.pdir==1 else 'SHORT')+f" @ {pt.pos.entry:.2f}"}</td></tr>
<tr><td>sinais internos</td><td>13s: {pt.contadores['hb13']} | 21s: {pt.contadores['preco21']} | 30s: {pt.contadores['estado30']}</td></tr>
<tr><td>sinais externos</td><td>{pt.contadores['externos']}</td></tr>
</table>
<p style="color:#888;font-size:13px">Custos aplicados: taxa {FEE_LADO*100:.3f}%/lado +
spread {SPREAD_RT*100:.3f}% ida-volta. Dinheiro nenhum envolvido.</p>
<p><a href="/status">/status</a> &middot; <a href="/log">/log</a> &middot; <a href="/ping">/ping</a></p>
</body></html>"""


def iniciar_sinais():
    for fn, nome in ((sinal_13s, "13s"), (sinal_21s, "21s"), (sinal_30s, "30s")):
        threading.Thread(target=fn, name=nome, daemon=True).start()
    log("sinais internos 13s / 21s / 30s iniciados")


iniciar_sinais()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
