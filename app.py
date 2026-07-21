# -*- coding: utf-8 -*-
"""
Servico de PAPER TRADING — HA-2h 250/30/25 — ETH + BTC — para hospedar no Render.

Roda a MESMA estrategia em DOIS ativos. Nao e' para diversificar: e' para
terminar o teste em ~5 meses em vez de 7,5. A correlacao medida entre os
retornos diarios das duas (rho=0,346 no cofre) da um ganho efetivo de amostra
de 1,49x. E o BTC ainda serve de prova independente: o edge la e' +0,0700%/trade
(t=5,52), praticamente igual ao do ETH — ou seja, nao e' artefato de um ativo so.

SINAIS INTERNOS (threads independentes):
  13s -> heartbeat  : prova de vida, contador
  21s -> preco      : le bid/ask dos 2 ativos e gerencia SL / TP / trailing
  30s -> estado     : grava em disco + checa virada do candle de 2h

SINAL EXTERNO:
  GET/POST /ping|/health -> keep-alive de fora (e o que REALMENTE impede o
                            Render free de dormir; os timers internos NAO)
  POST     /sinal        -> recebe JSON de fora (so registra, nunca opera)

Preco: endpoint PUBLICO do MEXC (so leitura de mercado, sem chave, sem conta).
Nao envia ordem para lugar nenhum. Trocado da Binance porque ela devolve HTTP
418 para IP de datacenter (bloqueia o Render).
"""
import os
import json
import time
import threading
from datetime import datetime, timezone

import numpy as np
import requests
import websocket
from flask import Flask, jsonify, request

from estrategia import sinal_atual, TF_MS
import estado_remoto
from paper import PaperTrader, FEE_LADO, N_MIN, T_MIN, CENARIOS, ATIVOS, REF_MEDIA, REF_WR

APP_INICIO = time.time()
UA = {"User-Agent": "Mozilla/5.0"}
TICKER = "https://contract.mexc.com/api/v1/contract/ticker?symbol={s}"
KLINE = "https://contract.mexc.com/api/v1/contract/kline/{s}?interval=Min60&start={t}"

app = Flask(__name__)
TRADERS = {a: PaperTrader(a) for a in ATIVOS}
log_recente = []
_loglock = threading.Lock()


def log(msg):
    linha = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z | {msg}"
    print(linha, flush=True)
    with _loglock:
        log_recente.append(linha)
        del log_recente[:-200]


# --------------------------------------------------- dados de mercado
def bid_ask(simbolo):
    """(bid, ask) do perp. Ordem a mercado paga exatamente isso."""
    r = requests.get(TICKER.format(s=simbolo), timeout=8, headers=UA)
    r.raise_for_status()
    d = r.json()["data"]
    return float(d["bid1"]), float(d["ask1"])


def candles_2h(simbolo):
    """Candles de 2h FECHADOS, alinhados a 00/02/04... UTC.

    O MEXC nao tem candle de 2h (so Min60 e Hour4), entao agrego dois de 1h —
    exatamente o que o backtest faz (slot = ts // 2h, offset 0).
    """
    inicio = int(time.time()) - 400 * 3600
    r = requests.get(KLINE.format(s=simbolo, t=inicio), timeout=12, headers=UA)
    r.raise_for_status()
    k = r.json()["data"]
    tsh = [int(x) for x in k["time"]]
    oh = [float(x) for x in k["open"]]; hh = [float(x) for x in k["high"]]
    lh = [float(x) for x in k["low"]]; ch = [float(x) for x in k["close"]]
    balde = {}
    for i, t in enumerate(tsh):
        b = (t // 7200) * 7200
        if b not in balde:
            balde[b] = [oh[i], hh[i], lh[i], ch[i], t, t]
        else:
            v = balde[b]
            v[1] = max(v[1], hh[i]); v[2] = min(v[2], lh[i])
            if t >= v[4]:
                v[3] = ch[i]; v[4] = t          # close = candle mais novo
            if t <= v[5]:
                v[0] = oh[i]; v[5] = t          # open  = candle mais antigo
    aberto = (int(time.time()) // 7200) * 7200
    ts, o, h, l, c = [], [], [], [], []
    for b in sorted(balde):
        if b >= aberto:
            continue                            # so candles FECHADOS
        v = balde[b]
        ts.append(b * 1000); o.append(v[0]); h.append(v[1]); l.append(v[2]); c.append(v[3])
    return ts, o, h, l, c


# --------------------------------------------------- WEBSOCKET (feed principal)
WS_URL = "wss://contract.mexc.com/edge"
POR_SIMBOLO = {pt.simbolo: pt for pt in TRADERS.values()}
ws_estado = {"conectado": False, "ultima_msg": 0.0, "reconexoes": 0, "ticks": 0,
             "pongs": 0, "conectado_desde": 0.0}


def _ws_ping(ws):
    """O MEXC exige ping no protocolo DELE (JSON), nao o ping do WebSocket.

    MEDIDO: sem este ping a conexao morre em 67s (opcode=8, close do servidor);
    com ele, 240s sem cair e o servidor respondendo pong a cada ~16s.
    """
    while True:
        time.sleep(15)
        if not ws_estado["conectado"]:
            return
        try:
            ws.send(json.dumps({"method": "ping"}))
        except Exception:
            return


def _ws_open(ws):
    for s in POR_SIMBOLO:
        ws.send(json.dumps({"method": "sub.deal", "param": {"symbol": s}}))
        ws.send(json.dumps({"method": "sub.ticker", "param": {"symbol": s}}))
    ws_estado["conectado"] = True
    ws_estado["ultima_msg"] = time.time()
    ws_estado["conectado_desde"] = time.time()
    threading.Thread(target=_ws_ping, args=(ws,), daemon=True).start()
    log(f"[ws] conectado | deal+ticker de {', '.join(POR_SIMBOLO)} | ping JSON a cada 15s")


def _ws_msg(ws, raw):
    try:
        m = json.loads(raw)
        ch = m.get("channel", "")
        if ch == "pong":                       # resposta ao nosso ping JSON
            ws_estado["pongs"] += 1
            ws_estado["ultima_msg"] = time.time()
            return
        if ch.startswith("rs."):
            return
        pt = POR_SIMBOLO.get(m.get("symbol"))
        if pt is None:
            return
        ws_estado["ultima_msg"] = time.time()
        d = m.get("data")
        if ch == "push.deal":
            # cada msg traz 1+ negocios; alimenta todos (gatilho fino, ~0,42s)
            for neg in (d if isinstance(d, list) else [d]):
                ws_estado["ticks"] += 1
                t = pt.on_tick(neg["p"], int(neg.get("t", time.time() * 1000)))
                if t:
                    log(f"[ws] {pt.ativo} FECHOU {'LONG' if t['dir']==1 else 'SHORT'} "
                        f"{t['motivo']} | {t['entry']:.2f} -> {t['saida']:.2f} | "
                        f"bruto {t['bruto']*100:+.3f}% | total={total_trades()}/{N_MIN}")
                    pt.salvar(forcar_gist=True)   # trade fechou -> grava no gist ja
        elif ch == "push.ticker":
            pt.atualiza_cotacao(d["bid1"], d["ask1"], d.get("fundingRate"))
    except Exception as e:
        log(f"[ws] erro ao processar: {type(e).__name__}: {e}")


def _ws_close(ws, *a):
    ws_estado["conectado"] = False


def _ws_err(ws, e):
    # opcode=8 e' o frame de CLOSE do servidor: queda normal, nao erro de codigo.
    if "opcode=8" in str(e):
        return
    log(f"[ws] erro: {type(e).__name__}: {e}")


def ws_loop():
    """Feed principal. Reconecta sozinho — 9 meses sem cair nao existe."""
    # sem isso, um connect que o servidor ignora (pacote descartado) pendura a
    # thread PARA SEMPRE — run_forever nao tem timeout de conexao proprio
    websocket.setdefaulttimeout(15)
    while True:
        try:
            ws = websocket.WebSocketApp(WS_URL, on_open=_ws_open, on_message=_ws_msg,
                                        on_close=_ws_close, on_error=_ws_err)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log(f"[ws] caiu: {type(e).__name__}: {e}")
        viveu = (time.time() - ws_estado["conectado_desde"]) if ws_estado["conectado_desde"] else 0
        ws_estado["conectado"] = False
        ws_estado["reconexoes"] += 1
        log(f"[ws] caiu apos {viveu:.0f}s vivo | reconectando em 5s "
            f"(reconexao #{ws_estado['reconexoes']}) | a rede de 21s cobre o intervalo")
        time.sleep(5)


def ws_vivo():
    """WebSocket entregou preco nos ultimos 60s?"""
    return ws_estado["conectado"] and (time.time() - ws_estado["ultima_msg"]) < 60


# --------------------------------------------------- sinais internos
def sinal_13s():
    """Heartbeat: prova de vida."""
    while True:
        try:
            for pt in TRADERS.values():
                pt.contadores["hb13"] += 1
            n = total_trades()
            if TRADERS["ETH"].contadores["hb13"] % 100 == 0:
                log(f"[13s] heartbeat | trades ETH+BTC = {n}/{N_MIN}")
        except Exception as e:
            log(f"[13s] erro: {e}")
        time.sleep(13)


def sinal_21s():
    """REDE DE SEGURANCA: o WebSocket e' o feed principal (tick a ~0,42s).
    Este timer so assume se o WS cair — assim a posicao nunca fica sem gestao
    de SL/TP/trailing. Se o WS esta vivo, aqui so conta o tick e sai."""
    while True:
        if ws_vivo():
            for pt in TRADERS.values():
                pt.contadores["preco21"] += 1
            time.sleep(21)
            continue
        log("[21s] WebSocket mudo -> assumindo o feed por REST")
        for a, pt in TRADERS.items():
            try:
                b, k = bid_ask(pt.simbolo)
                pt.contadores["preco21"] += 1
                t = pt.on_preco(b, k, int(time.time() * 1000))
                if t:
                    log(f"[21s/rede] {a} FECHOU {'LONG' if t['dir']==1 else 'SHORT'} "
                        f"{t['motivo']} | {t['entry']:.2f} -> {t['saida']:.2f} "
                        f"| bruto {t['bruto']*100:+.3f}% | total={total_trades()}/{N_MIN}")
                    pt.salvar(forcar_gist=True)   # trade fechou -> grava no gist ja
            except Exception as e:
                log(f"[21s] {a} erro: {e}")
        time.sleep(21)


def sinal_30s():
    """Estado: grava em disco e checa a virada do candle de 2h."""
    while True:
        for a, pt in TRADERS.items():
            try:
                pt.contadores["estado30"] += 1
                agora = int(time.time() * 1000)
                atual = (agora // TF_MS) * TF_MS
                if pt.ultimo_candle_ts is None or atual > pt.ultimo_candle_ts:
                    ts, o, h, l, c = candles_2h(pt.simbolo)
                    if len(c) >= 60:
                        d = sinal_atual(o, h, l, c, ts)
                        b, k = bid_ask(pt.simbolo)
                        fech = pt.on_candle_fechado(d, b, k, agora)
                        if fech:
                            log(f"[30s] {a} FECHOU por reversao | bruto {fech['bruto']*100:+.3f}%")
                        log(f"[30s] {a} candle 2h virou | sinal="
                            f"{ {1:'LONG', -1:'SHORT', 0:'nada'}[d] } | bid={b:.2f} ask={k:.2f}")
                    else:
                        log(f"[30s] {a} historico curto ({len(c)}), aguardando")
                pt.salvar()
            except Exception as e:
                log(f"[30s] {a} erro: {e}")
        time.sleep(30)


# --------------------------------------------------- agregacao
def total_trades():
    return sum(len(pt.trades) for pt in TRADERS.values())


def brutos_pool():
    """Retornos BRUTOS de todos os ativos juntos, em ordem cronologica."""
    tudo = []
    for pt in TRADERS.values():
        with pt.lock:
            tudo += [(t["fechamento"], t["bruto"]) for t in pt.trades]
    tudo.sort()
    return np.array([b for _, b in tudo], dtype=float)


def veredito_geral():
    """Resultado combinado ETH+BTC sob cada corretora."""
    b = brutos_pool()
    n = len(b)
    out = {"n": n, "n_min": N_MIN, "falta": max(N_MIN - n, 0), "cenarios": {}}
    dias = max((time.time() - APP_INICIO) / 86400, 1e-9)
    for nome, fee in CENARIOS.items():
        if n < 2:
            out["cenarios"][nome] = {"veredito": "coletando"}
            continue
        r = b - 2 * fee
        mu = float(r.mean()); sd = float(r.std(ddof=1))
        t = mu / (sd / np.sqrt(n)) if sd > 0 else 0.0
        eq = np.cumprod(1 + r)
        eq2 = np.cumprod(1 + np.maximum(2 * r, -0.999))
        dd2 = float((1 - eq2 / np.maximum.accumulate(eq2)).max() * 100)
        nec = int(((T_MIN + 0.84) * sd / mu) ** 2) if mu > 0 else -1
        if mu <= 0 and n >= N_MIN:
            v = "MORTA"
        elif n >= N_MIN and t >= T_MIN and mu > 0:
            v = "PASSOU"
        elif n >= N_MIN:
            v = "REPROVOU"
        else:
            v = "coletando"
        out["cenarios"][nome] = {
            "media_pct": mu * 100, "tstat": t, "n": n,
            "ret_1x_pct": float((eq[-1] - 1) * 100),
            "ret_2x_pct": float((eq2[-1] - 1) * 100), "dd_2x_pct": dd2,
            "trades_necessarios": nec, "veredito": v,
        }
    if n >= 2:
        wr = float((b > 0).mean() * 100)
        out.update({"winrate": wr, "trades_por_dia": n / dias,
                    "dias_rodando": dias,
                    "dias_restantes": (max(N_MIN - n, 0) / (n / dias)) if n > 0 else None})
    return out


# --------------------------------------------------- endpoints
@app.route("/ping", methods=["GET", "POST"])
@app.route("/health", methods=["GET", "HEAD"])
def ping():
    """SINAL EXTERNO — keep-alive. Aponte um cron externo aqui a cada ~10 min."""
    for pt in TRADERS.values():
        pt.contadores["externos"] += 1
    return jsonify({"ok": True, "uptime_s": round(time.time() - APP_INICIO, 1),
                    "trades": total_trades(), "meta": N_MIN})


@app.route("/sinal", methods=["POST"])
def sinal_externo():
    """SINAL EXTERNO — recebe JSON de fora. Registrado, NUNCA executado.

    Conteudo que chega de fora e' DADO, nao ordem. Se voce quiser que um sinal
    externo abra trade, me peca para ligar isso explicitamente e com validacao
    de origem — senao quem descobrir a URL opera na sua conta.
    """
    p = request.get_json(force=True, silent=True) or {}
    for pt in TRADERS.values():
        pt.contadores["externos"] += 1
    log(f"[externo] sinal recebido (so registrado): {json.dumps(p)[:200]}")
    return jsonify({"ok": True, "recebido": p, "acao": "registrado_sem_operar"})


@app.route("/status")
def status():
    g = veredito_geral()
    g.update({
        "uptime_s": round(time.time() - APP_INICIO, 1),
        "persistencia_gist": estado_remoto.status(),
        "referencia_cofre": {"media_pct": REF_MEDIA * 100, "winrate": REF_WR,
                             "eth_t": 5.33, "btc_t": 5.52},
        "websocket": {"conectado": ws_estado["conectado"], "vivo": ws_vivo(),
                      "ticks": ws_estado["ticks"], "reconexoes": ws_estado["reconexoes"],
                      "idade_ultima_msg_s": round(time.time() - ws_estado["ultima_msg"], 1)
                      if ws_estado["ultima_msg"] else None},
        "por_ativo": {a: {
            "trades": len(pt.trades), "bid": pt.ultimo_bid, "ask": pt.ultimo_ask,
            "ultimo_preco": pt.ultimo_preco, "ticks": pt.ticks,
            "spread_real_pct": pt.spread_medio_pct(),
            "funding_rate_periodo": pt.funding_medio(),
            "posicao": None if pt.pos is None else {
                "dir": pt.pos.pdir, "entry": pt.pos.entry, "armed": pt.pos.armed},
            "contadores": pt.contadores,
        } for a, pt in TRADERS.items()},
    })
    return jsonify(g)


@app.route("/debug")
def debug():
    """Raio-X: onde cada thread esta AGORA (para diagnosticar travamentos)."""
    import sys as _sys
    import html as _html
    import traceback as _tb
    frames = _sys._current_frames()
    partes = [f"uptime={time.time()-APP_INICIO:.0f}s | "
              f"threads vivas={threading.active_count()}"]
    for th in threading.enumerate():
        f = frames.get(th.ident)
        pilha = "".join(_tb.format_stack(f)) if f else "(sem frame)"
        partes.append(f"===== {th.name} (daemon={th.daemon})\n{pilha}")
    return "<pre>" + _html.escape("\n\n".join(partes)) + "</pre>"


@app.route("/log")
def ver_log():
    with _loglock:
        return "<pre>" + "\n".join(log_recente[-100:]) + "</pre>"


@app.route("/ativos")
def ativos_json():
    """Cada ativo LIDO SOZINHO — winrate/edge/DD proprios, sem misturar.
    Combinar ETH+BTC numa conta so infla o DD (perda de um conta pro outro);
    aqui cada um aparece com o risco real dele, a qualquer momento."""
    import numpy as _np
    out = {}
    for a, pt in TRADERS.items():
        r = _np.array([t["bruto"] for t in pt.trades], dtype=float)
        if len(r) == 0:
            out[a] = {"trades": 0}
            continue
        eq1 = _np.cumprod(1 + r)
        eq2 = _np.cumprod(1 + _np.maximum(2 * r, -0.999))
        dd1 = float((1 - eq1 / _np.maximum.accumulate(eq1)).max() * 100)
        dd2 = float((1 - eq2 / _np.maximum.accumulate(eq2)).max() * 100)
        mu = float(r.mean()); sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
        out[a] = {
            "trades": len(r), "winrate": float((r > 0).mean() * 100),
            "edge_pct": mu * 100, "desvio_pct": sd * 100,
            "tstat": (mu / (sd / (len(r) ** 0.5))) if sd > 0 else 0.0,
            "pior_trade_pct": float(r.min() * 100),
            "melhor_trade_pct": float(r.max() * 100),
            "ret_1x_pct": float((eq1[-1] - 1) * 100), "dd_1x_pct": dd1,
            "ret_2x_pct": float((eq2[-1] - 1) * 100), "dd_2x_pct": dd2,
            "trades_lista": [{"t": t["fechamento"], "dir": t["dir"],
                              "motivo": t["motivo"], "bruto_pct": t["bruto"] * 100}
                             for t in pt.trades],
        }
    return jsonify(out)


@app.route("/")
def home():
    g = veredito_geral()
    n = g["n"]
    dr = g.get("dias_restantes")
    falta = f"{dr:.0f} dias" if dr else "calculando..."
    linhas = []
    for nome, v in g["cenarios"].items():
        if "media_pct" not in v:
            linhas.append(f'<tr><td>{nome}</td><td colspan="6" style="color:#888">coletando...</td></tr>')
            continue
        cor = {"PASSOU": "#0a0", "MORTA": "#c00", "REPROVOU": "#c00", "coletando": "#888"}[v["veredito"]]
        nec = v["trades_necessarios"]
        linhas.append(
            f'<tr><td><b>{nome}</b></td><td align="right">{v["media_pct"]:+.4f}%</td>'
            f'<td align="right">{v["tstat"]:.2f}</td><td align="right">{v["ret_1x_pct"]:+.1f}%</td>'
            f'<td align="right">{v["ret_2x_pct"]:+.1f}%</td><td align="right">{v["dd_2x_pct"]:.1f}%</td>'
            f'<td style="color:{cor};font-weight:700">{v["veredito"]}</td></tr>')
    ativos = []
    for a, pt in TRADERS.items():
        p = "nenhuma" if pt.pos is None else (("LONG" if pt.pos.pdir == 1 else "SHORT") +
                                              f" @ {pt.pos.entry:.2f}" +
                                              (" [armado]" if pt.pos.armed else ""))
        sp = pt.spread_medio_pct()
        fr = pt.funding_medio()
        # estatisticas SEPARADAS por ativo — um BTC ruim nao pode se esconder
        # atras de um ETH bom (nem o contrario)
        rs = [t["bruto"] for t in pt.trades]
        if rs:
            wr = 100.0 * sum(1 for r in rs if r > 0) / len(rs)
            med = 100.0 * sum(rs) / len(rs)
            pior = 100.0 * min(rs)
            arr = np.array(rs, dtype=float)
            eq2 = np.cumprod(1 + np.maximum(2 * arr, -0.999))
            dd2 = float((1 - eq2 / np.maximum.accumulate(eq2)).max() * 100)  # DD SO deste ativo
            est = (f"<td align='right'>{wr:.0f}%</td>"
                   f"<td align='right'>{med:+.4f}%</td>"
                   f"<td align='right'>{pior:+.2f}%</td>"
                   f"<td align='right'>{dd2:.1f}%</td>")
        else:
            est = "<td align='right'>—</td>" * 4
        ativos.append(f"<tr><td><b>{a}</b></td><td>{len(pt.trades)} trades</td>"
                      f"{est}"
                      f"<td>{pt.ultimo_bid} / {pt.ultimo_ask}</td><td>{p}</td>"
                      f"<td>{(f'{sp:.5f}%' if sp else '—')}</td>"
                      f"<td>{(f'{fr*100:+.4f}%' if fr is not None else '—')}</td>"
                      f"<td>{pt.ticks:,}</td></tr>")
    wsv = ws_vivo()
    ws_html = (f'<span style="color:{"#0a0" if wsv else "#c00"};font-weight:700">'
               f'{"VIVO" if wsv else "MUDO (rede de 21s assumiu)"}</span> '
               f'&nbsp;{ws_estado["ticks"]:,} ticks &nbsp;|&nbsp; '
               f'{ws_estado["reconexoes"]} reconexoes')
    return f"""<html><head><meta charset="utf-8"><title>Paper HA-2h ETH+BTC</title>
<meta http-equiv="refresh" content="30"></head>
<body style="font-family:system-ui;max-width:860px;margin:40px auto;line-height:1.6">
<h2>Paper Trading — HA-2h 250/30/25 — ETH + BTC</h2>
<p><b>{n} / {N_MIN}</b> trades &nbsp;|&nbsp; faltam ~{falta} &nbsp;|&nbsp;
winrate {g.get('winrate', 0):.1f}% <span style="color:#888">(cofre: {REF_WR}%)</span>
&nbsp;|&nbsp; {g.get('trades_por_dia', 0):.1f} trades/dia</p>

<p style="font-size:14px">WebSocket (feed principal, tick a ~0,42s): {ws_html}</p>

<h3>Por ativo</h3>
<table cellpadding="6" style="border-collapse:collapse;font-size:14px">
<tr style="background:#f0f0f0"><th align="left">ativo</th><th>trades</th>
<th>winrate</th><th>media/trade</th><th>pior trade</th><th>DD 2x (so deste ativo)</th><th>bid/ask</th>
<th>posicao</th><th>spread real</th><th>funding/periodo</th><th>ticks</th></tr>
{''.join(ativos)}
</table>

<h3 style="margin-top:26px">Todas as corretoras, do mesmo teste</h3>
<p style="color:#888;font-size:13px">O sinal e' identico em todas — so a taxa muda. Guardo o
retorno bruto e aplico cada taxa em paralelo: este teste unico responde por todas,
sem precisar apostar 5 meses numa so.</p>
<table cellpadding="6" style="border-collapse:collapse;font-size:14px">
<tr style="background:#f0f0f0"><th align="left">corretora</th><th>edge/trade</th><th>t</th>
<th>1x</th><th>2x</th><th>DD 2x</th><th>veredito</th></tr>
{''.join(linhas)}
</table>
<p style="color:#888;font-size:13px">Criterio travado: <b>n &ge; {N_MIN} e t &ge; {T_MIN}</b>
(uni-caudal 95%, 81% de poder). Referencia do cofre: {REF_MEDIA*100:+.4f}%/trade
(ETH t=5,33 | BTC t=5,52).<br>
Gatilho pelo preco NEGOCIADO via WebSocket (~0,42s — 2,4x mais fino que o backtest de 1s);
preenchimento no lado real do livro (entra no ask, sai no bid), entao o spread verdadeiro
ja esta no numero. <b>Ainda fora da conta:</b> funding (medido acima, mas nao descontado —
na Lighter o periodo e' 1h) e a latencia de 300ms da conta Standard.
Dinheiro nenhum envolvido.</p>
<p><a href="/status">/status</a> &middot; <a href="/log">/log</a> &middot; <a href="/ping">/ping</a></p>
</body></html>"""


# As threads precisam nascer NO PROCESSO QUE ATENDE O SITE. O gunicorn importa
# o app num processo-mestre e faz FORK para o worker — e fork NAO copia threads:
# elas nasciam no mestre e o worker ficava so com uma foto congelada dos
# contadores (visto no /debug do Render: 'threads vivas=3', nenhuma nossa).
# Por isso: inicio preguicoso, por PID — cada processo (re)inicia as suas.
_ini_lock = threading.Lock()
_ini_pid = None


def garantir_iniciado():
    global _ini_pid
    if _ini_pid == os.getpid():
        return
    with _ini_lock:
        if _ini_pid == os.getpid():
            return
        threading.Thread(target=ws_loop, name="websocket", daemon=True).start()
        for fn in (sinal_13s, sinal_21s, sinal_30s):
            threading.Thread(target=fn, name=fn.__name__, daemon=True).start()
        _ini_pid = os.getpid()
        log(f"WebSocket + sinais 13s/21s/30s iniciados no processo {os.getpid()} "
            f"| ativos: {', '.join(TRADERS)} | meta {N_MIN} trades")


@app.before_request
def _garante_threads():
    garantir_iniciado()

if __name__ == "__main__":
    garantir_iniciado()          # local (python app.py): sem fork, inicia ja
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
