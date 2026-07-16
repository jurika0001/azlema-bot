# -*- coding: utf-8 -*-
"""
BACKTEST + VALIDACAO DO PORTE — HA-2h 250/30/25

Duas funcoes:
  1) VALIDA que o estrategia.py desta pasta (usado pelo paper trading) gera
     EXATAMENTE os mesmos sinais que o motor original do projeto. Se divergir,
     o paper trading nao esta testando a mesma estrategia — e nao vale nada.
  2) Roda o backtest no cofre 2017-20 (dados virgens) e reporta 1x/2x/3x.

Uso:
    python backtest.py            # valida o porte + backtest no cofre
    python backtest.py --2020     # backtest em 2020-2026
"""
import os
import sys
import argparse

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
NOVA = r"C:\Users\arthu\Downloads\strategy nova"
IA = r"C:\Users\arthu\OneDrive\Desktop\ia"
sys.path.insert(0, DIR)

from estrategia import build_dirs, Posicao, SL_PTS, TP_PTS, TR_PTS, TF_MS   # noqa: E402

CONFIG = {"tf": "2h", "fam": "HA", "sess": 1, "mm": 0, "azc": 0,
          "sl": 250, "tp": 30, "tr": 25, "norev": 0}
FEE_LADO = 1e-4


def _orig():
    """Carrega o motor original do projeto (so para comparar)."""
    sys.path.insert(0, NOVA); sys.path.insert(0, IA)
    import criativo3, criativo5
    criativo3.FEE = FEE_LADO; criativo5.FEE = FEE_LADO
    import legado_quant as L
    from criativo3 import Bars, seg_stats
    from criativo5 import run_flex
    from prova_cofre import load_vault
    return L, Bars, seg_stats, run_flex, load_vault


def validar_porte(bars, L):
    """O estrategia.py desta pasta gera os mesmos sinais que o original?"""
    d_orig = L.dirs_for(bars, CONFIG)
    d_novo = build_dirs(bars.ox, bars.hx, bars.lx, bars.cx, bars.tsx)
    # o original zera um aquecimento inicial; comparamos depois dele
    w = min(200, len(d_orig) // 8)
    a, b = d_orig[w:], d_novo[w:]
    iguais = int((a == b).sum()); total = len(a)
    difs = int((a != b).sum())
    print(f"  candles comparados : {total:,}")
    print(f"  sinais identicos   : {iguais:,} ({100*iguais/total:.4f}%)")
    print(f"  divergencias       : {difs:,}")
    if difs:
        idx = np.flatnonzero(a != b)[:5]
        for i in idx:
            print(f"    candle {i+w}: original={a[i]:+d}  porte={b[i]:+d}")
    return difs == 0


def backtest_porte(bars):
    """Roda a estrategia PORTADA candle a candle, com o caminho de 1s real."""
    d = build_dirs(bars.ox, bars.hx, bars.lx, bars.cx, bars.tsx)
    rets = []
    pos = None
    n = len(d)
    warm = min(2000, n // 10)
    for t in range(warm, n):
        base, cnt = bars.poff[t], bars.pcnt[t]
        if cnt < 1:
            continue
        o = bars.ox[t]
        op = bars.pn[base] * o          # preco de abertura real do candle
        des = int(d[t])
        # reversao (norev=0): sinal inverteu -> fecha na abertura
        if pos is not None and des != 0 and des != pos.pdir:
            rets.append(pos.retorno_bruto(op) - 2 * FEE_LADO)
            pos = None
        start = base
        if pos is None and des != 0:
            pos = Posicao(des, op, int(bars.tsx[t]))
            start = base + 1
        elif pos is None:
            continue
        # caminho de 1 segundo dentro do candle
        for j in range(start, base + cnt):
            fechou, saida, _ = pos.on_price(bars.pn[j] * o)
            if fechou:
                rets.append(pos.retorno_bruto(saida) - 2 * FEE_LADO)
                pos = None
                break
    return np.array(rets)


def relatorio(r, anos, titulo):
    print(f"\n=== {titulo} ===")
    if len(r) < 5:
        print("  trades insuficientes"); return
    mu, sd = r.mean(), r.std(ddof=1)
    print(f"  trades {len(r)} | winrate {(r>0).mean()*100:.1f}% | "
          f"media {mu*100:+.4f}%/trade | desvio {sd*100:.4f}% | "
          f"t={mu/(sd/np.sqrt(len(r))):.2f}")
    print(f"  {'alav':>4} {'retorno':>13} {'CAGR':>9} {'DD':>7}")
    for lev in (1, 2, 3):
        x = np.maximum(lev * r, -0.999)
        eq = np.cumprod(1 + x)
        dd = (1 - eq / np.maximum.accumulate(eq)).max() * 100
        cagr = (max(eq[-1], 1e-9) ** (1 / anos) - 1) * 100
        print(f"  {lev}x {(eq[-1]-1)*100:>12,.1f}% {cagr:>7.0f}%/a {dd:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--2020", dest="d2020", action="store_true",
                    help="usa 2020-2026 em vez do cofre 2017-20")
    args = ap.parse_args()

    L, Bars, seg_stats, run_flex, load_vault = _orig()
    if args.d2020:
        print("carregando 2020-2026 (dados de selecao)...")
        data = L.load_6y(); rotulo = "2020-2026 (selecao)"
    else:
        print("carregando cofre 2017-2020 (dados VIRGENS)...")
        data = load_vault("eth1s_2017"); rotulo = "COFRE 2017-2020 (virgem)"
    bars = Bars.time_bars(*data, L.TFS["2h"])
    anos = (data[0][-1] - data[0][0]) / (365.25 * 24 * 3600 * 1000)

    print("\n### 1) VALIDACAO DO PORTE (estrategia.py vs motor original)")
    ok = validar_porte(bars, L)
    print(f"  >>> {'PORTE FIEL — paper trading testa a MESMA estrategia' if ok else 'PORTE DIVERGENTE — NAO CONFIE NO PAPER'}")

    print("\n### 2) BACKTEST")
    d = L.dirs_for(bars, CONFIG)
    tt, rr = run_flex(bars, d, CONFIG["sl"], CONFIG["tp"], CONFIG["tr"],
                      no_rev=bool(CONFIG["norev"]))
    relatorio(rr, anos, f"MOTOR ORIGINAL — {rotulo}")
    rp = backtest_porte(bars)
    relatorio(rp, anos, f"CODIGO PORTADO (o que o paper roda) — {rotulo}")


if __name__ == "__main__":
    main()
