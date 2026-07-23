# -*- coding: utf-8 -*-
"""
PONTE entre o TRIO (que decide) e a Lighter (que executa).

O TRIO tem UMA posicao por vez entre ETH/BTC/SOL. Este modulo espelha isso na
corretora: quando o trio abre em X, abre em X (1x, saldo total); quando fecha,
fecha em X. Fica FORA do paper de proposito — um bug de execucao nao contamina
o teste estatistico.

REGRA CENTRAL: a posicao real e' a que a CORRETORA diz que existe. Num reinicio,
erro de rede ou ordem recusada, a memoria e a corretora divergem — e quem manda
e' a corretora, sempre (reconciliacao).
"""
import threading

from corretora_lighter import Lighter, disjuntor, REAL, CONFIGURADO, MERCADOS

_lock = threading.Lock()
_lighter = None
_hist = []


def corretora():
    global _lighter
    if _lighter is None:
        _lighter = Lighter()
    return _lighter


def _reg(ev, ok, det, extra=None):
    r = {"evento": ev, "ok": ok, "detalhe": det}
    if extra:
        r.update(extra)
    _hist.append(r); del _hist[:-200]
    print(f"[executor] {ev}: {'OK' if ok else 'FALHOU'} — {det}", flush=True)
    return r


def abrir(ativo, direcao, preco):
    """Espelha a abertura que o TRIO decidiu."""
    if ativo not in MERCADOS:
        return None
    with _lock:
        c = corretora()
        if not (REAL and c.pronto):
            return _reg("abrir", False, "modo simulado — nenhuma ordem enviada")
        if not disjuntor.liberado():
            return _reg("abrir", False, f"disjuntor: {disjuntor.motivo}")
        saldo = c.saldo()
        if saldo <= 0:
            return _reg("abrir", False, f"saldo indisponivel ({saldo})")
        base = c.tamanho_1x(ativo, saldo, preco)
        if base <= 0:
            return _reg("abrir", False,
                        f"saldo {saldo:.2f} nao cobre o minimo de {ativo}")
        ok, det = c.abrir(ativo, direcao, base, preco)
        return _reg("abrir", ok, det, {"ativo": ativo, "dir": direcao, "base": base})


def fechar(ativo, preco, motivo=""):
    if ativo not in MERCADOS:
        return None
    with _lock:
        c = corretora()
        if not (REAL and c.pronto):
            return _reg("fechar", False, "modo simulado")
        ok, det = c.fechar(ativo, preco)
        return _reg("fechar", ok, f"{det} (motivo: {motivo})", {"ativo": ativo})


def sincronizar(trio):
    """RECONCILIACAO — a corretora manda. Roda periodicamente e no arranque.

    Compara o que o TRIO acha aberto com o que a Lighter diz. Divergiu ->
    a corretora vence. Cobre reinicio, ordem recusada, resposta perdida,
    liquidacao. Posicao orfa (a Lighter tem, o trio nao sabe) -> fecha, para
    nao ficar exposto a algo que ninguem gerencia."""
    c = corretora()
    if not (REAL and c.pronto):
        return {"modo": "simulado"}
    pos = c.posicoes_todas()
    if pos is None:
        return {"erro": "corretora nao respondeu — nao mexo em nada"}
    tativo = trio.pos_ativo
    tdir = trio.pos.pdir if trio.pos else 0
    real = {a: d for a, (d, b) in pos.items()}
    # 1) a corretora tem alguma posicao que o trio NAO conhece -> fecha (orfa)
    for a, d in real.items():
        if a != tativo:
            preco = (trio.ultimo.get(a) or {}).get("preco")
            if preco and preco > 0:
                ok, det = c.fechar(a, preco)
                _reg("fechar_orfa", ok, f"{a} dir {d}: {det}")
            else:
                _reg("fechar_orfa", False, f"{a} orfa mas sem preco — refaz no proximo ciclo")
    # 2) o trio acha que tem posicao mas a corretora NAO tem -> alerta (nao mexe)
    if tativo and tativo not in real:
        _reg("divergencia", True,
             f"trio acha {tativo} dir {tdir}, corretora NAO tem. A corretora manda.")
    return {"ok": True, "corretora": real,
            "trio": ({tativo: tdir} if tativo else {})}


def status():
    return {"estrategia": "TRIO ETH+BTC+SOL (1 posicao por vez, 1x)",
            "configurado": CONFIGURADO, "corretora": corretora().status(),
            "ultimos_eventos": _hist[-10:]}
