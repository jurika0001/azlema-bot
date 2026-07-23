# -*- coding: utf-8 -*-
"""
PONTE entre a estrategia (que decide) e a corretora (que executa).

Fica DE FORA do paper.py de proposito: o paper trading continua rodando puro,
sem risco de um bug da execucao contaminar o teste estatistico. Aqui so
acontece o espelhamento: quando o paper abre/fecha, isto manda para a Lighter.

REGRA CENTRAL: a posicao real e' a que a CORRETORA diz que existe. A memoria do
bot e' apenas um palpite; num reinicio, num erro de rede ou numa ordem recusada,
os dois divergem — e quem manda e' a corretora, sempre.
"""
import os
import threading

from corretora_lighter import Lighter, disjuntor, REAL, CONFIGURADO

ATIVO_REAL = os.environ.get("ATIVO_REAL", "BTC").strip().upper()
_lock = threading.Lock()
_corretora = None
_historico = []


def corretora():
    global _corretora
    if _corretora is None:
        _corretora = Lighter(ATIVO_REAL)
    return _corretora


def _reg(evento, ok, detalhe, extra=None):
    r = {"evento": evento, "ok": ok, "detalhe": detalhe}
    if extra:
        r.update(extra)
    _historico.append(r)
    del _historico[:-200]
    print(f"[executor] {evento}: {'OK' if ok else 'FALHOU'} — {detalhe}", flush=True)
    return r


def abrir(ativo, direcao, preco):
    """Espelha na corretora a abertura que a estrategia decidiu."""
    if ativo != ATIVO_REAL:
        return None                      # so o ativo escolhido opera de verdade
    with _lock:
        c = corretora()
        if not (REAL and c.pronto):
            return _reg("abrir", False, "modo simulado — nenhuma ordem enviada")
        if not disjuntor.liberado():
            return _reg("abrir", False, f"disjuntor: {disjuntor.motivo}")
        saldo = c.saldo()
        if saldo <= 0:
            return _reg("abrir", False, f"saldo indisponivel ({saldo})")
        base = c.tamanho_1x(saldo, preco)     # 1x, respeitando o teto
        if base <= 0:
            return _reg("abrir", False,
                        f"saldo {saldo:.2f} nao cobre o minimo de {c.m['min_base']} {ativo}")
        ok, det = c.abrir(direcao, base, preco)
        return _reg("abrir", ok, det,
                    {"ativo": ativo, "dir": direcao, "base": base, "preco": preco})


def fechar(ativo, preco, motivo=""):
    """Espelha o fechamento. Sempre confere com a corretora antes."""
    if ativo != ATIVO_REAL:
        return None
    with _lock:
        c = corretora()
        if not (REAL and c.pronto):
            return _reg("fechar", False, "modo simulado")
        ok, det = c.fechar(preco)
        return _reg("fechar", ok, f"{det} (motivo: {motivo})", {"ativo": ativo})


def sincronizar(paper_traders):
    """RECONCILIACAO — roda no arranque e periodicamente.

    Compara o que o bot ACHA que tem aberto com o que a corretora diz. Se
    divergirem, a corretora vence. Isso cobre: reinicio do servico, ordem que
    foi recusada, ordem que executou mas a resposta se perdeu, e liquidacao.
    """
    c = corretora()
    if not (REAL and c.pronto):
        return {"modo": "simulado"}
    d_real, base = c.posicao()
    if d_real is None:
        return {"erro": "corretora nao respondeu — nao mexo em nada"}
    pt = paper_traders.get(ATIVO_REAL)
    d_bot = 0 if (pt is None or pt.pos is None) else pt.pos.pdir
    if d_real == d_bot:
        return {"ok": True, "dir": d_real, "base": base, "divergencia": False}
    _reg("reconciliacao", True,
         f"DIVERGENCIA: bot achava {d_bot}, corretora tem {d_real} ({base}). "
         f"A corretora manda.", {"dir_corretora": d_real, "dir_bot": d_bot})
    # posicao orfa na corretora (bot nao sabe dela) -> fecha, para nao ficar
    # exposto a uma posicao que ninguem esta gerenciando
    if d_real != 0 and d_bot == 0:
        preco = getattr(pt, "ultimo_preco", None) if pt else None
        if not preco or preco <= 0:
            return _reg("fechar_orfa", False,
                        "posicao orfa detectada mas SEM preco confiavel — "
                        "nao envio ordem as cegas. Sera' refeito no proximo ciclo.")
        ok, det = c.fechar(preco)
        _reg("fechar_orfa", ok, det)
    return {"ok": True, "divergencia": True, "dir_corretora": d_real, "dir_bot": d_bot}


def status():
    c = corretora()
    return {"ativo_real": ATIVO_REAL, "configurado": CONFIGURADO,
            "corretora": c.status(), "ultimos_eventos": _historico[-10:]}
