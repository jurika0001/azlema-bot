# -*- coding: utf-8 -*-
"""
ESTADO REMOTO via GitHub Gist — persistencia GRATIS que sobrevive a qualquer
reinicio/deploy do Render free (que apaga o disco local a cada restart).

SEGURANCA: o token NUNCA fica no codigo. Vem das variaveis de ambiente que
VOCE configura no painel do Render (Environment):
    GIST_ID     = id do gist privado que voce criou
    GIST_TOKEN  = Personal Access Token do GitHub, escopo 'gist' apenas

Sem essas variaveis, o modulo se desliga sozinho e o bot usa so o disco local
(comportamento antigo). Com elas, cada save vai tambem para o gist e cada
startup carrega de la — entao o teste de 5 meses nao perde nada num restart.
"""
import os
import json
import time
import threading

import requests

GIST_ID = os.environ.get("GIST_ID", "").strip()
GIST_TOKEN = os.environ.get("GIST_TOKEN", "").strip()
ATIVO = bool(GIST_ID and GIST_TOKEN)

_API = f"https://api.github.com/gists/{GIST_ID}"
_lock = threading.Lock()
_ultimo_save = 0.0


def _headers():
    return {"Authorization": f"Bearer {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def carregar(nome_arquivo):
    """Le um arquivo do gist. Devolve o dict, ou None se nao existir/erro."""
    if not ATIVO:
        return None
    try:
        r = requests.get(_API, headers=_headers(), timeout=20)
        r.raise_for_status()
        arqs = r.json().get("files", {})
        f = arqs.get(nome_arquivo)
        if not f:
            return None
        # gists grandes vem truncados: nesse caso baixa pelo raw_url
        if f.get("truncated") and f.get("raw_url"):
            rr = requests.get(f["raw_url"], timeout=20); rr.raise_for_status()
            return json.loads(rr.text)
        return json.loads(f["content"])
    except Exception as e:
        print(f"[gist] falha ao carregar {nome_arquivo}: {type(e).__name__}: {e}",
              flush=True)
        return None


def salvar(nome_arquivo, obj, min_intervalo=0.0):
    """Grava um arquivo no gist. min_intervalo limita a frequencia (segundos);
    passe 0 para forcar (ex: quando uma trade fecha)."""
    global _ultimo_save
    if not ATIVO:
        return False
    with _lock:
        agora = time.time()
        if min_intervalo > 0 and (agora - _ultimo_save) < min_intervalo:
            return False
        try:
            payload = {"files": {nome_arquivo: {"content": json.dumps(obj)}}}
            r = requests.patch(_API, headers=_headers(), json=payload, timeout=20)
            r.raise_for_status()
            _ultimo_save = agora
            return True
        except Exception as e:
            print(f"[gist] falha ao salvar {nome_arquivo}: {type(e).__name__}: {e}",
                  flush=True)
            return False


def status():
    return {"ativo": ATIVO, "gist_id": (GIST_ID[:8] + "...") if GIST_ID else None,
            "token_presente": bool(GIST_TOKEN)}
