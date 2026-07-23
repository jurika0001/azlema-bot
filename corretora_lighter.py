# -*- coding: utf-8 -*-
"""
EXECUCAO REAL NA LIGHTER (MULTI-MERCADO) — ETH, BTC, SOL com UM cliente.

>>> DINHEIRO DE VERDADE PASSA POR AQUI. Cada trava existe por um motivo. <<<

Usado pela estrategia TRIO: uma posicao por vez em QUALQUER um dos 3. A corretora
e' sempre a fonte da verdade sobre o que esta aberto (nunca a memoria do bot).

SEGURANCA DE CREDENCIAL — nada de chave no codigo. Variaveis de ambiente do Render:
    LIGHTER_API_KEY_PRIVATE  chave privada da API KEY (nao a da carteira!)
    LIGHTER_ACCOUNT_INDEX    indice da conta
    LIGHTER_API_KEY_INDEX    indice da api key
    MODO_EXECUCAO            "simulado" (padrao) ou "real"
A chave da carteira NUNCA vai pro servidor. So a chave da API, revogavel.
"""
import os
import asyncio
import threading
import time
from datetime import datetime, timezone

MERCADOS = {
    "ETH": {"id": 0, "size_dec": 4, "price_dec": 2, "min_base": 0.0050},
    "BTC": {"id": 1, "size_dec": 5, "price_dec": 1, "min_base": 0.00020},
    "SOL": {"id": 2, "size_dec": 3, "price_dec": 3, "min_base": 0.050},
}
POR_ID = {m["id"]: a for a, m in MERCADOS.items()}

MODO = os.environ.get("MODO_EXECUCAO", "simulado").strip().lower()
TETO_USD = float(os.environ.get("TETO_USD", "80"))
PERDA_DIA_MAX = float(os.environ.get("PERDA_DIA_MAX", "15"))
DESLIZAMENTO_MAX = float(os.environ.get("DESLIZAMENTO_MAX", "0.30"))
ERROS_MAX = 5

_API_PRIV = os.environ.get("LIGHTER_API_KEY_PRIVATE", "").strip()
_ACC_IDX = os.environ.get("LIGHTER_ACCOUNT_INDEX", "").strip()
_KEY_IDX = os.environ.get("LIGHTER_API_KEY_INDEX", "").strip()
_URL = os.environ.get("LIGHTER_URL", "https://mainnet.zklighter.elliot.ai")

CONFIGURADO = bool(_API_PRIV and _ACC_IDX and _KEY_IDX)
REAL = (MODO == "real") and CONFIGURADO


class Disjuntor:
    def __init__(self):
        self.erros = 0; self.travado = False; self.motivo = ""
        self.dia = None; self.pnl_dia = 0.0

    def registra_erro(self, e):
        self.erros += 1
        if self.erros >= ERROS_MAX:
            self.trava(f"{ERROS_MAX} erros seguidos ({e})")

    def registra_ok(self):
        self.erros = 0

    def registra_trade(self, pct):
        hoje = datetime.now(timezone.utc).date()
        if self.dia != hoje:
            self.dia, self.pnl_dia = hoje, 0.0
        self.pnl_dia += pct
        if self.pnl_dia <= -PERDA_DIA_MAX:
            self.trava(f"perda do dia {self.pnl_dia:.1f}% <= -{PERDA_DIA_MAX}%")

    def trava(self, motivo):
        self.travado = True; self.motivo = motivo
        print(f"[DISJUNTOR] OPERACAO TRAVADA: {motivo}", flush=True)

    def liberado(self):
        return not self.travado


disjuntor = Disjuntor()


def _int_seguro(nome, valor):
    """Converte para int SEM ecoar o conteudo (se a chave foi colada no campo
    errado, o valor NUNCA aparece no erro/painel — so tamanho e uma dica)."""
    v = (valor or "").strip().strip('"').strip("'")
    if v.lstrip("-").isdigit():
        return int(v)
    dica = f"tem {len(v)} caracteres"
    if v.lower().startswith("0x"):
        dica += " e comeca com 0x — isso parece a CHAVE, nao um indice!"
    elif len(v) > 12:
        dica += " — longo demais para um indice"
    raise ValueError(f"{nome} precisa ser um NUMERO inteiro; {dica}")


class Lighter:
    """Um cliente, tres mercados. Sem credencial -> fica em simulado."""

    def __init__(self):
        self.api = None; self.cliente = None; self.loop = None
        self.pronto = False; self.erro_init = None
        # CONECTA sempre que houver credencial (mesmo em simulado): conectar e LER
        # saldo/posicoes e' seguro (read-only). So ENVIAR ORDEM depende de REAL.
        # Assim da' pra verificar a conexao antes de virar a chave.
        if CONFIGURADO:
            self._conectar()

    def _ensure_loop(self):
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def _rodar(self, coro):
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=30)

    def _conectar(self):
        try:
            import lighter
            self._ensure_loop()

            # valida os indices SEM ecoar o conteudo (evita vazar chave se ela foi
            # colada no campo errado). So diz qual campo e' invalido e uma dica.
            acc_i = _int_seguro("LIGHTER_ACCOUNT_INDEX", _ACC_IDX)
            key_i = _int_seguro("LIGHTER_API_KEY_INDEX", _KEY_IDX)

            async def _setup():
                # criado DENTRO do loop -> ha' event loop rodando (corrige o erro
                # "no running event loop": a SDK e' async e cria recursos do loop)
                api = lighter.ApiClient(configuration=lighter.Configuration(host=_URL))
                cli = lighter.SignerClient(
                    url=_URL, private_key=_API_PRIV,
                    account_index=acc_i, api_key_index=key_i)
                res = cli.check_client()
                if asyncio.iscoroutine(res):
                    res = await res
                return api, cli, res

            self.api, self.cliente, err = asyncio.run_coroutine_threadsafe(
                _setup(), self.loop).result(timeout=30)
            if err is not None:
                raise RuntimeError(f"check_client: {err}")
            self.pronto = True
            print(f"[lighter] conectado | conta {_ACC_IDX} | ETH/BTC/SOL", flush=True)
        except Exception as e:
            self.erro_init = f"{type(e).__name__}: {e}"; self.pronto = False
            print(f"[lighter] FALHA ao conectar: {self.erro_init}", flush=True)

    # ------------------------------------------------ leitura (fonte da verdade)
    def _posicoes_raw(self):
        import lighter
        r = self._rodar(lighter.AccountApi(self.api).account(by="index", value=str(_ACC_IDX)))
        out = {}
        for acc in getattr(r, "accounts", []) or []:
            for p in getattr(acc, "positions", []) or []:
                mid = int(getattr(p, "market_id", -1))
                if mid not in POR_ID:
                    continue
                sz = float(getattr(p, "position", 0) or 0)
                if abs(sz) > 1e-12:
                    out[POR_ID[mid]] = (1 if sz > 0 else -1, abs(sz))
        return out

    def posicoes_todas(self):
        """Dict {ativo: (dir, base)} de TODAS as posicoes abertas na corretora.
        Vazio = nada aberto. None = corretora nao respondeu (nao assume nada).
        Leitura e' segura em qualquer modo (nao envia ordem)."""
        if not self.pronto:
            return {}
        try:
            r = self._posicoes_raw(); disjuntor.registra_ok(); return r
        except Exception as e:
            disjuntor.registra_erro(e)
            print(f"[lighter] erro ao ler posicoes: {e}", flush=True)
            return None

    def saldo(self):
        if not self.pronto:
            return 0.0
        try:
            import lighter
            r = self._rodar(lighter.AccountApi(self.api).account(by="index", value=str(_ACC_IDX)))
            for acc in getattr(r, "accounts", []) or []:
                v = getattr(acc, "available_balance", None) or getattr(acc, "collateral", None)
                if v is not None:
                    return float(v)
            return 0.0
        except Exception as e:
            disjuntor.registra_erro(e); return 0.0

    # ------------------------------------------------ escrita
    def _lotes(self, ativo, base):
        return int(round(base * (10 ** MERCADOS[ativo]["size_dec"])))

    def _preco_int(self, ativo, preco):
        return int(round(preco * (10 ** MERCADOS[ativo]["price_dec"])))

    def tamanho_1x(self, ativo, saldo_usd, preco):
        usd = min(saldo_usd, TETO_USD)
        base = usd / preco
        d = 10 ** MERCADOS[ativo]["size_dec"]
        base = int(base * d) / d
        return base if base >= MERCADOS[ativo]["min_base"] else 0.0

    def abrir(self, ativo, direcao, base, preco_ref):
        if not disjuntor.liberado():
            return (False, f"disjuntor travado: {disjuntor.motivo}")
        if not (REAL and self.pronto):
            return (False, "modo simulado (nenhuma ordem enviada)")
        # TRAVA: nunca abre se JA existe posicao em QUALQUER mercado
        pos = self.posicoes_todas()
        if pos is None:
            return (False, "posicoes desconhecidas — nao arrisco abrir")
        if pos:
            return (False, f"ja existe posicao aberta: {pos}")
        if base < MERCADOS[ativo]["min_base"]:
            return (False, f"tamanho {base} < minimo {MERCADOS[ativo]['min_base']}")
        pior = preco_ref * (1 + direcao * DESLIZAMENTO_MAX / 100.0)
        try:
            tx, tx_hash, err = self._rodar(self.cliente.create_market_order(
                market_index=MERCADOS[ativo]["id"],
                client_order_index=int(time.time()) % 1_000_000,
                base_amount=self._lotes(ativo, base),
                avg_execution_price=self._preco_int(ativo, pior),
                is_ask=(direcao == -1)))
            if err is not None:
                disjuntor.registra_erro(err); return (False, f"corretora recusou: {err}")
            disjuntor.registra_ok()
            return (True, f"{ativo} tx={tx_hash} base={base} pior={pior:.4f}")
        except Exception as e:
            disjuntor.registra_erro(e); return (False, f"{type(e).__name__}: {e}")

    def fechar(self, ativo, preco_ref):
        if not (REAL and self.pronto):
            return (False, "modo simulado")
        pos = self.posicoes_todas()
        if pos is None:
            return (False, "posicoes desconhecidas")
        if ativo not in pos:
            return (True, f"nada aberto em {ativo}")
        d, base = pos[ativo]; contra = -d
        pior = preco_ref * (1 + contra * DESLIZAMENTO_MAX / 100.0)
        try:
            tx, tx_hash, err = self._rodar(self.cliente.create_market_order(
                market_index=MERCADOS[ativo]["id"],
                client_order_index=int(time.time()) % 1_000_000,
                base_amount=self._lotes(ativo, base),
                avg_execution_price=self._preco_int(ativo, pior),
                is_ask=(contra == -1)))
            if err is not None:
                disjuntor.registra_erro(err); return (False, f"corretora recusou: {err}")
            disjuntor.registra_ok()
            return (True, f"fechado {ativo} tx={tx_hash} base={base}")
        except Exception as e:
            disjuntor.registra_erro(e); return (False, f"{type(e).__name__}: {e}")

    def status(self):
        pos = self.posicoes_todas() if self.pronto else {}
        return {"modo": "REAL" if (REAL and self.pronto) else "simulado",
                "configurado": CONFIGURADO, "conectado": self.pronto,
                "saldo": self.saldo() if self.pronto else None,
                "erro_init": self.erro_init, "posicoes_na_corretora": pos,
                "disjuntor": {"travado": disjuntor.travado, "motivo": disjuntor.motivo,
                              "erros": disjuntor.erros, "pnl_dia_pct": disjuntor.pnl_dia},
                "limites": {"teto_usd": TETO_USD, "perda_dia_max_pct": PERDA_DIA_MAX,
                            "deslizamento_max_pct": DESLIZAMENTO_MAX, "alavancagem": "1x"}}
