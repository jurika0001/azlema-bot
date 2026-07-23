# -*- coding: utf-8 -*-
"""
EXECUCAO REAL NA LIGHTER — camada que efetivamente envia ordem.

>>> DINHEIRO DE VERDADE PASSA POR AQUI. Cada trava abaixo existe por um motivo. <<<

SEGURANCA DE CREDENCIAL
  Nada de chave no codigo. Tudo vem das variaveis de ambiente do Render:
    LIGHTER_API_KEY_PRIVATE  chave privada da API KEY (nao e' a da carteira!)
    LIGHTER_ACCOUNT_INDEX    indice da sua conta
    LIGHTER_API_KEY_INDEX    indice da api key (ex: 3)
    MODO_EXECUCAO            "simulado" (padrao) ou "real"

  A chave da API e' gerada pelo system_setup.py da SDK, rodado UMA VEZ por voce
  na sua maquina, com a chave da sua carteira. A carteira NUNCA vai para o
  servidor — so a chave da API, que voce pode revogar quando quiser.

TRAVAS (todas ligadas por padrao)
  1. MODO_EXECUCAO=simulado por padrao — so opera de verdade se voce escrever
     "real" explicitamente. Nao existe caminho acidental para o dinheiro.
  2. Alavancagem 1x: o valor da posicao nunca passa do saldo.
  3. Teto absoluto de posicao (TETO_USD).
  4. Protecao de deslizamento: a ordem carrega o pior preco aceitavel; se o
     mercado correr, ela NAO executa a qualquer preco.
  5. Reconciliacao: a posicao real vem SEMPRE da corretora, nunca da memoria.
     Se o bot reiniciar, ele descobre o que tem aberto perguntando pra ela.
  6. Disjuntor: erros seguidos ou perda diaria acima do limite -> para de operar.
  7. Nunca abre uma posicao se ja existir outra aberta.
"""
import os
import asyncio
import threading
import time
from datetime import datetime, timezone

MERCADOS = {                       # market_id, decimais de tamanho e de preco
    "ETH": {"id": 0, "size_dec": 4, "price_dec": 2, "min_base": 0.0050},
    "BTC": {"id": 1, "size_dec": 5, "price_dec": 1, "min_base": 0.00020},
    "SOL": {"id": 2, "size_dec": 3, "price_dec": 3, "min_base": 0.050},
}

MODO = os.environ.get("MODO_EXECUCAO", "simulado").strip().lower()
TETO_USD = float(os.environ.get("TETO_USD", "100"))      # teto duro de posicao
PERDA_DIA_MAX = float(os.environ.get("PERDA_DIA_MAX", "15"))  # % -> disjuntor
DESLIZAMENTO_MAX = float(os.environ.get("DESLIZAMENTO_MAX", "0.30"))  # %
ERROS_MAX = 5

_API_PRIV = os.environ.get("LIGHTER_API_KEY_PRIVATE", "").strip()
_ACC_IDX = os.environ.get("LIGHTER_ACCOUNT_INDEX", "").strip()
_KEY_IDX = os.environ.get("LIGHTER_API_KEY_INDEX", "").strip()
_URL = os.environ.get("LIGHTER_URL", "https://mainnet.zklighter.elliot.ai")
_CHAIN = int(os.environ.get("LIGHTER_CHAIN_ID", "304"))

CONFIGURADO = bool(_API_PRIV and _ACC_IDX and _KEY_IDX)
REAL = (MODO == "real") and CONFIGURADO


class Disjuntor:
    """Para de operar quando algo vai claramente mal."""

    def __init__(self):
        self.erros = 0
        self.travado = False
        self.motivo = ""
        self.dia = None
        self.pnl_dia = 0.0

    def registra_erro(self, e):
        self.erros += 1
        if self.erros >= ERROS_MAX:
            self.trava(f"{ERROS_MAX} erros seguidos na corretora ({e})")

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
        self.travado = True
        self.motivo = motivo
        print(f"[DISJUNTOR] OPERACAO TRAVADA: {motivo}", flush=True)

    def liberado(self):
        return not self.travado


disjuntor = Disjuntor()


class Lighter:
    """Adaptador da corretora. Sem credencial -> fica em modo simulado."""

    def __init__(self, ativo="BTC"):
        self.ativo = ativo
        self.m = MERCADOS[ativo]
        self.cliente = None
        self.api = None
        self.loop = None
        self.pronto = False
        self.erro_init = None
        if REAL:
            self._conectar()

    # ------------------------------------------------------ infra
    def _rodar(self, coro):
        """A SDK e' async; o bot e' de threads. Isola num loop proprio."""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            threading.Thread(target=self.loop.run_forever, daemon=True).start()
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=30)

    def _conectar(self):
        try:
            import lighter
            self.api = lighter.ApiClient(configuration=lighter.Configuration(host=_URL))
            self.cliente = lighter.SignerClient(
                url=_URL, private_key=_API_PRIV,
                account_index=int(_ACC_IDX), api_key_index=int(_KEY_IDX),
            )
            err = self.cliente.check_client()
            if err is not None:
                raise RuntimeError(f"check_client: {err}")
            self.pronto = True
            print(f"[lighter] conectado | conta {_ACC_IDX} | {self.ativo} "
                  f"(market {self.m['id']}) | MODO REAL", flush=True)
        except Exception as e:
            self.erro_init = f"{type(e).__name__}: {e}"
            self.pronto = False
            print(f"[lighter] FALHA ao conectar: {self.erro_init} — fica em SIMULADO",
                  flush=True)

    # ------------------------------------------------------ leitura
    def posicao(self):
        """FONTE DA VERDADE: pergunta a posicao para a CORRETORA, nao a memoria.
        Devolve (direcao, tamanho_base) — (0, 0.0) se nao houver nada aberto."""
        if not (REAL and self.pronto):
            return (0, 0.0)
        try:
            import lighter
            r = self._rodar(lighter.AccountApi(self.api).account(
                by="index", value=str(_ACC_IDX)))
            disjuntor.registra_ok()
            for acc in getattr(r, "accounts", []) or []:
                for p in getattr(acc, "positions", []) or []:
                    if int(getattr(p, "market_id", -1)) != self.m["id"]:
                        continue
                    sz = float(getattr(p, "position", 0) or 0)
                    if abs(sz) < 1e-12:
                        return (0, 0.0)
                    return (1 if sz > 0 else -1, abs(sz))
            return (0, 0.0)
        except Exception as e:
            disjuntor.registra_erro(e)
            print(f"[lighter] erro ao ler posicao: {e}", flush=True)
            return (None, 0.0)          # None = DESCONHECIDO (nao assume nada)

    def saldo(self):
        if not (REAL and self.pronto):
            return 0.0
        try:
            import lighter
            r = self._rodar(lighter.AccountApi(self.api).account(
                by="index", value=str(_ACC_IDX)))
            for acc in getattr(r, "accounts", []) or []:
                v = getattr(acc, "available_balance", None) or getattr(acc, "collateral", None)
                if v is not None:
                    return float(v)
            return 0.0
        except Exception as e:
            disjuntor.registra_erro(e)
            return 0.0

    # ------------------------------------------------------ escrita
    def _lotes(self, base):
        return int(round(base * (10 ** self.m["size_dec"])))

    def _preco_int(self, preco):
        return int(round(preco * (10 ** self.m["price_dec"])))

    def tamanho_1x(self, saldo_usd, preco):
        """Alavancagem 1x: valor da posicao <= saldo. Respeita o teto duro."""
        usd = min(saldo_usd, TETO_USD)
        base = usd / preco
        base = int(base * (10 ** self.m["size_dec"])) / (10 ** self.m["size_dec"])
        if base < self.m["min_base"]:
            return 0.0
        return base

    def abrir(self, direcao, base, preco_ref):
        """Abre a mercado com protecao de deslizamento.
        direcao: +1 compra, -1 vende. Devolve (ok, detalhe)."""
        if not disjuntor.liberado():
            return (False, f"disjuntor travado: {disjuntor.motivo}")
        if not (REAL and self.pronto):
            return (False, "modo simulado (nenhuma ordem enviada)")
        # TRAVA: nunca abre por cima de posicao existente
        d_atual, sz = self.posicao()
        if d_atual is None:
            return (False, "posicao desconhecida — nao arrisco abrir")
        if d_atual != 0:
            return (False, f"ja existe posicao aberta ({d_atual}, {sz})")
        if base < self.m["min_base"]:
            return (False, f"tamanho {base} < minimo {self.m['min_base']}")
        # pior preco aceitavel: compra nao paga acima, venda nao aceita abaixo
        pior = preco_ref * (1 + direcao * DESLIZAMENTO_MAX / 100.0)
        try:
            tx, tx_hash, err = self._rodar(self.cliente.create_market_order(
                market_index=self.m["id"],
                client_order_index=int(time.time()) % 1_000_000,
                base_amount=self._lotes(base),
                avg_execution_price=self._preco_int(pior),
                is_ask=(direcao == -1),
            ))
            if err is not None:
                disjuntor.registra_erro(err)
                return (False, f"corretora recusou: {err}")
            disjuntor.registra_ok()
            return (True, f"tx={tx_hash} base={base} pior_preco={pior:.2f}")
        except Exception as e:
            disjuntor.registra_erro(e)
            return (False, f"{type(e).__name__}: {e}")

    def fechar(self, preco_ref):
        """Fecha o que estiver aberto, na direcao contraria."""
        if not (REAL and self.pronto):
            return (False, "modo simulado")
        d, base = self.posicao()
        if d is None:
            return (False, "posicao desconhecida")
        if d == 0:
            return (True, "nada aberto")
        contra = -d
        pior = preco_ref * (1 + contra * DESLIZAMENTO_MAX / 100.0)
        try:
            tx, tx_hash, err = self._rodar(self.cliente.create_market_order(
                market_index=self.m["id"],
                client_order_index=int(time.time()) % 1_000_000,
                base_amount=self._lotes(base),
                avg_execution_price=self._preco_int(pior),
                is_ask=(contra == -1),
            ))
            if err is not None:
                disjuntor.registra_erro(err)
                return (False, f"corretora recusou: {err}")
            disjuntor.registra_ok()
            return (True, f"fechado tx={tx_hash} base={base}")
        except Exception as e:
            disjuntor.registra_erro(e)
            return (False, f"{type(e).__name__}: {e}")

    def status(self):
        d, sz = (self.posicao() if (REAL and self.pronto) else (0, 0.0))
        return {
            "modo": "REAL" if (REAL and self.pronto) else "simulado",
            "configurado": CONFIGURADO, "conectado": self.pronto,
            "erro_init": self.erro_init, "ativo": self.ativo,
            "market_id": self.m["id"],
            "posicao_na_corretora": {"dir": d, "base": sz},
            "disjuntor": {"travado": disjuntor.travado, "motivo": disjuntor.motivo,
                          "erros": disjuntor.erros, "pnl_dia_pct": disjuntor.pnl_dia},
            "limites": {"teto_usd": TETO_USD, "perda_dia_max_pct": PERDA_DIA_MAX,
                        "deslizamento_max_pct": DESLIZAMENTO_MAX, "alavancagem": "1x"},
        }
