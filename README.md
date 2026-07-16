# Paper Trading — HA-2h 250/30/25 (ETH)

Testa em **tempo real**, sobre candles que ainda não existem, a única estratégia
que sobreviveu a todos os juízes do projeto. **Nenhum dinheiro envolvido. Nenhuma
ordem enviada a lugar nenhum.**

## Por que este teste existe

Já testamos 115.030 estratégias. Todos os outros juízes (fatias, walk-forward,
cofre) olham para o **passado** — e por mais rigorosos que sejam, a estratégia foi
escolhida *depois* de olhar esses dados. Isso deixa uma brecha: com 115 mil
tentativas, alguma passa por sorte.

Paper trading prospectivo é o único juiz que fecha essa brecha, porque os dados
**ainda não existem** quando a aposta é feita.

## A estratégia (parâmetros travados)

| | |
|---|---|
| Candle | 2h |
| Sinal | Heikin-Ashi + AzLema concordando, sessão 12–22 UTC |
| Stop loss | 250 pontos = **2,50%** (só vale antes de armar) |
| TP | 30 pontos = **0,30%** — **não fecha**, apenas *arma* o trailing |
| Trailing | 25 pontos = **0,25%** (só depois de armar) |
| Reversão | sinal inverteu → fecha na abertura do candle seguinte |

**Porte validado:** `backtest.py` compara o `estrategia.py` (que o paper roda)
contra o motor original do projeto: **12.324 candles, 0 divergências**. O paper
testa exatamente a mesma estratégia que passou no cofre.

## Referência a bater (cofre virgem 2017-2020)

```
3.561 trades | winrate 87,2% | média +0,0518%/trade | desvio 0,8030% | t = 3,85
1x  +462,3%   (82%/a)   DD 21,0%
2x  +2.391,8% (206%/a)  DD 39,2%
```

---

# ⏱️ Quanto tempo até saber se passou

**Resposta curta: ~9 meses.**

O edge é **+0,0518% por trade** com desvio de **0,8030%**. O sinal é pequeno
perto do ruído, então é preciso volume de trades para separar vantagem de sorte:

| Confiança | Trades | Tempo (a 3,4 trades/dia) |
|---|---|---|
| **95%** ← critério | **925** | **~9 meses** |
| 99% | 1.602 | ~15,5 meses |
| 99,9% | 2.605 | ~25 meses |

### Critério de aprovação (travado antes de começar, para não mover a trave)

> **PASSOU** = `n ≥ 925` **E** `t-stat ≥ 1,96` **E** média > 0

O `/status` calcula isso sozinho e mostra `PASSOU` / `REPROVOU` / `coletando`.

### Não espere 9 meses em silêncio — sinais de alerta antes disso

- **~1 mês (100 trades):** o winrate deveria estar perto de 87%. Se estiver
  abaixo de ~75%, algo está errado (spread, execução ou o edge sumiu).
- **~3 meses (300 trades):** a média/trade deveria estar perto de +0,05%. Se
  estiver **negativa**, pode parar — não vai virar.
- **Qualquer momento:** trade individual pior que −2,5% significa bug (o SL é 2,5%).

### ⚠️ O spread muda tudo

O edge de +0,0518%/trade **já inclui** a taxa do KCEX (0,02% ida-volta), mas
**não inclui o spread**, e você opera a mercado (paga spread em 100% das trades):

| Spread ida-volta | Edge líquido | Tempo até 95% |
|---|---|---|
| 0,00% | +0,0518% | 9 meses |
| 0,01% | +0,0418% | 14 meses |
| 0,02% | +0,0318% | 24 meses |
| 0,03% | +0,0218% | 50 meses |
| **0,05%** | **+0,0018%** | **inviável** |
| **0,06%** | **−0,0082%** | **estratégia morta** |

**Um spread de 0,06% zera a estratégia inteira.** Ajuste `SPREAD_RT` em
`paper.py` para o valor real medido na sua corretora antes de confiar em
qualquer número. É a variável mais importante do projeto — mais que a taxa,
mais que a alavancagem.

---

# Deploy no Render

```bash
cd "C:\Users\arthu\OneDrive\Documentos\testerender"
git init && git add . && git commit -m "paper trading HA-2h"
# suba para um repo no GitHub, depois no Render: New > Blueprint > aponte o repo
```

O `render.yaml` já está configurado. Local, para testar:

```bash
pip install -r requirements.txt
python app.py          # abre em http://localhost:10000
```

## Endpoints

| Rota | O que faz |
|---|---|
| `/` | painel (atualiza sozinho a cada 30s) |
| `/status` | JSON com o veredito e todas as estatísticas |
| `/log` | últimas 100 linhas |
| `/ping` | **sinal externo** — keep-alive |
| `/sinal` | **sinal externo** — recebe JSON (só registra, não opera) |

## Sinais internos

| Timer | Função |
|---|---|
| **13s** | heartbeat — prova de vida, contador |
| **21s** | preço — lê o ETH e gerencia SL / TP / trailing |
| **30s** | estado — grava em disco + checa virada do candle de 2h |

---

# ⚠️ Três coisas realistas que você precisa saber

### 1. Os sinais internos NÃO impedem o Render de dormir

O Render free derruba o serviço após **15 min sem tráfego de entrada** — ele
olha requisições HTTP externas, não atividade interna. Seus timers de 13/21/30s
rodam felizes enquanto o serviço é desligado embaixo deles.

**Duas soluções:**
- **Plano `starter` pago** (já no `render.yaml`) — não dorme. É o que recomendo:
  o teste dura 9 meses, e serviço dormindo = trades perdidas = teste inválido.
- **Free + pinger externo** — aponte cron-job.org ou UptimeRobot para
  `https://SEU-APP.onrender.com/ping` a cada 10 min. Funciona, mas o cold start
  ainda faz perder trades.

### 2. Ler preço a cada 21s ≠ backtest de 1 segundo

O backtest simula SL/TP/trailing com **resolução de 1 segundo**. O paper lê a
cada **21s** — 21x menos. O trailing (0,25%) vai disparar em preços piores do que
no backtest. Com um edge de só 0,052%/trade, **essa diferença sozinha pode comer
boa parte da vantagem**.

Isso não é bug, é limite de arquitetura — e é exatamente o tipo de coisa que o
paper trading existe para revelar. Se você quiser fidelidade real ao backtest, me
peça para trocar o polling de 21s por um **WebSocket de trades** (streaming
contínuo, resolução de milissegundos, sem custo extra).

### 3. `/sinal` registra, não executa

Conteúdo que chega de fora é **dado, não ordem**. O endpoint grava o que recebeu
e responde `registrado_sem_operar`. Se você quiser que um sinal externo abra
trade, isso precisa ser ligado explicitamente e com validação de origem — senão
qualquer um que descubra a URL abre posições na sua conta.

---

# O que este teste NÃO resolve

- **Slippage real** — paper preenche no preço lido; a mercado você preenche pior.
- **Funding** — perpétuo cobra ~0,01%/8h (~11%/a a 1x, ~22%/a a 2x). Não está
  em nenhum número aqui. A 2x, isso derruba 206%/a para ~184%/a.
- **Liquidez** — o backtest assume que você sempre é preenchido.

Se o paper passar em 9 meses, aí sim vale um teste com dinheiro pequeno de
verdade — que é o único juiz acima deste.
