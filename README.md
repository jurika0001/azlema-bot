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

## Referência a bater (cofre virgem 2017-2020, taxa 0% = Lighter)

```
3.561 trades | winrate 87,2% | média +0,0718%/trade | desvio 0,8030% | t = 5,33
1x  +1.045,8% (134%/a)  DD 18,1%   ->  78 USDT vira  894 USDT em 2,9a
2x  +10.240,9% (403%/a) DD 33,4%   ->  78 USDT vira 8.066 USDT em 2,9a
```

---

# ⏱️ Quanto tempo até saber se passou

**Resposta: ~9,5 meses** (981 trades, a ~3,4 trades/dia).

### ⚠️ Correção de um erro importante

A conta `n = (1,96 × desvio/média)²` **está errada** e foi usada numa versão
anterior deste arquivo. Nesse `n`, o t-stat **esperado** é exatamente 1,96 —
então **metade das amostras cai abaixo por puro acaso**. Isso dá **50% de poder**:
cara ou coroa. Rodar 4,6 meses e ver "REPROVOU" teria 50% de chance de estar
matando uma estratégia boa.

O correto inclui o **poder** do teste: `n = ((z_α + z_β) × desvio/média)²`.

Verificado por simulação (4.000 amostras por ponto, edge real de +0,0718%):

| Trades | Tempo | Poder (chance de detectar um edge que existe) |
|---|---|---|
| 481 | 4,6 meses | **50,2%** ← inútil |
| **981** | **9,5 meses** | **79,4%** ← o critério |
| 1.313 | 12,7 meses | 90% |
| 2.000 | 19,3 meses | 98,2% |

### Critério de aprovação (travado antes de começar, para não mover a trave)

> **PASSOU** = `n ≥ 981` **E** `t-stat ≥ 1,96` **E** média > 0

O `/status` calcula sozinho e mostra `PASSOU` / `MORTA` / `coletando`.

---

# 🏦 Por que Lighter — e por que a taxa zero é obrigatória

O edge é fino demais para pagar taxa. **Um único teste mede todas as corretoras
ao mesmo tempo** (guardo o retorno bruto e aplico cada taxa em paralelo), então
você não precisa apostar 9 meses numa só:

| Corretora | Taker | KYC | API | Edge/trade | Trades p/ 80% poder |
|---|---|---|---|---|---|
| **Lighter Standard** | **0%** | **Não** | **Sim (fica no Standard)** | **+0,0718%** | **981 → 9,5 meses** |
| KCEX | 0,01% | Não | ❌ 403 anti-bot | +0,0518% | 1.884 → 18 meses |
| MEXC / Paradex API | 0,02% | MEXC sim | Sim | +0,0318% | ~5.000 → 48 meses |
| Aster | 0,035% | Não | Sim | +0,0018% | morta |
| Hyperliquid | 0,045% | Não | Sim | −0,018% | morta |
| Phemex | 0,06% | Não | Sim | −0,048% | morta |

**A taxa zero não é só lucro maior — é o que torna o teste viável.**
Confirmado na doc da API: *"API traders are not forced into Premium — they
default to Standard (zero fees)"*. A Paradex é a armadilha: zero no varejo,
**0,02% em quem usa API**.

## Custos de entrar/sair na Lighter (conta de 78 USDT)

- **Lighter cobra ZERO de protocolo** em depósito e saque.
- Você paga **gas da Ethereum**: ~**$4–18** para depositar, **$8–43** ida e volta.
- Em 78 USDT isso é **5–23% do capital só para entrar**. É o preço do
  experimento — com essa quantia, isso é um **teste, não um investimento**.
- **É custo único, não por trade** — não toca no edge. O trading é off-chain
  com liquidação em lote; gas só na entrada e na saída.
- Lote mínimo do ETH: **0,0001 ETH ≈ $0,19** → 78 USDT sobra folgado
  (a 2x são $156 = 0,083 ETH).

### Não espere 9,5 meses em silêncio — sinais de alerta antes disso

- **~1 mês (100 trades):** o winrate deveria estar perto de 87%. Se estiver
  abaixo de ~75%, algo está errado (execução, latência, ou o edge sumiu).
- **~3 meses (300 trades):** a média/trade deveria estar perto de +0,07%. Se
  estiver **claramente negativa**, pode parar — não vai virar.
- **Qualquer momento:** trade individual pior que −2,5% significa bug (SL = 2,5%).

⚠️ **Não conclua nada por "ainda não deu significativo"** antes dos 981 trades —
foi exatamente esse o erro de 50% de poder descrito acima.

### ✅ O spread: falso alarme (medido, resolvido)

Uma versão anterior deste arquivo alarmava que spread de 0,06% mataria tudo,
usando **0,02% como chute**. Medido de verdade no ETH perp (16/07/2026):

**Spread real: 0,00053%** (1 centavo em $1.875) — **97x abaixo** do que mataria.
**99% do edge sobrevive.** O ETH perp é líquido demais para isso ser problema.

E não é mais estimativa: o motor usa o **bid/ask real de cada instante**
(entra no ask, sai no bid), então o spread verdadeiro entra no preço de fill.

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

- **Latência de 300ms da conta Standard.** É assim que a Lighter banca a taxa
  zero: Standard tem **300ms de latência no taker** (Premium tem 0ms, mas cobra).
  Seu trailing dispara, a ordem sai 300ms depois — e ele dispara justamente
  quando o preço está correndo contra você. Com edge de 7 bps, isso importa.
  **Este é o principal risco não medido.**
- **Funding** — perp cobra funding, e na Lighter o período é de **1 hora**
  (não 8h). Não está em nenhum número aqui. A 2x pode ser relevante.
- **Slippage de verdade** — o paper preenche no bid/ask do topo do livro; ordem
  maior "anda" o livro. Com 78 USDT isso é irrelevante; com conta grande, não.
- **Viés de seleção** — esta estratégia é 1 escolhida entre 115.030 testadas.
  Só o teste prospectivo (este) fecha essa brecha.
- **Risco da própria Lighter** — DEX novo: risco de contrato, de sequencer, e
  de o modelo de taxa zero mudar.

Se passar em 9,5 meses, aí sim vale o teste com dinheiro pequeno de verdade —
que é o único juiz acima deste.
