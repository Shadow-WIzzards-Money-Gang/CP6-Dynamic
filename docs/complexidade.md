# Complexidade assintótica (Big O)

## Variáveis

| Símbolo | Significado | Valor na instância G01 |
|---|---|---|
| n | número de SKUs | Q1: 30 · Q2: 24 |
| A | níveis de preço permitidos | Q1: 7 (+1 opção "pausar") · Q2: 6 |
| k | SKUs do subconjunto da força bruta | 7 |
| C | combinações da força bruta = ∏ \|opções_i\| ≤ (A+1)^k | 288.000 |
| P | tamanho da população do GA | 100 (B_robusta) · 30 (A_enxuta) |
| G | gerações executadas pelo GA | ≤ 300 (parada por paciência ≈ 90–110) |
| t | tamanho do torneio | 3 |
| S_inc | passos de incentivo do Greedy Q1 | 5 |
| T | semanas | 12 |
| S | estoque máximo de um SKU (estados da DP) | 540 |
| K | maior demanda semanal observada no Monte Carlo | ≈ 102 |
| N | cenários de Monte Carlo | 2.000 (construção) · 5.000 (validação) |
| L | taxas de urgência testadas | 5 |

## Questão 1

| Algoritmo | Tempo | Memória | Trecho que domina |
|---|---|---|---|
| Pré-cálculo do modelo | O(n·A) | O(n·A) | `construir_instancia` — tabelas `caixa`, `margem`, `desconto` |
| Força Bruta (vetorizada) | O(C) | O(C) | `np.add.outer` em `forca_bruta` monta o tensor com o caixa de todas as combinações |
| Força Bruta (ingênua) | O(C·k) | O(k) | laço `itertools.product` + `caixa_total` em `forca_bruta_ingenua` |
| Algoritmo Genético | O(G·P·(n log n + t)) | O(P·n + G) | `reparar` (argsort de n chaves por indivíduo); fitness é só O(P·n) graças às tabelas |
| Greedy de incentivos | O(n·S_inc·log n) | O(n·S_inc) | `heapq` em `alocar_incentivos` (cada SKU entra na fila no máximo S_inc vezes) |
| Referência exata (`otimo_por_decomposicao`) | O(n·A + n log n) | O(n·A) | ordenação dos ganhos de desconto |

**Quando deixa de ser viável:** a força bruta cresce como (A+1)^k. Com 8 opções por SKU,
k = 7 → 2,1 milhões de combinações (no pior caso), k = 9 → 134 milhões e k = 30 → 1,2·10²⁷.
Acima de `LIMITE_COMBINACOES = 50 milhões` a função recusa a execução e o GA assume.
O GA não garante o ótimo, mas seu custo é linear em G·P e quase linear em n.

## Questão 2

| Algoritmo | Tempo | Memória | Trecho que domina |
|---|---|---|---|
| Monte Carlo — estimativa | O(A·N·n) | O(A·N·n) | tensor `demanda` (A, N, n) em `estimar_demanda` |
| Programação Dinâmica | O(n·T·S·K·A) | O(n·(T·S + S·K)) | `(gamma * valor[t+1][proximo]) @ prob.T` em `resolver_sku` |
| Greedy semanal | O(T·n·(K + log n)) | O(T·n) | `valor_da_acao` (soma sobre K) e ordenação dos candidatos |
| Monte Carlo — validação | O(N·T·n) | O(N·T·n) | laço semanal de `simular` (trajetórias guardadas para os gráficos) |
| Pipeline completo | O(A·N·n + L·(n·T·S·K·A + N·T·n)) | O(A·N·n + N·T·n) | a DP é resolvida uma vez por taxa de urgência testada |

**Por que a DP é viável e a enumeração não:** enumerar todas as sequências de preços custa
A^T = 6¹² ≈ 2,2 bilhões por SKU. A DP guarda V_{t+1}(s) para todo estoque s e reutiliza essa
tabela para todos os estados e decisões da semana t: o custo vira T·S·K·A ≈ 12·540·102·6 ≈ 4 milhões
de operações por SKU.

**Quando deixa de ser viável:** a DP por SKU escala linearmente em S e K. O limite seria um
estado conjunto (ex.: caixa total + estoques de todos os SKUs), que cresce exponencialmente com n.
Por isso o acoplamento entre SKUs (dívidas) é tratado fora da DP: pela taxa de urgência e pela
validação no Monte Carlo. Se S·K ficar grande demais (ex.: estoques de dezenas de milhares de
unidades), a alternativa é agregar o estoque em faixas (por exemplo, de 10 em 10 unidades).
