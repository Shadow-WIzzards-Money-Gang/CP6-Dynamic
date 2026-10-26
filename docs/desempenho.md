# Desempenho medido

Medições com `time.perf_counter()` (tempo de parede) e `tracemalloc` (pico de memória alocada na
chamada), feitas pela função `comum.medicao.medir`. Ambiente de referência: Python 3.13,
NumPy 2.5, pandas 3.0. Os valores mudam de máquina para máquina; para reproduzir:

```bash
python questao_01/benchmark.py   # -> questao_01/resultados/desempenho_tamanhos.csv, otimizacao_forca_bruta.csv
python questao_02/benchmark.py   # -> questao_02/resultados/desempenho_tamanhos.csv, otimizacao_dp.csv
```

## Questão 1 — tamanhos de entrada

| Algoritmo | Entrada | n | Trabalho | Tempo (s) | Pico (MB) |
|---|---|---|---|---|---|
| Força Bruta | pequena | 5 | 7.200 combinações | 0,0006 | 0,18 |
| Força Bruta | média | 6 | 57.600 combinações | 0,0010 | 0,67 |
| Força Bruta | grande | 7 | 288.000 combinações | 0,0035 | 2,82 |
| GA (B_robusta) | pequena | 10 | 70 gerações · 7.000 avaliações | 0,061 | 0,08 |
| GA (B_robusta) | média (oficial) | 30 | 108 gerações · 10.800 avaliações | 0,228 | 0,21 |
| GA (B_robusta) | grande (catálogo ×4) | 120 | 269 gerações · 26.900 avaliações | 1,990 | 0,78 |
| Greedy | pequena | 10 | — | 0,0020 | 0,01 |
| Greedy | média | 30 | — | 0,0023 | 0,01 |
| Greedy | grande | 120 | — | 0,0037 | 0,04 |

Leitura: a memória da força bruta cresce junto com C (tensor de todas as combinações), ×5 de k=6 para
k=7. O GA cresce mais que linearmente com n porque, com mais genes, precisa de mais gerações até
estabilizar (70 → 269). O Greedy é desprezível frente aos demais.

## Questão 2 — tamanhos de entrada

| Entrada | n | S | K | Etapa | Tempo (s) | Pico (MB) |
|---|---|---|---|---|---|---|
| pequena | 8 | 389 | 105 | Monte Carlo estimativa (2.000) | 0,010 | 3,2 |
| | | | | DP | 0,008 | 1,1 |
| | | | | Greedy | 0,009 | 0,01 |
| | | | | Monte Carlo validação (5.000) | 0,053 | 8,2 |
| média (oficial) | 24 | 540 | 102 | Monte Carlo estimativa | 0,023 | 9,6 |
| | | | | DP | 0,019 | 2,1 |
| | | | | Greedy | 0,028 | 0,02 |
| | | | | Monte Carlo validação | 0,138 | 23,2 |
| grande (estoque ×4) | 24 | 2.160 | 102 | Monte Carlo estimativa | 0,019 | 9,6 |
| | | | | DP | 0,044 | 8,3 |
| | | | | Greedy | 0,028 | 0,02 |
| | | | | Monte Carlo validação | 0,131 | 23,2 |

Leitura: multiplicar o estoque por 4 (S: 540 → 2.160) multiplica a memória da DP por ≈ 4 (2,1 → 8,3 MB),
como previsto por O(T·S + S·K), e não altera o Monte Carlo (que depende de N e n, não de S).
Triplicar n (8 → 24) multiplica a memória do Monte Carlo por ≈ 3, como previsto por O(N·n).

## Otimizações demonstradas (antes × depois)

### Q1 — Força Bruta: laço Python → tensor NumPy (`np.add.outer`)

| k | Combinações | Antes (s) | Depois (s) | Aceleração |
|---|---|---|---|---|
| 4 | 1.200 | 0,0178 | 0,0003 | ≈ 55× |
| 5 | 7.200 | 0,1052 | 0,0005 | ≈ 232× |
| 6 | 57.600 | 0,7364 | 0,0009 | ≈ 840× |

Mesmo ótimo nas duas versões (verificado em `tests/test_questao_01.py`). Troca de recurso:
a versão ingênua usa O(k) de memória; a vetorizada usa O(C) (0,67 MB para k=6).

### Q2 — DP: top-down com memoização em Python → bottom-up vetorizado

| Versão | Tempo (s) | Pico (MB) | Mesmo resultado |
|---|---|---|---|
| antes: recursão + `lru_cache` | 2,165 | 7,4 | sim |
| depois: bottom-up com NumPy | 0,005 | 1,2 | sim |

≈ 450× mais rápida e com menos memória (o cache do `lru_cache` guarda cada (t, s, a) como objeto
Python; a versão vetorizada guarda só as matrizes V e política).

Outra otimização estrutural, presente desde a primeira versão: na Q1 o caixa de cada (SKU, nível)
é pré-calculado uma vez, então avaliar um indivíduo do GA custa O(n) em vez de recalcular o modelo.
