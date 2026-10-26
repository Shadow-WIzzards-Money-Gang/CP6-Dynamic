# Decisões de modelagem

Cada decisão abaixo é uma suposição do grupo e pode ser alterada por `parametros.json` ou pela
linha de comando, sem mudar código.

## Comuns às duas questões

- **Caixa inclui o custo unitário.** O caixa de uma venda é `preço·(1 - taxa) - custo_unitario - logística`,
  a mesma convenção que o notebook oficial usa para calibrar metas e dívidas. Interpretação: o
  custo do produto é pago ao fornecedor/reposto à medida que vende.
- **Demanda com elasticidade constante:** `demanda = base · nível^(-elasticidade)`. É o modelo
  mais simples coerente com o campo `elasticidade` do catálogo.
- **Integridade:** antes de qualquer cálculo, os hashes SHA-256 do `manifesto_dados.json` são
  conferidos (`comum/dados.py`). Arquivo alterado → erro.
- **Reprodutibilidade:** toda aleatoriedade usa `numpy.random.default_rng` com seed explícita
  (padrão: `seed_final`). Na Q2, `SeedSequence(seed).spawn(3)` gera três fluxos independentes:
  estimativa, construção e validação.

## Questão 1

| Decisão | Valor padrão | Motivo |
|---|---|---|
| Piso de margem por SKU | `margem_minima_percentual` do config | "respeitar piso de margem": níveis abaixo do piso são proibidos para aquele SKU |
| Opção "pausar venda" | sempre disponível | 3 SKUs (Q1-004, Q1-026, Q1-028) não têm nenhum nível que respeite o piso; também modela "SKU indisponível" |
| Desconto indiscriminado | no máximo 30% dos SKUs com nível < 1,0 | traduz "evitar política de descontos indiscriminada" em restrição verificável |
| Subconjunto da força bruta | 7 SKUs de maior receita atual | ≥ 6 exigido; são os SKUs mais relevantes financeiramente |
| Incentivo (Greedy) | 2%, 4%, …, 10% do preço por unidade | cupom/cashback/frete: o cliente percebe preço menor, o varejista recebe o preço cheio e a diferença sai do orçamento |
| Orçamento promocional | verba separada (`orcamento_e_verba_separada: true`) | o orçamento é uma verba já destinada a incentivos; com `false` o Greedy só aceita passos em que o ganho de caixa supera o custo |

**Configurações do GA comparadas** (5 seeds cada):

| | A_enxuta | B_robusta (principal) |
|---|---|---|
| População | 30 | 100 |
| Seleção | torneio 2 (pressão baixa) | torneio 3 |
| Crossover | uniforme, 0,8 | uniforme, 0,9 |
| Mutação | 1/n por gene | 1/n por gene |
| Elitismo | 1 | 2 |
| Parada | 100 gerações ou 25 sem melhora | 300 gerações ou 60 sem melhora |

Justificativas: crossover uniforme porque os genes (SKUs) não têm ordem nem blocos a preservar;
mutação 1/n mantém em média um gene alterado por filho (explora sem destruir); torneio não depende
da escala do fitness (caixa em centenas de milhares de reais); elitismo garante que o melhor nunca
piora; paciência evita gastar gerações depois da convergência. Resultado: B atingiu o ótimo nas
5 seeds (gap 0%); A, ≈ 3× mais barata em avaliações, teve gap médio de 0,17% e parou abaixo
do ótimo em 3 das 5 seeds (convergência prematura: população pequena + paciência curta).

**Achado da instância G01:** a demanda é pouco elástica (elasticidade média ≈ 1,4) e as margens
são apertadas, então o ótimo sobe o preço para o teto da grade (1,10) em 27 SKUs e pausa 3.
Descontos não compensam, e a restrição de 30% nem chega a ser ativa. O teto da grade é a
restrição que limita o caixa. Com os SKUs independentes, o ótimo exato pode ser calculado por
decomposição (`otimo_por_decomposicao`), o que usamos como referência para medir o gap do GA no
catálogo completo. O GA continua necessário para restrições que acoplam os SKUs de forma não
decomponível (ex.: margem global mínima, capacidade logística).

## Questão 2

| Decisão | Valor padrão | Motivo |
|---|---|---|
| Choques | LogNormal com média 1 e σ do config | multiplicativos, sempre positivos, sem viés na média |
| Efeito do concorrente | `(preço_conc / preço_conc_ref)^1,0` | concorrente 8% mais barato → demanda cai ≈ 8% |
| Números aleatórios comuns | mesmo cenário em todos os níveis | a diferença entre níveis reflete o preço, não o acaso |
| Estado da DP | (semana, estoque) por SKU | o estado conjunto (caixa + 24 estoques) é intratável; o acoplamento via dívidas fica na taxa de urgência e na validação |
| Valor residual | 50% da margem atual por unidade que sobra | estoque final ainda vale algo, mas menos que caixa na mão |
| Taxa de urgência λ (γ = 1/(1+λ)) | escolhida entre 0; 0,02; 0,05; 0,10; 0,20 | maior λ antecipa caixa; escolhe-se a que maximiza P(pagar as 3 dívidas) nos 2.000 cenários de construção (desempate: caixa P10) |
| Destaque (Greedy) | +25% de demanda, custo 6% da receita esperada, máx. 4 por semana | recurso escasso semanal; o custo sai do caixa da semana |
| Benefício do destaque | diferença de valor da DP com e sem destaque | considera que vender mais agora reduz o estoque futuro |
| Percentil de risco | P10 do caixa final | coerente com `limite_risco_falha = 0,10` |
| Falha | caixa no vencimento < valor da parcela | a parcela é descontada mesmo assim (saldo negativo = atraso que se acumula) |

**Achado da instância G01:** com o plano DP + Greedy, a probabilidade de pagar as três dívidas
nos 5.000 cenários de validação é 100%, com caixa final médio de R$ 245 mil (P10 = R$ 240 mil),
contra R$ 84 mil mantendo o preço atual fixo. Para mostrar que a cadeia de algoritmos faz
diferença quando há risco, rodamos um **teste de estresse** (dívidas ×1,77, concorrente −8%,
frete +15%), salvo em `questao_02/resultados/cenario_estresse/`:

| Política | P(pagar as 3) | Caixa final médio | Primeira falha |
|---|---|---|---|
| Preço atual fixo | 0% | −R$ 153,9 mil | semana 8 (233) e 12 (4.767) |
| DP sem Greedy | 2,9% | −R$ 6,4 mil | semana 12 |
| DP + Greedy | 44,0% | −R$ 0,6 mil | semana 12 |

No estresse, a taxa de urgência escolhida foi 0,10 (P = 43% na construção, contra 28% com λ = 0).
