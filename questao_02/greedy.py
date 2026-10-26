"""Etapa C — Greedy semanal: quais SKUs recebem o recurso promocional escasso (destaque/mídia/frete).

Destaque no SKU i na semana t: demanda × (1 + aumento_destaque), custo fixo contratado de
custo_destaque_pct × receita esperada com o destaque. Limites por semana: `max_destaques` SKUs
e `orcamento_semanal` em R$.

Benefício usa o valor da DP (o plano de preços orienta o Greedy):
    benefício = Q_t(s, a | com destaque) - Q_t(s, a | sem destaque)
Assim, vender mais agora "paga" a perda de vendas futuras (V_{t+1} com menos estoque).

Regra gulosa (semana a semana, seguindo o estoque esperado do plano):
    ordenar candidatos por benefício / custo, aceitar enquanto benefício > custo,
    houver vaga e couber no orçamento.

Complexidade (T semanas, n SKUs, K demanda máx.):
    Tempo   O(T · n · (K + log n))      Memória O(T · n)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from questao_02.dynamic_programming import PlanoDP, valor_da_acao, vendas_esperadas
from questao_02.modelo import InstanciaQ2
from questao_02.monte_carlo import EstatisticasDemanda


@dataclass(frozen=True)
class PlanoDestaques:
    destaques: np.ndarray        # (T, n) bool
    custo: np.ndarray            # (T, n) R$
    estoque_esperado: np.ndarray # (n, T+1) trajetória esperada já com os destaques
    nivel: np.ndarray            # (n, T)
    tabela: pd.DataFrame         # alocações escolhidas


def alocar_destaques(inst: InstanciaQ2, stats: EstatisticasDemanda, plano: PlanoDP) -> PlanoDestaques:
    T, n = inst.semanas, inst.n
    destaques = np.zeros((T, n), dtype=bool)
    custo = np.zeros((T, n))
    estoque = np.empty((n, T + 1), dtype=np.int64)
    nivel = np.empty((n, T), dtype=np.int64)
    estoque[:, 0] = inst.estoque
    linhas = []

    for t in range(T):
        candidatos = []
        for i, sol in enumerate(plano.solucoes):
            s = int(estoque[i, t])
            nivel[i, t] = a = sol.politica[t, s]
            if s == 0:
                continue
            m, v_seguinte = plano.margem[i, a], sol.valor[t + 1]
            beneficio = (valor_da_acao(stats.prob_destaque[i][a], m, v_seguinte, s, plano.gamma)
                         - valor_da_acao(stats.prob[i][a], m, v_seguinte, s, plano.gamma))
            gasto = inst.custo_destaque_pct * inst.preco[i, a] * vendas_esperadas(stats.prob_destaque[i][a], s)
            if beneficio > gasto > 0:
                candidatos.append((beneficio / gasto, i, beneficio, gasto))

        restante = inst.orcamento_semanal
        for razao, i, beneficio, gasto in sorted(candidatos, reverse=True):
            if destaques[t].sum() >= inst.max_destaques:
                break
            if gasto <= restante:
                restante -= gasto
                destaques[t, i], custo[t, i] = True, gasto
                linhas.append({"semana": t + 1, "sku": inst.skus[i], "nivel_preco": inst.niveis[nivel[i, t]],
                               "beneficio": round(beneficio, 2), "custo": round(gasto, 2),
                               "beneficio_por_real": round(razao, 3)})

        for i in range(n):
            s, a = int(estoque[i, t]), nivel[i, t]
            prob = stats.prob_destaque[i][a] if destaques[t, i] else stats.prob[i][a]
            estoque[i, t + 1] = s - round(vendas_esperadas(prob, s))

    return PlanoDestaques(destaques, custo, estoque, nivel, pd.DataFrame(linhas))
