"""Etapa B — Programação Dinâmica: política semanal de preços por SKU.

Formulação (para cada SKU, independentemente):
    Estado      (t, s): semana t ∈ {0..T-1} e estoque restante s ∈ {0..S}
    Decisão     a: nível de preço da semana
    Transição   s' = s - min(s, D),  D ~ P_a (histograma estimado pelo Monte Carlo)
    Recompensa  r = min(s, D) · m_a,  m_a = preco_a·(1-taxa) - custo - E[logística]
    Objetivo    V_t(s) = max_a Σ_k P_a(k) · [ min(s,k)·m_a + γ · V_{t+1}(s - min(s,k)) ]
    Terminal    V_T(s) = s · valor_residual   (estoque que sobra ainda vale algo, mas menos)

γ = 1 / (1 + taxa_urgencia): com urgência > 0, um real que entra cedo vale mais do que um real no
fim — empurra caixa para antes dos vencimentos (semanas 4 e 8). A taxa é escolhida por fora
(main.py) pela que maximiza a probabilidade de pagar as três dívidas no Monte Carlo de construção.

Reutilização de subproblemas: a tabela V_{t+1}(·) é calculada UMA vez e reutilizada por todos os
estados (s) e decisões (a) da semana t. Sem a tabela, cada caminho de decisões seria recalculado:
A^T sequências (6^12 ≈ 2,2 bilhões) por SKU.

Complexidade por SKU (T semanas, S estoque máx., A níveis, K demanda máx.):
    Tempo   O(T · S · K · A)     Memória O(T · S + S · K)
Total: n vezes isso. A versão `resolver_sku_top_down` (recursão + memoização) dá o mesmo resultado
e é usada como "antes" na comparação de desempenho.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from questao_02.modelo import InstanciaQ2
from questao_02.monte_carlo import EstatisticasDemanda


@dataclass(frozen=True)
class SolucaoSKU:
    valor: np.ndarray             # (T+1, S+1)  V_t(s)
    politica: np.ndarray          # (T, S+1)    argmax a
    vendas_esperadas: np.ndarray  # (S+1, A)    E[min(s, D_a)]


def resolver_sku(prob: np.ndarray, margem: np.ndarray, estoque_max: int, semanas: int,
                 gamma: float, valor_residual: float) -> SolucaoSKU:
    """Bottom-up vetorizado com NumPy."""
    estados = np.arange(estoque_max + 1)[:, None]                  # (S+1, 1)
    vendas = np.minimum(estados, np.arange(prob.shape[1]))          # (S+1, K+1)
    proximo = estados - vendas                                      # (S+1, K+1)
    vendas_esperadas = vendas @ prob.T                              # (S+1, A)

    valor = np.empty((semanas + 1, estoque_max + 1))
    politica = np.empty((semanas, estoque_max + 1), dtype=np.int64)
    valor[semanas] = estados.ravel() * valor_residual
    for t in range(semanas - 1, -1, -1):
        futuro_esperado = (gamma * valor[t + 1][proximo]) @ prob.T  # (S+1, A) — reusa V_{t+1}
        q = vendas_esperadas * margem + futuro_esperado
        politica[t] = q.argmax(axis=1)
        valor[t] = q[np.arange(estoque_max + 1), politica[t]]
    return SolucaoSKU(valor, politica, vendas_esperadas)


def resolver_sku_top_down(prob: np.ndarray, margem: np.ndarray, estoque_max: int, semanas: int,
                          gamma: float, valor_residual: float) -> SolucaoSKU:
    """Top-down com memoização (lru_cache) em Python puro — mesma recorrência, sem vetorização."""
    suporte = [(k, p) for k, p in enumerate(prob.T.tolist())]      # [(k, [P_a(k) para cada a])]
    niveis = range(len(margem))
    margem = margem.tolist()

    @lru_cache(maxsize=None)
    def q(t: int, s: int, a: int) -> float:
        total = 0.0
        for k, p in suporte:
            if p[a]:
                v = min(s, k)
                total += p[a] * (v * margem[a] + gamma * valor(t + 1, s - v))
        return total

    @lru_cache(maxsize=None)
    def valor(t: int, s: int) -> float:
        if t == semanas:
            return s * valor_residual
        return max(q(t, s, a) for a in niveis)

    v = np.array([[valor(t, s) for s in range(estoque_max + 1)] for t in range(semanas + 1)])
    pi = np.array([[max(niveis, key=lambda a: q(t, s, a)) for s in range(estoque_max + 1)]
                   for t in range(semanas)])
    estados = np.arange(estoque_max + 1)[:, None]
    vendas_esperadas = np.minimum(estados, np.arange(prob.shape[1])) @ prob.T
    return SolucaoSKU(v, pi, vendas_esperadas)


@dataclass(frozen=True)
class PlanoDP:
    solucoes: list[SolucaoSKU]
    margem: np.ndarray            # (n, A) margem esperada usada pela DP
    gamma: float
    politica: np.ndarray          # (n, T, S_max+1) — formato usado pela simulação

    def trajetoria_esperada(self, inst: InstanciaQ2) -> tuple[np.ndarray, np.ndarray]:
        """Caminho ótimo reconstruído seguindo a política a partir do estoque inicial, com vendas
        iguais às esperadas: estoque (n, T+1) e nível escolhido (n, T)."""
        estoque = np.empty((inst.n, inst.semanas + 1), dtype=np.int64)
        nivel = np.empty((inst.n, inst.semanas), dtype=np.int64)
        estoque[:, 0] = inst.estoque
        for i, sol in enumerate(self.solucoes):
            for t in range(inst.semanas):
                s = estoque[i, t]
                nivel[i, t] = a = sol.politica[t, s]
                estoque[i, t + 1] = s - round(sol.vendas_esperadas[s, a])
        return estoque, nivel


def vendas_esperadas(prob_a: np.ndarray, estoque: int) -> float:
    return float(np.minimum(estoque, np.arange(len(prob_a))) @ prob_a)


def valor_da_acao(prob_a: np.ndarray, margem_a: float, valor_seguinte: np.ndarray,
                  estoque: int, gamma: float) -> float:
    """Q_t(s, a) = Σ_k P_a(k)·[min(s,k)·m_a + γ·V_{t+1}(s - min(s,k))] para um único estado."""
    vendas = np.minimum(estoque, np.arange(len(prob_a)))
    return float(prob_a @ (vendas * margem_a + gamma * valor_seguinte[estoque - vendas]))


def resolver_dp(inst: InstanciaQ2, stats: EstatisticasDemanda, taxa_urgencia: float = 0.0) -> PlanoDP:
    if taxa_urgencia < 0:
        raise ValueError("taxa_urgencia deve ser >= 0")
    gamma = 1 / (1 + taxa_urgencia)
    margem = inst.margem_antes_logistica - stats.logistica_media[:, None]
    residual = inst.fator_valor_residual * np.maximum(margem[:, inst.nivel_atual], 0)

    solucoes = [
        resolver_sku(stats.prob[i], margem[i], int(inst.estoque[i]), inst.semanas, gamma, residual[i])
        for i in range(inst.n)
    ]
    politica = np.zeros((inst.n, inst.semanas, int(inst.estoque.max()) + 1), dtype=np.int64)
    for i, sol in enumerate(solucoes):
        politica[i, :, : sol.politica.shape[1]] = sol.politica
    return PlanoDP(solucoes, margem, gamma, politica)
