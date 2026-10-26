"""Etapa C — Greedy: distribui o orçamento de incentivo comercial sobre a política do GA.

Incentivo (cupom/cashback/frete subsidiado) = s% do preço por unidade vendida, em passos discretos.
O cliente percebe o preço p·(1-s), então a demanda sobe pela mesma elasticidade; o varejista
continua recebendo p (a diferença sai do orçamento promocional):

    unidades(s) = min(estoque, floor(demanda_base · (nivel·(1-s))^(-e)))
    ganho_caixa = Δunidades · margem_unitária      custo = s · p · unidades(s)

Regra gulosa: sempre aplicar o próximo passo de incentivo com MAIOR caixa ganho por real
investido (fila de prioridade), enquanto houver orçamento. Se o orçamento NÃO for uma verba
separada (sai do próprio caixa), só são aceitos passos que se pagam (ganho > custo).

Complexidade (n SKUs, S passos de incentivo):
    Tempo:   O(n·S) para tabelar + O(n·S·log n) para a fila de prioridade.
    Memória: O(n·S).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from questao_01.modelo import InstanciaQ1


@dataclass(frozen=True)
class ResultadoGreedy:
    incentivo: np.ndarray       # (n,) fração do preço concedida a cada SKU
    unidades: np.ndarray        # (n,)
    caixa_operacional: float    # caixa das vendas (sem descontar o incentivo)
    custo_incentivo: float
    tabela: pd.DataFrame        # antes/depois por SKU


def _tabelar(inst: InstanciaQ1, politica: np.ndarray, passos: np.ndarray):
    """Unidades, caixa e custo de cada SKU para cada passo de incentivo: matrizes (n, S+1)."""
    idx = np.arange(inst.n)
    nivel = np.append(inst.niveis, 0.0)[politica][:, None]   # pausados: nível 0 -> sem venda
    margem = inst.margem[idx, politica][:, None]
    ativo = (politica != inst.pausar)[:, None]

    with np.errstate(divide="ignore"):
        demanda = inst.demanda_base[:, None] * (nivel * (1 - passos)) ** (-inst.elasticidade[:, None])
    unidades = np.where(ativo, np.minimum(inst.estoque[:, None], np.floor(demanda)), 0.0)
    caixa = unidades * margem
    custo = passos * (inst.preco_atual[:, None] * nivel) * unidades
    return unidades, caixa, custo


def alocar_incentivos(inst: InstanciaQ1, politica: np.ndarray, passos: list[float],
                      verba_separada: bool = True) -> ResultadoGreedy:
    if inst.orcamento_promocional < 0:
        raise ValueError("Orçamento promocional negativo.")
    passos = np.concatenate([[0.0], np.sort(np.asarray(passos, dtype=float))])
    if passos[-1] >= 1:
        raise ValueError("Passos de incentivo devem ser frações < 1.")

    unidades, caixa, custo = _tabelar(inst, politica, passos)
    passo_atual = np.zeros(inst.n, dtype=int)
    restante = inst.orcamento_promocional

    def candidato(i: int):
        """Próximo passo do SKU i como entrada da fila (-razão, i), ou None se não compensa."""
        k = passo_atual[i] + 1
        if k >= len(passos):
            return None
        ganho, gasto = caixa[i, k] - caixa[i, k - 1], custo[i, k] - custo[i, k - 1]
        if ganho <= 0 or gasto <= 0 or (not verba_separada and ganho <= gasto):
            return None
        return (-ganho / gasto, i)

    fila = [c for i in range(inst.n) if (c := candidato(i))]
    heapq.heapify(fila)
    while fila:
        _, i = heapq.heappop(fila)
        k = passo_atual[i] + 1
        gasto = custo[i, k] - custo[i, k - 1]
        if gasto > restante:
            continue                      # passo não cabe; os seguintes desse SKU custam ainda mais
        restante -= gasto
        passo_atual[i] = k
        if c := candidato(i):
            heapq.heappush(fila, c)

    idx = np.arange(inst.n)
    final = (idx, passo_atual)
    tabela = pd.DataFrame({
        "sku": inst.skus,
        "incentivo_pct": passos[passo_atual] * 100,
        "unidades_antes": unidades[:, 0],
        "unidades_depois": unidades[final],
        "caixa_antes": np.round(caixa[:, 0], 2),
        "caixa_depois": np.round(caixa[final], 2),
        "custo_incentivo": np.round(custo[final], 2),
    })
    return ResultadoGreedy(
        incentivo=passos[passo_atual],
        unidades=unidades[final],
        caixa_operacional=float(caixa[final].sum()),
        custo_incentivo=float(custo[final].sum()),
        tabela=tabela,
    )
