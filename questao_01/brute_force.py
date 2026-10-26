"""Etapa A — Força Bruta: ótimo exato em um subconjunto pequeno de SKUs (benchmark do GA).

Espaço de busca: C = prod(|opções_i|) <= (A+1)^k, com k SKUs e A níveis de preço (+ pausar).
    Tempo:   O(C)  (versão vetorizada; cada combinação custa O(1) amortizado via somas acumuladas)
             O(C·k) na versão ingênua, que reavalia a política inteira a cada combinação.
    Memória: O(C)  na vetorizada (tensor com o caixa de todas as combinações); O(k) na ingênua.
Com A+1 = 8 opções: k=7 -> 2,1 mi combinações; k=9 -> 134 mi (deixa de ser viável -> GA).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from questao_01.modelo import InstanciaQ1

LIMITE_COMBINACOES = 50_000_000


@dataclass(frozen=True)
class ResultadoBF:
    politica: np.ndarray
    caixa: float
    combinacoes: int


def contar_combinacoes(inst: InstanciaQ1) -> int:
    return math.prod(len(op) for op in inst.opcoes)


def _checar_tamanho(inst: InstanciaQ1) -> int:
    total = contar_combinacoes(inst)
    if total > LIMITE_COMBINACOES:
        raise ValueError(f"{total:,} combinações: força bruta inviável, use o algoritmo genético.")
    return total


def forca_bruta(inst: InstanciaQ1) -> ResultadoBF:
    """Versão otimizada: monta, por somas externas (np.add.outer), um tensor com o caixa e o
    número de descontos de TODAS as combinações; depois descarta as inviáveis e pega o máximo."""
    total = _checar_tamanho(inst)
    opcoes = inst.opcoes

    caixa = np.zeros(())
    descontos = np.zeros((), dtype=np.int8)
    for i, op in enumerate(opcoes):
        caixa = np.add.outer(caixa, inst.caixa[i, op])
        descontos = np.add.outer(descontos, inst.desconto[i, op].astype(np.int8))

    caixa[descontos > inst.max_descontos] = -np.inf
    melhor = np.unravel_index(int(np.argmax(caixa)), caixa.shape)
    politica = np.array([op[j] for op, j in zip(opcoes, melhor)])
    return ResultadoBF(politica, float(caixa[melhor]), total)


def forca_bruta_ingenua(inst: InstanciaQ1) -> ResultadoBF:
    """Versão 'antes da otimização': percorre itertools.product e avalia cada política em Python."""
    total = _checar_tamanho(inst)
    melhor_politica, melhor_caixa = None, -math.inf
    for combinacao in itertools.product(*inst.opcoes):
        politica = np.array(combinacao)
        if inst.n_descontos(politica) > inst.max_descontos:
            continue
        caixa = inst.caixa_total(politica)
        if caixa > melhor_caixa:
            melhor_politica, melhor_caixa = politica, caixa
    return ResultadoBF(melhor_politica, float(melhor_caixa), total)
