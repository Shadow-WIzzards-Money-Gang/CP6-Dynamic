"""Etapas A e D — Monte Carlo.

A) `estimar_demanda`: sorteia N cenários (demanda, concorrente, logística) e, para cada SKU e nível
   de preço, monta a distribuição empírica da demanda semanal P(D = k), com e sem destaque.
   Esses histogramas são a entrada da Programação Dinâmica.
   Usa números aleatórios comuns: o MESMO cenário é avaliado em todos os níveis, de modo que a
   diferença entre níveis reflete o preço e não o acaso.
       Tempo O(A·N·n)   Memória O(A·N·n)   (N cenários, n SKUs, A níveis)

D) `simular`: aplica a política completa (preço da DP em função do estoque real + destaques do Greedy)
   em cenários NOVOS, semana a semana, pagando as dívidas nos vencimentos.
       Tempo O(N·T·n)   Memória O(N·T·n)   (trajetórias guardadas para os gráficos)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from questao_02.modelo import InstanciaQ2


def choque_lognormal(rng: np.random.Generator, sigma: float, forma) -> np.ndarray:
    """Fator multiplicativo com média 1: exp(σZ - σ²/2)."""
    return np.exp(sigma * rng.standard_normal(forma) - sigma**2 / 2)


def demanda_base_cenarios(inst: InstanciaQ2, rng: np.random.Generator,
                          n_cenarios: int) -> tuple[np.ndarray, np.ndarray]:
    """Demanda no nível 1.0 sem destaque, (N, n), e fator de custo logístico, (N, n)."""
    forma = (n_cenarios, inst.n)
    concorrente = inst.concorrente_base * choque_lognormal(rng, inst.sigma_concorrente, forma)
    fator_conc = (concorrente / inst.concorrente_ref) ** inst.elasticidade_concorrencia
    demanda = inst.demanda_base * fator_conc * choque_lognormal(rng, inst.sigma_demanda, forma)
    return demanda, choque_lognormal(rng, inst.sigma_logistica, forma)


@dataclass(frozen=True)
class EstatisticasDemanda:
    prob: list[np.ndarray]           # por SKU: (A, K+1) com P(D = k) sem destaque
    prob_destaque: list[np.ndarray]  # por SKU: (A, K+1) com destaque
    logistica_media: np.ndarray      # (n,)
    resumo: pd.DataFrame


def _histograma(amostras: np.ndarray) -> np.ndarray:
    """amostras (A, N) inteiras -> (A, max+1) probabilidades."""
    tamanho = int(amostras.max()) + 1
    return np.stack([np.bincount(linha, minlength=tamanho) for linha in amostras]) / amostras.shape[1]


def estimar_demanda(inst: InstanciaQ2, n_cenarios: int, rng: np.random.Generator) -> EstatisticasDemanda:
    if n_cenarios < 1:
        raise ValueError("n_cenarios deve ser positivo")
    base, fator_log = demanda_base_cenarios(inst, rng, n_cenarios)
    efeito_preco = inst.niveis[:, None] ** (-inst.elasticidade)          # (A, n)
    demanda = base[None] * efeito_preco[:, None, :]                       # (A, N, n)
    d_normal = np.rint(demanda).astype(np.int64)
    d_destaque = np.rint(demanda * (1 + inst.aumento_destaque)).astype(np.int64)

    logistica_media = inst.logistica_base * fator_log.mean(axis=0)
    margem = inst.margem_antes_logistica - logistica_media[:, None]      # (n, A)
    media = d_normal.mean(axis=1).T                                        # (n, A)
    resumo = pd.DataFrame({
        "sku": np.repeat(inst.skus, len(inst.niveis)),
        "nivel_preco": np.tile(inst.niveis, inst.n),
        "demanda_media": media.ravel(),
        "demanda_p10": np.percentile(d_normal, 10, axis=1).T.ravel(),
        "demanda_p90": np.percentile(d_normal, 90, axis=1).T.ravel(),
        "caixa_semanal_medio_sem_limite_estoque": (media * margem).ravel(),
    })
    return EstatisticasDemanda(
        prob=[_histograma(d_normal[:, :, i]) for i in range(inst.n)],
        prob_destaque=[_histograma(d_destaque[:, :, i]) for i in range(inst.n)],
        logistica_media=logistica_media,
        resumo=resumo,
    )


@dataclass(frozen=True)
class ResultadoSimulacao:
    caixa_semanal: np.ndarray    # (N, T) saldo após pagamentos da semana
    pagou: np.ndarray            # (N, D) bool — compromisso pago em dia
    saldo_vencimento: np.ndarray # (N, D) caixa disponível no vencimento, antes de pagar
    estoque: np.ndarray          # (N, T+1, n)
    nivel: np.ndarray            # (N, T, n) índice do nível de preço praticado
    receita: np.ndarray          # (N,)
    resultado_operacional: np.ndarray  # (N,) caixa gerado nas 12 semanas (antes das dívidas)

    @property
    def caixa_final(self) -> np.ndarray:
        return self.caixa_semanal[:, -1]


def simular(inst: InstanciaQ2, politica: np.ndarray, destaques: np.ndarray, custo_destaques: np.ndarray,
            n_cenarios: int, rng: np.random.Generator) -> ResultadoSimulacao:
    """politica: (n, T, S+1) nível por (SKU, semana, estoque); destaques e custos: (T, n)."""
    N, T, n = n_cenarios, inst.semanas, inst.n
    sku = np.arange(n)
    margem = inst.margem_antes_logistica                     # (n, A)
    preco = inst.preco

    estoque = np.empty((N, T + 1, n), dtype=np.int64)
    estoque[:, 0] = inst.estoque
    nivel = np.empty((N, T, n), dtype=np.int8)
    caixa_semanal = np.empty((N, T))
    saldo_vencimento = np.empty((N, len(inst.dividas)))
    caixa = np.full(N, inst.caixa_inicial)
    receita = np.zeros(N)
    operacional = np.zeros(N)

    for t in range(T):
        atual = estoque[:, t]
        a = politica[sku, t, atual]                                       # (N, n)
        base, fator_log = demanda_base_cenarios(inst, rng, N)
        demanda = np.rint(base * inst.niveis[a] ** (-inst.elasticidade)
                          * np.where(destaques[t], 1 + inst.aumento_destaque, 1.0)).astype(np.int64)
        vendas = np.minimum(atual, demanda)

        fluxo = (vendas * (margem[sku, a] - inst.logistica_base * fator_log)).sum(axis=1) - custo_destaques[t].sum()
        caixa += fluxo
        operacional += fluxo
        receita += (vendas * preco[sku, a]).sum(axis=1)
        estoque[:, t + 1] = atual - vendas
        nivel[:, t] = a

        semana = t + 1
        if semana in inst.dividas:
            saldo_vencimento[:, list(inst.dividas).index(semana)] = caixa
            caixa -= inst.dividas[semana]
        caixa_semanal[:, t] = caixa

    pagou = saldo_vencimento >= np.array(list(inst.dividas.values()))
    return ResultadoSimulacao(caixa_semanal, pagou, saldo_vencimento, estoque, nivel, receita, operacional)


def politica_constante(inst: InstanciaQ2, indice_nivel: int) -> np.ndarray:
    """Política sem reprecificação (ex.: preço atual o tempo todo), no formato (n, T, S+1)."""
    return np.full((inst.n, inst.semanas, int(inst.estoque.max()) + 1), indice_nivel, dtype=np.int64)


def indicadores(inst: InstanciaQ2, sim: ResultadoSimulacao, percentil_risco: float) -> dict:
    final = sim.caixa_final
    falha = ~sim.pagou.all(axis=1)
    estoque_final = sim.estoque[:, -1]
    deficit = np.array(list(inst.dividas.values())) - sim.saldo_vencimento     # > 0 quando falta caixa
    primeira_falha = np.where(falha, np.argmax(~sim.pagou, axis=1), -1)
    semanas = list(inst.dividas)
    return {
        **{f"prob_pagar_semana_{s}": float(sim.pagou[:, j].mean()) for j, s in enumerate(inst.dividas)},
        "prob_pagar_todas": float(sim.pagou.all(axis=1).mean()),
        "caixa_final_medio": float(final.mean()),
        f"caixa_final_p{percentil_risco:g}": float(np.percentile(final, percentil_risco)),
        "caixa_final_p50": float(np.percentile(final, 50)),
        "caixa_final_p90": float(np.percentile(final, 90)),
        "margem_media_pct": float(100 * sim.resultado_operacional.sum() / sim.receita.sum()),
        "receita_media": float(sim.receita.mean()),
        "estoque_final_medio_unidades": float(estoque_final.sum(axis=1).mean()),
        "estoque_final_medio_valor_custo": float((estoque_final * inst.custo_unitario).sum(axis=1).mean()),
        "cenarios_com_falha": int(falha.sum()),
        **{f"falhas_iniciadas_semana_{s}": int((primeira_falha == j).sum()) for j, s in enumerate(semanas)},
        "deficit_medio_no_vencimento_quando_falha": float(deficit[~sim.pagou].mean()) if falha.any() else 0.0,
    }
