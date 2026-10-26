"""Escalabilidade da Questão 2 (três tamanhos de entrada) + otimização da DP antes/depois.

    python questao_02/benchmark.py
Gera resultados/desempenho_tamanhos.csv e resultados/otimizacao_dp.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from comum.dados import carregar_parametros, carregar_questao
from comum.medicao import medir
from questao_02.dynamic_programming import resolver_dp, resolver_sku, resolver_sku_top_down
from questao_02.greedy import alocar_destaques
from questao_02.modelo import aplicar_cenario, construir_instancia
from questao_02.monte_carlo import estimar_demanda, simular

PASTA = Path(__file__).resolve().parent

TAMANHOS = {  # nome: (máx. de SKUs, fator de estoque)
    "pequena": (8, 1.0),
    "media": (24, 1.0),
    "grande": (24, 4.0),
}


def medir_tamanhos(catalogo, config, par, seed: int) -> pd.DataFrame:
    linhas = []
    for nome, (max_skus, fator) in TAMANHOS.items():
        cat, cfg = aplicar_cenario(catalogo, config, max_skus=max_skus, estoque_fator=fator)
        inst = construir_instancia(cat, cfg, par)
        rng = np.random.default_rng(seed)
        stats, m_mc = medir("monte_carlo_estimativa", estimar_demanda, inst, par["cenarios_construcao"], rng)
        plano, m_dp = medir("dp", resolver_dp, inst, stats, 0.0)
        dest, m_gr = medir("greedy", alocar_destaques, inst, stats, plano)
        _, m_val = medir("monte_carlo_validacao", simular, inst, plano.politica, dest.destaques, dest.custo,
                         par["cenarios_validacao"], rng)
        k_max = max(p.shape[1] for p in stats.prob) - 1
        for med in (m_mc, m_dp, m_gr, m_val):
            linhas.append({"entrada": nome, "n_skus": inst.n, "estoque_max_S": int(inst.estoque.max()),
                           "demanda_max_K": k_max, **med.como_dict()})
    return pd.DataFrame(linhas)


def comparar_dp(catalogo, config, par, seed: int, n_skus: int = 6) -> pd.DataFrame:
    cat, cfg = aplicar_cenario(catalogo, config, max_skus=n_skus)
    inst = construir_instancia(cat, cfg, par)
    stats = estimar_demanda(inst, par["cenarios_construcao"], np.random.default_rng(seed))
    margem = inst.margem_antes_logistica - stats.logistica_media[:, None]

    def resolver_todos(funcao):
        return [funcao(stats.prob[i], margem[i], int(inst.estoque[i]), inst.semanas, 1.0, 0.0)
                for i in range(inst.n)]

    antes, m_antes = medir("antes_top_down_memo", resolver_todos, resolver_sku_top_down)
    depois, m_depois = medir("depois_bottom_up_numpy", resolver_todos, resolver_sku)
    iguais = all(np.allclose(a.valor, d.valor) for a, d in zip(antes, depois))
    return pd.DataFrame([
        {**m.como_dict(), "n_skus": inst.n, "mesmo_resultado": iguais,
         "aceleracao_x": m_antes.segundos / m_depois.segundos}
        for m in (m_antes, m_depois)
    ])


def main() -> None:
    par = carregar_parametros(PASTA / "parametros.json")
    catalogo, config, _ = carregar_questao(2)
    saida = PASTA / "resultados"
    saida.mkdir(exist_ok=True)

    tamanhos = medir_tamanhos(catalogo, config, par, config["seed_final"])
    otimizacao = comparar_dp(catalogo, config, par, config["seed_final"])
    tamanhos.to_csv(saida / "desempenho_tamanhos.csv", index=False)
    otimizacao.to_csv(saida / "otimizacao_dp.csv", index=False)

    pd.set_option("display.width", 140)
    print(tamanhos.round(4).to_string(index=False))
    print()
    print(otimizacao.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
