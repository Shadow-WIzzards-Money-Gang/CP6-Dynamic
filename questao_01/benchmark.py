"""Medição de desempenho da Questão 1 em entradas pequena, média e grande + otimização antes/depois.

    python questao_01/benchmark.py
Gera resultados/desempenho_tamanhos.csv e resultados/otimizacao_forca_bruta.csv.
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
from questao_01.brute_force import forca_bruta, forca_bruta_ingenua
from questao_01.genetic_algorithm import ConfigGA, executar_ga
from questao_01.greedy import alocar_incentivos
from questao_01.main import escolher_subconjunto
from questao_01.modelo import InstanciaQ1, construir_instancia

PASTA = Path(__file__).resolve().parent


def ampliar(catalogo: pd.DataFrame, vezes: int) -> pd.DataFrame:
    """Catálogo sintético maior: cópias do original com SKUs renomeados (só para medir escala)."""
    copias = [catalogo.assign(sku=catalogo["sku"] + f"-c{c}") for c in range(vezes)]
    return pd.concat(copias, ignore_index=True)


def medir_tamanhos(inst_base: InstanciaQ1, catalogo, config, param, seed: int) -> pd.DataFrame:
    cfg = ConfigGA(**param["ga"]["principal"])
    passos = param["incentivo"]["passos"]
    fracao = param["fracao_max_skus_com_desconto"]
    linhas = []

    for k in (5, 6, 7):
        sub = inst_base.subinstancia(escolher_subconjunto(inst_base, k))
        res, med = medir("forca_bruta", forca_bruta, sub)
        linhas.append({"algoritmo": "Forca Bruta", "n": k, "combinacoes": res.combinacoes,
                       "segundos": med.segundos, "pico_mb": med.pico_mb})

    for nome, inst in [("pequena", inst_base.subinstancia(np.arange(10))), ("media", inst_base),
                       ("grande", construir_instancia(ampliar(catalogo, 4), config, fracao))]:
        ga, med_ga = medir("ga", executar_ga, inst, cfg, seed)
        _, med_gr = medir("greedy", alocar_incentivos, inst, ga.politica, passos)
        linhas += [
            {"algoritmo": "GA", "entrada": nome, "n": inst.n, "geracoes": ga.geracoes,
             "avaliacoes": ga.avaliacoes, "segundos": med_ga.segundos, "pico_mb": med_ga.pico_mb},
            {"algoritmo": "Greedy", "entrada": nome, "n": inst.n,
             "segundos": med_gr.segundos, "pico_mb": med_gr.pico_mb},
        ]
    return pd.DataFrame(linhas)


def comparar_forca_bruta(inst: InstanciaQ1) -> pd.DataFrame:
    linhas = []
    for k in (4, 5, 6):
        sub = inst.subinstancia(escolher_subconjunto(inst, k))
        for versao, funcao in [("antes_ingenua", forca_bruta_ingenua), ("depois_vetorizada", forca_bruta)]:
            res, med = medir(versao, funcao, sub)
            linhas.append({"k": k, "versao": versao, "combinacoes": res.combinacoes, "caixa": res.caixa,
                           "segundos": med.segundos, "pico_mb": med.pico_mb})
    tabela = pd.DataFrame(linhas)
    tempos = tabela.pivot(index="k", columns="versao", values="segundos")
    tabela["aceleracao_x"] = tabela["k"].map(tempos["antes_ingenua"] / tempos["depois_vetorizada"])
    return tabela


def main() -> None:
    param = carregar_parametros(PASTA / "parametros.json")
    catalogo, config, _ = carregar_questao(1)
    inst = construir_instancia(catalogo, config, param["fracao_max_skus_com_desconto"])
    saida = PASTA / "resultados"
    saida.mkdir(exist_ok=True)

    tamanhos = medir_tamanhos(inst, catalogo, config, param, config["seed_final"])
    otimizacao = comparar_forca_bruta(inst)
    tamanhos.to_csv(saida / "desempenho_tamanhos.csv", index=False)
    otimizacao.to_csv(saida / "otimizacao_forca_bruta.csv", index=False)

    pd.set_option("display.width", 140)
    print(tamanhos.round(4).to_string(index=False))
    print()
    print(otimizacao.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
