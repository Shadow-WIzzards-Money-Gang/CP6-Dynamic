"""Questão 1 — Operação Caixa 90: Força Bruta -> Algoritmo Genético -> Greedy.

Fluxo de dados:
    1. Força Bruta resolve um subconjunto de k SKUs e serve de gabarito para o GA (gap %).
    2. GA (configuração principal e de comparação) gera a política de preços do catálogo completo.
    3. A política do GA alimenta o Greedy, que distribui o orçamento de incentivo comercial.

Uso:
    python questao_01/main.py                       # instância oficial, seed_final do grupo
    python questao_01/main.py --seed 14 --modo completo
    python questao_01/main.py --meta-fator 1.10 --orcamento-fator 0.75 --remover-sku Q1-012
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.dados import carregar_parametros, carregar_questao, salvar_json
from comum.medicao import medir
from questao_01.brute_force import forca_bruta
from questao_01.genetic_algorithm import ConfigGA, executar_ga
from questao_01.greedy import alocar_incentivos
from questao_01.modelo import (InstanciaQ1, aplicar_cenario, construir_instancia,
                               otimo_por_decomposicao)

PASTA = Path(__file__).resolve().parent


def ler_argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grupo", default="G01")
    p.add_argument("--dados", type=Path, help="pasta com os arquivos oficiais (padrão: data/<grupo>)")
    p.add_argument("--parametros", type=Path, default=PASTA / "parametros.json")
    p.add_argument("--saida", type=Path, default=PASTA / "resultados")
    p.add_argument("--seed", type=int, help="seed dos algoritmos (padrão: seed_final da instância)")
    p.add_argument("--modo", choices=["rapido", "completo"], default="completo",
                   help="rapido = 1 repetição do GA; completo = repetições do parametros.json")
    cenario = p.add_argument_group("cenário-surpresa")
    cenario.add_argument("--meta-fator", type=float, default=1.0, help="ex.: 1.10 = meta de caixa +10%%")
    cenario.add_argument("--orcamento-fator", type=float, default=1.0, help="ex.: 0.75 = orçamento -25%%")
    cenario.add_argument("--frete-fator", type=float, default=1.0, help="ex.: 1.15 = custo logístico +15%%")
    cenario.add_argument("--remover-sku", nargs="*", default=[], help="SKUs indisponíveis")
    cenario.add_argument("--fracao-desconto", type=float, help="fração máxima de SKUs com desconto")
    cenario.add_argument("--subconjunto", type=int, help="k SKUs da força bruta")
    ga = p.add_argument_group("algoritmo genético (configuração principal)")
    ga.add_argument("--populacao", type=int)
    ga.add_argument("--mutacao", type=float)
    ga.add_argument("--geracoes", type=int)
    return p.parse_args(argv)


def configs_ga(param: dict, args: argparse.Namespace) -> tuple[ConfigGA, ConfigGA]:
    principal = ConfigGA(**param["ga"]["principal"])
    ajustes = {"populacao": args.populacao, "taxa_mutacao": args.mutacao, "geracoes": args.geracoes}
    principal = replace(principal, **{k: v for k, v in ajustes.items() if v is not None})
    return principal, ConfigGA(**param["ga"]["comparacao"])


def gap_percentual(otimo: float, obtido: float) -> float:
    return 100 * (otimo - obtido) / abs(otimo) if otimo else 0.0


def escolher_subconjunto(inst: InstanciaQ1, k: int) -> np.ndarray:
    """Os k SKUs de maior receita no preço atual: os mais relevantes financeiramente."""
    if not 1 <= k <= inst.n:
        raise ValueError(f"subconjunto deve ter entre 1 e {inst.n} SKUs")
    receita = inst.receita[np.arange(inst.n), inst.politica_atual()]
    return np.sort(np.argsort(-receita)[:k])


def etapa_bf_vs_ga(inst: InstanciaQ1, k: int, cfgs: tuple[ConfigGA, ...], seeds: list[int]):
    sub = inst.subinstancia(escolher_subconjunto(inst, k))
    bf, med_bf = medir("forca_bruta", forca_bruta, sub)
    linhas = [{"algoritmo": "Forca Bruta", "config": "-", "seed": "-", "caixa": bf.caixa, "gap_pct": 0.0,
               "avaliacoes": bf.combinacoes, "segundos": med_bf.segundos, "pico_mb": med_bf.pico_mb}]
    for cfg in cfgs:
        for seed in seeds:
            ga, med = medir("ga", executar_ga, sub, cfg, seed)
            linhas.append({"algoritmo": "GA", "config": cfg.nome, "seed": seed, "caixa": ga.caixa,
                           "gap_pct": gap_percentual(bf.caixa, ga.caixa), "avaliacoes": ga.avaliacoes,
                           "segundos": med.segundos, "pico_mb": med.pico_mb})
    return sub, bf, pd.DataFrame(linhas)


def etapa_ga_completo(inst: InstanciaQ1, cfgs: tuple[ConfigGA, ...], seeds: list[int]):
    referencia = inst.caixa_total(otimo_por_decomposicao(inst))
    linhas, execucoes = [], {}
    for cfg in cfgs:
        for seed in seeds:
            ga, med = medir("ga", executar_ga, inst, cfg, seed)
            execucoes[(cfg.nome, seed)] = ga
            linhas.append({"config": cfg.nome, "seed": seed, "populacao": cfg.populacao,
                           "caixa": ga.caixa, "gap_pct_vs_exato": gap_percentual(referencia, ga.caixa),
                           "geracoes": ga.geracoes, "avaliacoes": ga.avaliacoes,
                           "segundos": med.segundos, "pico_mb": med.pico_mb})
    return execucoes, pd.DataFrame(linhas), float(referencia)


def resumo_final(inst: InstanciaQ1, politica: np.ndarray, greedy, verba_separada: bool) -> dict:
    preco = inst.preco_atual * np.append(inst.niveis, 0.0)[politica]
    receita = float((greedy.unidades * preco).sum())
    caixa_divida = greedy.caixa_operacional - (0.0 if verba_separada else greedy.custo_incentivo)
    restante = inst.estoque - greedy.unidades
    return {
        "caixa_operacional": round(greedy.caixa_operacional, 2),
        "incentivo_utilizado": round(greedy.custo_incentivo, 2),
        "orcamento_promocional": round(inst.orcamento_promocional, 2),
        "uso_orcamento_pct": round(100 * greedy.custo_incentivo / inst.orcamento_promocional, 2)
        if inst.orcamento_promocional else 0.0,
        "caixa_disponivel_para_divida": round(caixa_divida, 2),
        "meta_caixa_90d": round(inst.meta_caixa, 2),
        "meta_atingida": bool(caixa_divida >= inst.meta_caixa),
        "folga_ou_insuficiencia": round(caixa_divida - inst.meta_caixa, 2),
        "divida_total_90d": round(inst.divida, 2),
        "cobertura_divida_pct": round(100 * caixa_divida / inst.divida, 2),
        "margem_percentual": round(100 * greedy.caixa_operacional / receita, 2) if receita else 0.0,
        "margem_minima_pct": round(100 * inst.margem_minima, 2),
        "unidades_vendidas": int(greedy.unidades.sum()),
        "estoque_remanescente": int(restante.sum()),
        "estoque_remanescente_valor_custo": round(float((restante * inst.custo_unitario).sum()), 2),
    }


def grafico_fitness(execucoes: dict, seed: int, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (nome, s), ga in execucoes.items():
        if s != seed:
            continue
        linha, = ax.plot(ga.melhor_por_geracao, label=f"{nome} — melhor")
        ax.plot(ga.media_por_geracao, ls="--", color=linha.get_color(), alpha=0.6, label=f"{nome} — média")
    ax.set(title="Fitness (caixa em R$) por geração do GA — catálogo completo",
           xlabel="Geração", ylabel="Caixa (R$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main(argv=None) -> dict:
    args = ler_argumentos(argv)
    param = carregar_parametros(args.parametros)
    catalogo, config, manifesto = carregar_questao(1, args.grupo, args.dados)
    catalogo, config = aplicar_cenario(catalogo, config, meta_fator=args.meta_fator,
                                       orcamento_fator=args.orcamento_fator, frete_fator=args.frete_fator,
                                       remover_skus=tuple(args.remover_sku))

    seed = args.seed if args.seed is not None else config["seed_final"]
    repeticoes = 1 if args.modo == "rapido" else param["repeticoes_ga"]
    seeds = [seed + i for i in range(repeticoes)]
    fracao = args.fracao_desconto if args.fracao_desconto is not None else param["fracao_max_skus_com_desconto"]
    k = args.subconjunto or param["tamanho_subconjunto_bf"]
    principal, comparacao = configs_ga(param, args)
    verba_separada = param["incentivo"]["orcamento_e_verba_separada"]

    inst = construir_instancia(catalogo, config, fracao)
    args.saida.mkdir(parents=True, exist_ok=True)

    # A) Força Bruta x GA no subconjunto pequeno
    sub, bf, tabela_bf = etapa_bf_vs_ga(inst, k, (principal, comparacao), seeds)
    tabela_bf.to_csv(args.saida / "bf_vs_ga_subconjunto.csv", index=False)

    # B) GA no catálogo completo (duas configurações)
    execucoes, tabela_ga, referencia = etapa_ga_completo(inst, (principal, comparacao), seeds)
    tabela_ga.to_csv(args.saida / "ga_configuracoes.csv", index=False)
    grafico_fitness(execucoes, seed, args.saida / "fitness_ga.png")
    melhor_ga = max((ga for (nome, _), ga in execucoes.items() if nome == principal.nome), key=lambda g: g.caixa)
    politica = melhor_ga.politica
    inst.descrever(politica).to_csv(args.saida / "politica_precos_ga.csv", index=False)

    # C) Greedy sobre a política do GA
    greedy, med_greedy = medir("greedy", alocar_incentivos, inst, politica,
                               param["incentivo"]["passos"], verba_separada)
    greedy.tabela.to_csv(args.saida / "greedy_antes_depois.csv", index=False)

    resumo = {
        "instancia": {"grupo": manifesto["grupo"], "id_instancia": manifesto["id_instancia"],
                      "seed_final": manifesto["seed_final"], "seed_algoritmos": seed, "skus": inst.n,
                      "max_skus_com_desconto": inst.max_descontos},
        "cenario": {"meta_fator": args.meta_fator, "orcamento_fator": args.orcamento_fator,
                    "frete_fator": args.frete_fator, "skus_removidos": args.remover_sku},
        "forca_bruta": {"skus": sub.skus.tolist(), "combinacoes": bf.combinacoes, "caixa_otimo": bf.caixa,
                        "gap_medio_ga_pct": round(float(tabela_bf.query("algoritmo == 'GA'")["gap_pct"].mean()), 4)},
        "ga": {"config_principal": asdict(principal), "caixa_melhor": melhor_ga.caixa,
               "referencia_exata": referencia, "gap_pct": gap_percentual(referencia, melhor_ga.caixa)},
        "antes_do_greedy": inst.metricas(politica),
        "status_quo_preco_atual": inst.metricas(inst.politica_atual()),
        "resultado_final": resumo_final(inst, politica, greedy, verba_separada),
        "greedy_medicao": med_greedy.como_dict(),
    }
    salvar_json(resumo, args.saida / "resumo.json")
    imprimir(resumo, tabela_bf, tabela_ga)
    return resumo


def imprimir(resumo: dict, tabela_bf: pd.DataFrame, tabela_ga: pd.DataFrame) -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:,.4f}".format)
    agrega = {"caixa": "mean", "gap_pct": "mean", "segundos": "mean", "pico_mb": "max"}
    print("\n=== Força Bruta x GA (subconjunto) ===")
    print(tabela_bf.groupby(["algoritmo", "config"]).agg(agrega).round(4))
    print("\n=== GA no catálogo completo ===")
    print(tabela_ga.groupby("config")[["caixa", "gap_pct_vs_exato", "geracoes", "segundos", "pico_mb"]]
          .mean().round(4))
    print("\n=== Resultado final (GA + Greedy) ===")
    for chave, valor in resumo["resultado_final"].items():
        print(f"{chave:>34}: {valor}")


if __name__ == "__main__":
    main()
