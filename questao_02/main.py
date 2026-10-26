"""Questão 2 — Plano de Sobrevivência em 12 semanas: Monte Carlo -> DP -> Greedy -> Monte Carlo.

Fluxo de dados:
    A. Monte Carlo (construção) estima P(D = k) por SKU e nível de preço           -> histogramas
    B. Programação Dinâmica usa os histogramas e devolve preço(semana, estoque)      -> política
    C. Greedy usa o valor da DP para escolher os destaques semanais                  -> plano completo
    D. Monte Carlo (validação, cenários NOVOS) mede risco de não pagar as dívidas

A taxa de urgência da DP é escolhida pela maior probabilidade de pagar as três dívidas nos
cenários de construção (desempate: maior caixa final no percentil de risco).

Uso:
    python questao_02/main.py
    python questao_02/main.py --seed 14 --cenarios 5000
    python questao_02/main.py --concorrente-fator 0.92 --frete-fator 1.15 --orcamento-fator 0.75
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.dados import carregar_parametros, carregar_questao, salvar_json
from comum.medicao import Medicao, medir
from questao_02.dynamic_programming import PlanoDP, resolver_dp
from questao_02.greedy import PlanoDestaques, alocar_destaques
from questao_02.modelo import InstanciaQ2, aplicar_cenario, construir_instancia
from questao_02.monte_carlo import (ResultadoSimulacao, estimar_demanda, indicadores, politica_constante,
                                    simular)

PASTA = Path(__file__).resolve().parent


def ler_argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grupo", default="G01")
    p.add_argument("--dados", type=Path, help="pasta com os arquivos oficiais (padrão: data/<grupo>)")
    p.add_argument("--parametros", type=Path, default=PASTA / "parametros.json")
    p.add_argument("--saida", type=Path, default=PASTA / "resultados")
    p.add_argument("--seed", type=int, help="seed dos componentes aleatórios (padrão: seed_final)")
    p.add_argument("--cenarios", type=int, help="cenários da validação final (mínimo exigido: 5000)")
    p.add_argument("--cenarios-construcao", type=int, help="cenários da estimativa (mínimo exigido: 2000)")
    c = p.add_argument_group("cenário-surpresa")
    c.add_argument("--concorrente-fator", type=float, default=1.0, help="ex.: 0.92 = concorrente 8%% mais barato")
    c.add_argument("--frete-fator", type=float, default=1.0, help="ex.: 1.15 = custo logístico +15%%")
    c.add_argument("--orcamento-fator", type=float, default=1.0, help="ex.: 0.75 = orçamento -25%%")
    c.add_argument("--divida-fator", type=float, default=1.0)
    c.add_argument("--estoque-fator", type=float, default=1.0)
    c.add_argument("--remover-sku", nargs="*", default=[])
    c.add_argument("--max-skus", type=int, help="usa apenas os primeiros N SKUs (escalabilidade)")
    return p.parse_args(argv)


@dataclass
class ResultadoPipeline:
    taxa_urgencia: float
    plano: PlanoDP
    destaques: PlanoDestaques
    escolha: pd.DataFrame
    estatisticas_resumo: pd.DataFrame
    medicoes: list[Medicao]


def construir_politica(inst: InstanciaQ2, taxas: list[float], n_construcao: int, percentil: float,
                       seeds: list[np.random.SeedSequence]) -> ResultadoPipeline:
    """Etapas A -> B -> C (+ teste do plano nos cenários de construção para escolher a urgência)."""
    stats, med_mc = medir("A_monte_carlo_estimativa", estimar_demanda, inst, n_construcao,
                          np.random.default_rng(seeds[0]))
    medicoes, linhas, candidatos = [med_mc], [], {}
    chave_risco = f"caixa_final_p{percentil:g}"

    for taxa in taxas:
        plano, med_dp = medir(f"B_dp_urgencia_{taxa}", resolver_dp, inst, stats, taxa)
        destaques, med_gr = medir(f"C_greedy_urgencia_{taxa}", alocar_destaques, inst, stats, plano)
        sim = simular(inst, plano.politica, destaques.destaques, destaques.custo, n_construcao,
                      np.random.default_rng(seeds[1]))          # mesmos cenários para todas as taxas
        ind = indicadores(inst, sim, percentil)
        candidatos[taxa] = (plano, destaques)
        medicoes += [med_dp, med_gr]
        linhas.append({"taxa_urgencia": taxa, "prob_pagar_todas": ind["prob_pagar_todas"],
                       chave_risco: ind[chave_risco], "caixa_final_medio": ind["caixa_final_medio"],
                       "destaques": int(destaques.destaques.sum())})

    escolha = pd.DataFrame(linhas)
    melhor = escolha.sort_values(["prob_pagar_todas", chave_risco], ascending=False).iloc[0]
    taxa = float(melhor["taxa_urgencia"])
    plano, destaques = candidatos[taxa]
    return ResultadoPipeline(taxa, plano, destaques, escolha, stats.resumo, medicoes)


def validar(inst: InstanciaQ2, res: ResultadoPipeline, n_validacao: int, percentil: float,
            seed: np.random.SeedSequence) -> tuple[pd.DataFrame, dict[str, ResultadoSimulacao], Medicao]:
    """Etapa D: plano completo e duas referências nos MESMOS cenários novos de validação."""
    sem_destaque = (np.zeros((inst.semanas, inst.n), dtype=bool), np.zeros((inst.semanas, inst.n)))
    politicas = {
        "preco_atual_fixo": (politica_constante(inst, inst.nivel_atual), *sem_destaque),
        "dp_sem_greedy": (res.plano.politica, *sem_destaque),
        "dp_mais_greedy": (res.plano.politica, res.destaques.destaques, res.destaques.custo),
    }
    simulacoes, medicao = {}, None
    for nome, (politica, destaques, custo) in politicas.items():
        sim, med = medir("D_monte_carlo_validacao", simular, inst, politica, destaques, custo,
                         n_validacao, np.random.default_rng(seed))
        simulacoes[nome] = sim
        medicao = med if nome == "dp_mais_greedy" else medicao
    tabela = pd.DataFrame({nome: indicadores(inst, sim, percentil) for nome, sim in simulacoes.items()})
    return tabela, simulacoes, medicao


def plano_semanal(inst: InstanciaQ2, res: ResultadoPipeline) -> pd.DataFrame:
    estoque, nivel = res.destaques.estoque_esperado, res.destaques.nivel
    linhas = []
    for i, sku in enumerate(inst.skus):
        for t in range(inst.semanas):
            vende = estoque[i, t] > 0
            linhas.append({"sku": sku, "semana": t + 1,
                           "nivel_preco": inst.niveis[nivel[i, t]] if vende else np.nan,
                           "preco": round(inst.preco[i, nivel[i, t]], 2) if vende else np.nan,
                           "destaque": bool(res.destaques.destaques[t, i]),
                           "estoque_inicio": int(estoque[i, t]), "estoque_fim": int(estoque[i, t + 1])})
    return pd.DataFrame(linhas)


def grafico_distribuicao(inst: InstanciaQ2, simulacoes: dict, percentil: float, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for nome, sim in simulacoes.items():
        ax.hist(sim.caixa_final / 1e3, bins=60, alpha=0.55, label=nome)
        ax.axvline(np.percentile(sim.caixa_final, percentil) / 1e3, ls="--", lw=1, color="black", alpha=0.5)
    ax.axvline(0, color="red", lw=1.5, label="saldo zero (falha)")
    ax.set(title=f"Caixa final após as três dívidas — {len(sim.caixa_final)} cenários de validação "
                 f"(tracejado: P{percentil:g})", xlabel="Caixa final (R$ mil)", ylabel="Cenários")
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def skus_representativos(inst: InstanciaQ2, res: ResultadoPipeline, quantidade: int = 4) -> list[int]:
    """Metade com maior valor de estoque a custo, metade com mais destaques do Greedy."""
    por_valor = np.argsort(-inst.estoque * inst.custo_unitario)
    por_destaque = np.argsort(-res.destaques.destaques.sum(axis=0), kind="stable")
    escolhidos = list(por_valor[: quantidade // 2])
    escolhidos += [i for i in por_destaque if i not in escolhidos][: quantidade - len(escolhidos)]
    return escolhidos


def grafico_trajetorias(inst: InstanciaQ2, res: ResultadoPipeline, sim: ResultadoSimulacao, caminho: Path) -> None:
    escolhidos = skus_representativos(inst, res)
    semanas = np.arange(inst.semanas + 1)
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, i in zip(eixos.ravel(), escolhidos):
        faixa = np.percentile(sim.estoque[:, :, i], [10, 50, 90], axis=0)
        ax.fill_between(semanas, faixa[0], faixa[2], alpha=0.25, label="estoque P10–P90 (validação)")
        ax.plot(semanas, faixa[1], label="estoque mediano")
        ax.plot(semanas, res.destaques.estoque_esperado[i], ls="--", label="estoque esperado (plano)")
        ax.set(title=inst.skus[i], ylabel="Estoque (un.)")
        preco = ax.twinx()
        nivel = np.where(res.destaques.estoque_esperado[i, :-1] > 0,
                         inst.niveis[res.destaques.nivel[i]], np.nan)
        preco.step(semanas[1:], nivel, where="pre", color="tab:red", label="nível de preço")
        destaque = res.destaques.destaques[:, i]
        preco.scatter(semanas[1:][destaque], nivel[destaque], marker="*", s=90, color="gold",
                      edgecolor="black", zorder=3, label="destaque (Greedy)")
        preco.set_ylim(inst.niveis.min() - 0.05, inst.niveis.max() + 0.05)
        preco.set_ylabel("Nível de preço")
    eixos[0, 0].legend(loc="lower left", fontsize=8)
    fig.legend(*preco.get_legend_handles_labels(), loc="upper right", fontsize=8)
    for ax in eixos[1]:
        ax.set_xlabel("Semana")
    fig.suptitle("Trajetória semanal de preço e estoque — SKUs representativos")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main(argv=None) -> dict:
    args = ler_argumentos(argv)
    par = carregar_parametros(args.parametros)
    catalogo, config, manifesto = carregar_questao(2, args.grupo, args.dados)
    catalogo, config = aplicar_cenario(
        catalogo, config, concorrente_fator=args.concorrente_fator, frete_fator=args.frete_fator,
        orcamento_fator=args.orcamento_fator, divida_fator=args.divida_fator,
        estoque_fator=args.estoque_fator, remover_skus=args.remover_sku, max_skus=args.max_skus)
    inst = construir_instancia(catalogo, config, par)

    seed = args.seed if args.seed is not None else config["seed_final"]
    n_construcao = args.cenarios_construcao or par["cenarios_construcao"]
    n_validacao = args.cenarios or par["cenarios_validacao"]
    if n_construcao < 2000 or n_validacao < 5000:
        print("Aviso: o enunciado exige >= 2000 cenários de construção e >= 5000 de validação.")
    percentil = par["percentil_risco"]
    seeds = np.random.SeedSequence(seed).spawn(3)   # estimativa, construção, validação: fluxos independentes

    res = construir_politica(inst, par["taxas_urgencia"], n_construcao, percentil, seeds[:2])
    validacao, simulacoes, med_val = validar(inst, res, n_validacao, percentil, seeds[2])

    saida = args.saida
    saida.mkdir(parents=True, exist_ok=True)
    res.estatisticas_resumo.to_csv(saida / "monte_carlo_estatisticas.csv", index=False)
    res.escolha.to_csv(saida / "escolha_taxa_urgencia.csv", index=False)
    plano_semanal(inst, res).to_csv(saida / "plano_precos_semanal.csv", index=False)
    res.destaques.tabela.to_csv(saida / "destaques_greedy.csv", index=False)
    validacao.to_csv(saida / "validacao_comparativo.csv")
    grafico_distribuicao(inst, simulacoes, percentil, saida / "distribuicao_caixa_final.png")
    grafico_trajetorias(inst, res, simulacoes["dp_mais_greedy"], saida / "trajetorias_skus.png")

    resumo = {
        "instancia": {"grupo": manifesto["grupo"], "id_instancia": manifesto["id_instancia"],
                      "seed_final": manifesto["seed_final"], "seed_algoritmos": seed, "skus": inst.n,
                      "cenarios_construcao": n_construcao, "cenarios_validacao": n_validacao},
        "cenario": {k: getattr(args, k) for k in ("concorrente_fator", "frete_fator", "orcamento_fator",
                                                  "divida_fator", "estoque_fator", "remover_sku", "max_skus")},
        "dividas": inst.dividas,
        "caixa_inicial": inst.caixa_inicial,
        "taxa_urgencia_escolhida": res.taxa_urgencia,
        "destaques_alocados": int(res.destaques.destaques.sum()),
        "custo_destaques_total": round(float(res.destaques.custo.sum()), 2),
        "validacao": validacao.to_dict(),
        "medicoes": [m.como_dict() for m in res.medicoes + [med_val]],
    }
    salvar_json(resumo, saida / "resumo.json")

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:,.4f}".format)
    print("\n=== Escolha da taxa de urgência (cenários de construção) ===")
    print(res.escolha.round(4).to_string(index=False))
    print(f"\n=== Validação em {n_validacao} cenários novos ===")
    print(validacao.round(4).to_string())
    return resumo


if __name__ == "__main__":
    main()
