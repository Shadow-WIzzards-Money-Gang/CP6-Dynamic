import itertools
import json

import numpy as np
import pandas as pd
import pytest

from comum.dados import RAIZ, carregar_questao
from questao_02.dynamic_programming import resolver_dp, resolver_sku, resolver_sku_top_down
from questao_02.greedy import alocar_destaques
from questao_02.main import main
from questao_02.modelo import aplicar_cenario, construir_instancia
from questao_02.monte_carlo import choque_lognormal, estimar_demanda, indicadores, politica_constante, simular

PARAMETROS = json.loads((RAIZ / "questao_02" / "parametros.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def oficial():
    catalogo, config, _ = carregar_questao(2)
    return catalogo, config


def instancia(oficial, **cenario):
    catalogo, config = aplicar_cenario(*oficial, **cenario)
    return construir_instancia(catalogo, config, PARAMETROS)


@pytest.fixture(scope="module")
def pequena(oficial):
    inst = instancia(oficial, max_skus=6)
    stats = estimar_demanda(inst, 2000, np.random.default_rng(1))
    return inst, stats


def test_choque_lognormal_tem_media_um():
    amostras = choque_lognormal(np.random.default_rng(0), 0.15, 200_000)
    assert amostras.mean() == pytest.approx(1.0, abs=0.005)
    assert (amostras > 0).all()


def test_estimativa_reprodutivel_e_histogramas_validos(pequena):
    inst, stats = pequena
    de_novo = estimar_demanda(inst, 2000, np.random.default_rng(1))
    for p, q in zip(stats.prob, de_novo.prob):
        assert np.array_equal(p, q)
        assert np.allclose(p.sum(axis=1), 1.0)
    # preço maior -> demanda média menor (elasticidade positiva)
    media = stats.resumo.groupby("nivel_preco")["demanda_media"].sum()
    assert media.is_monotonic_decreasing


def test_dp_bottom_up_igual_top_down(pequena):
    inst, stats = pequena
    margem = inst.margem_antes_logistica - stats.logistica_media[:, None]
    for i in range(3):
        a = resolver_sku(stats.prob[i], margem[i], int(inst.estoque[i]), 6, 0.97, 2.0)
        b = resolver_sku_top_down(stats.prob[i], margem[i], int(inst.estoque[i]), 6, 0.97, 2.0)
        assert np.allclose(a.valor, b.valor)


def test_dp_igual_enumeracao_exaustiva_com_demanda_deterministica():
    """Demanda determinística: a DP deve achar a melhor sequência entre todas as A^T possíveis."""
    demanda = np.array([5, 3, 1])                 # nível 0 (barato) vende 5/semana ... nível 2 vende 1
    margem = np.array([1.0, 4.0, 9.0])
    prob = np.zeros((3, 6))
    prob[np.arange(3), demanda] = 1.0
    estoque, semanas = 7, 3

    melhor = -np.inf
    for sequencia in itertools.product(range(3), repeat=semanas):
        s, total = estoque, 0.0
        for a in sequencia:
            vendas = min(s, demanda[a])
            total, s = total + vendas * margem[a], s - vendas
        melhor = max(melhor, total + 0.5 * s)
    sol = resolver_sku(prob, margem, estoque, semanas, gamma=1.0, valor_residual=0.5)
    assert sol.valor[0, estoque] == pytest.approx(melhor)


def test_dp_estoque_zero_vale_zero(pequena):
    inst, stats = pequena
    plano = resolver_dp(inst, stats)
    assert all(np.allclose(sol.valor[:, 0], 0) for sol in plano.solucoes)
    with pytest.raises(ValueError):
        resolver_dp(inst, stats, -0.1)


def test_greedy_respeita_vagas_e_orcamento(pequena):
    inst, stats = pequena
    plano = alocar_destaques(inst, stats, resolver_dp(inst, stats))
    assert (plano.destaques.sum(axis=1) <= inst.max_destaques).all()
    assert (plano.custo.sum(axis=1) <= inst.orcamento_semanal + 1e-6).all()
    assert (plano.estoque_esperado >= 0).all()


def test_simulacao_reprodutivel_e_estoque_nao_negativo(pequena):
    inst, stats = pequena
    plano = resolver_dp(inst, stats)
    sem = np.zeros((inst.semanas, inst.n), dtype=bool), np.zeros((inst.semanas, inst.n))
    a = simular(inst, plano.politica, *sem, 500, np.random.default_rng(3))
    b = simular(inst, plano.politica, *sem, 500, np.random.default_rng(3))
    assert np.array_equal(a.caixa_final, b.caixa_final)
    assert (a.estoque >= 0).all()
    assert (np.diff(a.estoque, axis=1) <= 0).all()


@pytest.mark.parametrize("fator, esperado", [(0.0, 1.0), (100.0, 0.0)])
def test_probabilidade_de_pagamento_nos_extremos(oficial, fator, esperado):
    inst = instancia(oficial, max_skus=4, divida_fator=fator)
    sem = np.zeros((inst.semanas, inst.n), dtype=bool), np.zeros((inst.semanas, inst.n))
    sim = simular(inst, politica_constante(inst, inst.nivel_atual), *sem, 300, np.random.default_rng(0))
    assert indicadores(inst, sim, 10)["prob_pagar_todas"] == esperado


def test_cenario_invalido(oficial):
    with pytest.raises(ValueError):
        instancia(oficial, remover_skus=["Q2-999"])
    catalogo, config = oficial
    config = {**config, "dividas": {"semana_20": 1.0}}
    with pytest.raises(ValueError, match="horizonte"):
        construir_instancia(*aplicar_cenario(catalogo, config), PARAMETROS)


def test_concorrente_mais_barato_reduz_demanda(oficial):
    base = estimar_demanda(instancia(oficial), 2000, np.random.default_rng(0)).resumo
    barato = estimar_demanda(instancia(oficial, concorrente_fator=0.92), 2000, np.random.default_rng(0)).resumo
    assert barato["demanda_media"].sum() < base["demanda_media"].sum()


def test_main_ponta_a_ponta(tmp_path):
    resumo = main(["--saida", str(tmp_path), "--cenarios", "300", "--cenarios-construcao", "300",
                   "--max-skus", "6", "--frete-fator", "1.15"])
    validacao = pd.DataFrame(resumo["validacao"])
    assert set(validacao.columns) == {"preco_atual_fixo", "dp_sem_greedy", "dp_mais_greedy"}
    assert (validacao.loc["prob_pagar_todas"].between(0, 1)).all()
    assert (tmp_path / "trajetorias_skus.png").exists()
