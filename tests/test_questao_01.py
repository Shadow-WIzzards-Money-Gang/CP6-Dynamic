import numpy as np
import pandas as pd
import pytest

from comum.dados import carregar_questao
from questao_01 import brute_force
from questao_01.brute_force import forca_bruta, forca_bruta_ingenua
from questao_01.genetic_algorithm import ConfigGA, executar_ga, reparar
from questao_01.greedy import alocar_incentivos
from questao_01.main import main
from questao_01.modelo import aplicar_cenario, construir_instancia, otimo_por_decomposicao

CONFIG = {
    "meta_caixa_90d": 1000.0, "divida_total_90d": 900.0, "orcamento_promocional": 50.0,
    "margem_minima_percentual": 0.10, "niveis_preco_relativos": [0.8, 0.9, 1.0, 1.1],
}


def catalogo_pequeno() -> pd.DataFrame:
    return pd.DataFrame({
        "sku": ["A", "B", "C"],
        "custo_unitario": [50.0, 80.0, 10.0],
        "preco_atual": [100.0, 100.0, 30.0],
        "estoque": [10, 5, 100],
        "demanda_base_90d": [8, 5, 60],
        "elasticidade": [2.0, 0.5, 3.0],
        "taxa_marketplace": [0.1, 0.1, 0.1],
        "custo_logistico": [5.0, 5.0, 2.0],
    })


@pytest.fixture(scope="module")
def instancia_oficial():
    catalogo, config, _ = carregar_questao(1)
    return construir_instancia(catalogo, config, 0.3)


def test_modelo_calcula_caixa_e_piso_de_margem():
    inst = construir_instancia(catalogo_pequeno(), CONFIG, 1.0)
    # SKU A no nível 1.0: 8 unidades x (100*0.9 - 50 - 5) = 280
    assert inst.caixa[0, 2] == pytest.approx(280.0)
    # SKU B: margem (90*0.9-85)/90 < 10% no nível 0.9 -> proibido; pausar é sempre permitido
    assert not inst.permitido[1, 1]
    assert inst.permitido[:, inst.pausar].all()
    assert (inst.unidades <= inst.estoque[:, None]).all()


def test_catalogo_vazio_e_fracao_invalida():
    with pytest.raises(ValueError):
        construir_instancia(catalogo_pequeno().iloc[0:0], CONFIG, 0.3)
    with pytest.raises(ValueError):
        construir_instancia(catalogo_pequeno(), CONFIG, 1.5)


@pytest.mark.parametrize("seed", range(5))
def test_forca_bruta_vetorizada_igual_ingenua_e_decomposicao(instancia_oficial, seed):
    indices = np.sort(np.random.default_rng(seed).choice(instancia_oficial.n, 5, replace=False))
    sub = instancia_oficial.subinstancia(indices)
    rapida, lenta = forca_bruta(sub), forca_bruta_ingenua(sub)
    assert rapida.caixa == pytest.approx(lenta.caixa)
    assert rapida.caixa == pytest.approx(sub.caixa_total(otimo_por_decomposicao(sub)))
    assert sub.viavel(rapida.politica)


def test_limite_de_desconto_e_respeitado():
    inst = construir_instancia(catalogo_pequeno(), CONFIG, 0.0)   # nenhum desconto permitido
    assert inst.n_descontos(forca_bruta(inst).politica) == 0


def test_forca_bruta_recusa_espaco_grande(instancia_oficial, monkeypatch):
    monkeypatch.setattr(brute_force, "LIMITE_COMBINACOES", 10)
    with pytest.raises(ValueError, match="inviável"):
        forca_bruta(instancia_oficial)


def test_ga_reprodutivel_viavel_e_alcanca_otimo(instancia_oficial):
    sub = instancia_oficial.subinstancia(np.arange(6))
    cfg = ConfigGA(populacao=60, geracoes=200)
    a, b = executar_ga(sub, cfg, seed=7), executar_ga(sub, cfg, seed=7)
    assert a.caixa == b.caixa and (a.politica == b.politica).all()
    assert sub.viavel(a.politica)
    assert a.caixa == pytest.approx(forca_bruta(sub).caixa)
    assert np.all(np.diff(a.melhor_por_geracao) >= -1e-9)   # elitismo: o melhor nunca piora


def test_reparo_limita_descontos(instancia_oficial):
    rng = np.random.default_rng(0)
    pop = np.zeros((20, instancia_oficial.n), dtype=int)           # nível 0.80 = desconto em tudo
    pop = np.where(instancia_oficial.permitido[:, 0], pop, instancia_oficial.pausar)
    reparada = reparar(pop, instancia_oficial, rng)
    assert (instancia_oficial.n_descontos(reparada) <= instancia_oficial.max_descontos).all()


@pytest.mark.parametrize("kwargs", [{"populacao": 1}, {"elitismo": 100}, {"torneio": 0}, {"geracoes": 0}])
def test_config_ga_invalida(kwargs):
    with pytest.raises(ValueError):
        ConfigGA(**kwargs)


def test_greedy_respeita_orcamento(instancia_oficial):
    politica = otimo_por_decomposicao(instancia_oficial)
    res = alocar_incentivos(instancia_oficial, politica, [0.02, 0.05, 0.1])
    assert res.custo_incentivo <= instancia_oficial.orcamento_promocional + 1e-6
    assert res.caixa_operacional >= instancia_oficial.caixa_total(politica)
    assert (res.incentivo[politica == instancia_oficial.pausar] == 0).all()


def test_greedy_orcamento_zero_e_negativo():
    cat, cfg = aplicar_cenario(catalogo_pequeno(), CONFIG, orcamento_fator=0.0)
    inst = construir_instancia(cat, cfg, 1.0)
    res = alocar_incentivos(inst, inst.politica_atual(), [0.05])
    assert res.custo_incentivo == 0 and (res.incentivo == 0).all()
    cat, cfg = aplicar_cenario(catalogo_pequeno(), CONFIG, orcamento_fator=-1.0)
    with pytest.raises(ValueError):
        alocar_incentivos(construir_instancia(cat, cfg, 1.0), inst.politica_atual(), [0.05])


def test_greedy_sem_verba_separada_so_aceita_passos_que_se_pagam(instancia_oficial):
    politica = otimo_por_decomposicao(instancia_oficial)
    res = alocar_incentivos(instancia_oficial, politica, [0.02, 0.05], verba_separada=False)
    ganho = res.caixa_operacional - instancia_oficial.caixa_total(politica)
    assert ganho >= res.custo_incentivo


def test_cenario_surpresa():
    cat, cfg = aplicar_cenario(catalogo_pequeno(), CONFIG, meta_fator=1.1, frete_fator=1.15, remover_skus=("B",))
    assert list(cat["sku"]) == ["A", "C"]
    assert cfg["meta_caixa_90d"] == pytest.approx(1100.0)
    assert cat["custo_logistico"].iloc[0] == pytest.approx(5.75)
    with pytest.raises(ValueError):
        aplicar_cenario(catalogo_pequeno(), CONFIG, remover_skus=("Z",))


def test_main_ponta_a_ponta(tmp_path):
    resumo = main(["--modo", "rapido", "--saida", str(tmp_path), "--meta-fator", "1.1",
                   "--remover-sku", "Q1-001"])
    _, config, _ = carregar_questao(1)
    final = resumo["resultado_final"]
    assert final["meta_caixa_90d"] == pytest.approx(1.1 * config["meta_caixa_90d"], abs=0.01)
    assert final["meta_atingida"] == (final["folga_ou_insuficiencia"] >= 0)
    assert resumo["instancia"]["skus"] == 29
    assert (tmp_path / "fitness_ga.png").exists() and (tmp_path / "resumo.json").exists()
