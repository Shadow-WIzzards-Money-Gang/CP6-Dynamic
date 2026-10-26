import shutil

import pytest

from comum.dados import PASTA_DADOS, carregar_questao, verificar_integridade
from comum.medicao import medir


def test_instancia_oficial_integra():
    manifesto = verificar_integridade(PASTA_DADOS / "G01")
    assert manifesto["seed_final"] == 1016


def test_arquivo_alterado_e_detectado(tmp_path):
    copia = tmp_path / "G01"
    shutil.copytree(PASTA_DADOS / "G01", copia)
    with open(copia / "q1_config_G01.json", "a", encoding="utf-8") as arquivo:
        arquivo.write(" ")
    with pytest.raises(ValueError, match="hash divergente"):
        verificar_integridade(copia)


@pytest.mark.parametrize("questao", [0, 3])
def test_questao_invalida(questao):
    with pytest.raises(ValueError):
        carregar_questao(questao)


def test_pasta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_questao(1, pasta=tmp_path / "nao_existe")


def test_medir_retorna_resultado_e_metricas():
    resultado, med = medir("soma", sum, range(1000))
    assert resultado == sum(range(1000))
    assert med.segundos >= 0 and med.pico_mb >= 0
