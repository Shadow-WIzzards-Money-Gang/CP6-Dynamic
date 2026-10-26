"""Leitura dos arquivos oficiais da instância, verificação de integridade e gravação de saídas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "data"


def sha256_arquivo(caminho: Path) -> str:
    """Hash SHA-256 de um arquivo, lido em blocos de 64 KiB (mesma regra do notebook gerador)."""
    h = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_integridade(pasta: Path) -> dict:
    """Confere os hashes do manifesto. Lança ValueError se algum arquivo oficial foi alterado."""
    manifesto = json.loads((pasta / "manifesto_dados.json").read_text(encoding="utf-8"))
    alterados = [
        nome for nome, esperado in manifesto["arquivos"].items()
        if sha256_arquivo(pasta / nome) != esperado
    ]
    if alterados:
        raise ValueError(f"Arquivos oficiais alterados (hash divergente): {alterados}")
    return manifesto


def carregar_questao(questao: int, grupo: str = "G01", pasta: Path | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Retorna (catálogo, config, manifesto) da questão 1 ou 2, após validar a integridade."""
    if questao not in (1, 2):
        raise ValueError("questao deve ser 1 ou 2")
    pasta = Path(pasta) if pasta else PASTA_DADOS / grupo
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {pasta}")

    manifesto = verificar_integridade(pasta)
    catalogo = pd.read_csv(pasta / f"q{questao}_catalogo_{grupo}.csv", encoding="utf-8-sig")
    config = json.loads((pasta / f"q{questao}_config_{grupo}.json").read_text(encoding="utf-8"))
    return catalogo, config, manifesto


def carregar_parametros(caminho: Path) -> dict:
    """Parâmetros de modelagem do grupo (não fazem parte da instância oficial)."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def salvar_json(objeto: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
