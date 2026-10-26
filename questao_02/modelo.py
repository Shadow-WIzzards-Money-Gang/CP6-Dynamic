"""Modelo da Questão 2 (Plano de Sobrevivência em 12 semanas).

Demanda semanal do SKU i no nível de preço a, em um cenário com choques aleatórios:

    D_ia = round( demanda_base_i · nivel_a^(-e_i) · (conc_t / conc_ref_i)^β · ε_t · (1 + u·destaque) )

    ε_t    ~ LogNormal com média 1 e dispersão sigma_demanda            (demanda do consumidor)
    conc_t = preco_concorrente_base · LogNormal(média 1, sigma_conc)     (concorrente mais barato -> D cai)
    log_t  = custo_logistico_base   · LogNormal(média 1, sigma_log)      (custo de atendimento)

Caixa semanal do SKU = vendas · (preco·(1 - taxa) - custo_unitario - log_t), com vendas = min(estoque, D).
custo_unitario é tratado como saída de caixa na venda, mesma convenção do gerador oficial.
Dívidas são pagas nas semanas 4, 8 e 12; faltando caixa, o compromisso falha e o saldo fica negativo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InstanciaQ2:
    skus: np.ndarray
    niveis: np.ndarray              # (A,)
    preco_atual: np.ndarray         # (n,)
    custo_unitario: np.ndarray
    taxa_marketplace: np.ndarray
    elasticidade: np.ndarray
    demanda_base: np.ndarray
    estoque: np.ndarray             # (n,) int
    logistica_base: np.ndarray
    concorrente_base: np.ndarray    # preço do concorrente no cenário (pode ter sofrido choque)
    concorrente_ref: np.ndarray     # preço do concorrente do arquivo oficial (referência da demanda)
    semanas: int
    caixa_inicial: float
    dividas: dict[int, float]       # semana -> valor
    sigma_demanda: float
    sigma_concorrente: float
    sigma_logistica: float
    orcamento_semanal: float
    elasticidade_concorrencia: float
    aumento_destaque: float
    custo_destaque_pct: float
    max_destaques: int
    fator_valor_residual: float

    @property
    def n(self) -> int:
        return len(self.skus)

    @property
    def preco(self) -> np.ndarray:
        """(n, A) preço de cada SKU em cada nível."""
        return self.preco_atual[:, None] * self.niveis

    @property
    def margem_antes_logistica(self) -> np.ndarray:
        """(n, A) preco·(1-taxa) - custo_unitario."""
        return self.preco * (1 - self.taxa_marketplace[:, None]) - self.custo_unitario[:, None]

    @property
    def nivel_atual(self) -> int:
        return int(np.argmin(np.abs(self.niveis - 1.0)))


def aplicar_cenario(catalogo: pd.DataFrame, config: dict, *, concorrente_fator: float = 1.0,
                    frete_fator: float = 1.0, orcamento_fator: float = 1.0, divida_fator: float = 1.0,
                    estoque_fator: float = 1.0, remover_skus=(), max_skus: int | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    """Cenário-surpresa e tamanhos de entrada sem alterar os arquivos oficiais."""
    remover = tuple(remover_skus or ())
    desconhecidos = set(remover) - set(catalogo["sku"])
    if desconhecidos:
        raise ValueError(f"SKUs inexistentes: {sorted(desconhecidos)}")
    catalogo = catalogo[~catalogo["sku"].isin(remover)].head(max_skus).reset_index(drop=True)
    if catalogo.empty:
        raise ValueError("Catálogo vazio após aplicar o cenário.")

    catalogo["concorrente_ref"] = catalogo["preco_concorrente_base"]
    catalogo["preco_concorrente_base"] *= concorrente_fator
    catalogo["custo_logistico_base"] *= frete_fator
    catalogo["estoque_inicial"] = np.maximum(1, np.round(catalogo["estoque_inicial"] * estoque_fator)).astype(int)

    config = {**config,
              "orcamento_promocional_semanal": config["orcamento_promocional_semanal"] * orcamento_fator,
              "dividas": {k: v * divida_fator for k, v in config["dividas"].items()}}
    return catalogo, config


def construir_instancia(catalogo: pd.DataFrame, config: dict, parametros: dict) -> InstanciaQ2:
    def col(nome: str) -> np.ndarray:
        return catalogo[nome].to_numpy(dtype=float)

    variab = config["variabilidade"]
    destaque = parametros["destaque"]
    dividas = {int(chave.split("_")[-1]): float(valor) for chave, valor in config["dividas"].items()}
    if any(not 1 <= semana <= config["numero_semanas"] for semana in dividas):
        raise ValueError("Dívida fora do horizonte de semanas.")

    return InstanciaQ2(
        skus=catalogo["sku"].to_numpy(),
        niveis=np.asarray(config["niveis_preco_relativos"], dtype=float),
        preco_atual=col("preco_atual"),
        custo_unitario=col("custo_unitario"),
        taxa_marketplace=col("taxa_marketplace"),
        elasticidade=col("elasticidade"),
        demanda_base=col("demanda_base_semanal"),
        estoque=catalogo["estoque_inicial"].to_numpy(dtype=int),
        logistica_base=col("custo_logistico_base"),
        concorrente_base=col("preco_concorrente_base"),
        concorrente_ref=catalogo.get("concorrente_ref", catalogo["preco_concorrente_base"]).to_numpy(dtype=float),
        semanas=int(config["numero_semanas"]),
        caixa_inicial=float(config["caixa_inicial"]),
        dividas=dict(sorted(dividas.items())),
        sigma_demanda=float(variab["sigma_demanda"]),
        sigma_concorrente=float(variab["sigma_preco_concorrente"]),
        sigma_logistica=float(variab["sigma_custo_logistico"]),
        orcamento_semanal=float(config["orcamento_promocional_semanal"]),
        elasticidade_concorrencia=float(parametros["elasticidade_concorrencia"]),
        aumento_destaque=float(destaque["aumento_demanda"]),
        custo_destaque_pct=float(destaque["custo_pct_receita"]),
        max_destaques=int(destaque["max_por_semana"]),
        fator_valor_residual=float(parametros["fator_valor_residual"]),
    )
