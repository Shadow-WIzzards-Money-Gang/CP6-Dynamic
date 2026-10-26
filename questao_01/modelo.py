"""Modelo econômico da Questão 1 (Operação Caixa 90).

Decisão: para cada SKU i, escolher um nível de preço j (multiplicador sobre preco_atual)
ou PAUSAR a venda (quando nenhum nível respeita a margem mínima, por exemplo).

    preco_ij    = preco_atual_i * nivel_j
    demanda_ij  = demanda_base_90d_i * nivel_j ** (-elasticidade_i)        (elasticidade constante)
    unidades_ij = min(estoque_i, floor(demanda_ij))
    margem_ij   = preco_ij * (1 - taxa_marketplace_i) - custo_unitario_i - custo_logistico_i
    caixa_ij    = unidades_ij * margem_ij

Restrições:
    1. piso de margem: margem_ij / preco_ij >= margem_minima_percentual  (nível fora disso é proibido)
    2. sem desconto indiscriminado: no máximo K SKUs com nível < 1.0, K = floor(fração * n)

Todas as combinações (SKU, nível) são pré-calculadas em tabelas n x (A+1). Assim, avaliar
uma política custa O(n) (uma soma de n posições) em vez de recalcular o modelo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InstanciaQ1:
    skus: np.ndarray            # (n,)   identificadores
    niveis: np.ndarray          # (A,)   multiplicadores de preço
    preco_atual: np.ndarray     # (n,)
    estoque: np.ndarray         # (n,)
    demanda_base: np.ndarray    # (n,)
    elasticidade: np.ndarray    # (n,)
    custo_unitario: np.ndarray  # (n,)
    margem: np.ndarray          # (n, A+1) margem unitária em R$ (coluna A = pausar)
    unidades: np.ndarray        # (n, A+1)
    receita: np.ndarray         # (n, A+1)
    caixa: np.ndarray           # (n, A+1)
    permitido: np.ndarray       # (n, A+1) bool — respeita o piso de margem
    desconto: np.ndarray        # (n, A+1) bool — nível < 1.0
    fracao_max_desconto: float
    meta_caixa: float
    divida: float
    orcamento_promocional: float
    margem_minima: float

    @property
    def n(self) -> int:
        return len(self.skus)

    @property
    def pausar(self) -> int:
        """Índice da opção 'pausar venda' (sempre permitida, 0 unidades)."""
        return len(self.niveis)

    @property
    def max_descontos(self) -> int:
        return math.floor(self.fracao_max_desconto * self.n)

    @property
    def opcoes(self) -> list[np.ndarray]:
        """Para cada SKU, os índices de opção permitidos."""
        return [np.flatnonzero(linha) for linha in self.permitido]

    @property
    def opcao_sem_desconto(self) -> np.ndarray:
        """Melhor opção permitida SEM desconto para cada SKU (usada para reparar soluções)."""
        caixa = np.where(self.permitido & ~self.desconto, self.caixa, -np.inf)
        return caixa.argmax(axis=1)

    def caixa_total(self, politica: np.ndarray) -> np.ndarray | float:
        """Caixa de uma política (n,) ou de uma população (P, n). Custo O(n) por política."""
        return self.caixa[np.arange(self.n), politica].sum(axis=-1)

    def n_descontos(self, politica: np.ndarray) -> np.ndarray | int:
        return self.desconto[np.arange(self.n), politica].sum(axis=-1)

    def viavel(self, politica: np.ndarray) -> bool:
        idx = np.arange(self.n)
        return bool(self.permitido[idx, politica].all() and self.n_descontos(politica) <= self.max_descontos)

    def subinstancia(self, indices: np.ndarray) -> "InstanciaQ1":
        """Mesma instância restrita a alguns SKUs (K é recalculado pela mesma fração)."""
        por_sku = {
            campo.name: getattr(self, campo.name)[indices]
            for campo in fields(self)
            if isinstance(getattr(self, campo.name), np.ndarray) and campo.name != "niveis"
        }
        return replace(self, **por_sku)

    def politica_atual(self) -> np.ndarray:
        """Status quo: preço atual (nível 1.0) se permitido; senão a melhor opção sem desconto."""
        atual = int(np.argmin(np.abs(self.niveis - 1.0)))
        return np.where(self.permitido[:, atual], atual, self.opcao_sem_desconto)

    def metricas(self, politica: np.ndarray) -> dict:
        idx = np.arange(self.n)
        receita = float(self.receita[idx, politica].sum())
        caixa = float(self.caixa[idx, politica].sum())
        unidades = self.unidades[idx, politica]
        restante = self.estoque - unidades
        return {
            "receita": receita,
            "caixa": caixa,
            "margem_percentual": caixa / receita if receita else 0.0,
            "unidades_vendidas": int(unidades.sum()),
            "estoque_remanescente": int(restante.sum()),
            "estoque_remanescente_custo": float((restante * self.custo_unitario).sum()),
            "skus_com_desconto": int(self.n_descontos(politica)),
            "skus_pausados": int((politica == self.pausar).sum()),
        }

    def descrever(self, politica: np.ndarray) -> pd.DataFrame:
        idx = np.arange(self.n)
        nivel = np.append(self.niveis, np.nan)[politica]
        return pd.DataFrame({
            "sku": self.skus,
            "nivel_preco": nivel,
            "preco": np.round(self.preco_atual * nivel, 2),
            "unidades": self.unidades[idx, politica],
            "estoque_restante": self.estoque - self.unidades[idx, politica],
            "caixa": np.round(self.caixa[idx, politica], 2),
        })


def aplicar_cenario(catalogo: pd.DataFrame, config: dict, *, meta_fator: float = 1.0,
                    orcamento_fator: float = 1.0, frete_fator: float = 1.0,
                    remover_skus: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict]:
    """Cenário-surpresa sem tocar nos arquivos oficiais: devolve cópias ajustadas."""
    desconhecidos = set(remover_skus) - set(catalogo["sku"])
    if desconhecidos:
        raise ValueError(f"SKUs inexistentes: {sorted(desconhecidos)}")
    catalogo = catalogo[~catalogo["sku"].isin(remover_skus)].reset_index(drop=True)
    catalogo["custo_logistico"] = catalogo["custo_logistico"] * frete_fator
    config = {**config,
              "meta_caixa_90d": config["meta_caixa_90d"] * meta_fator,
              "orcamento_promocional": config["orcamento_promocional"] * orcamento_fator}
    return catalogo, config


def construir_instancia(catalogo: pd.DataFrame, config: dict, fracao_max_desconto: float) -> InstanciaQ1:
    if catalogo.empty:
        raise ValueError("Catálogo vazio.")
    if not 0 <= fracao_max_desconto <= 1:
        raise ValueError("fracao_max_desconto deve estar em [0, 1].")

    niveis = np.asarray(config["niveis_preco_relativos"], dtype=float)

    def col(nome: str) -> np.ndarray:
        """Coluna do catálogo como vetor (n, 1), pronta para broadcasting com os níveis (A,)."""
        return catalogo[nome].to_numpy(dtype=float)[:, None]

    preco = col("preco_atual") * niveis                                        # (n, A) por broadcasting
    margem = preco * (1 - col("taxa_marketplace")) - col("custo_unitario") - col("custo_logistico")
    demanda = np.floor(col("demanda_base_90d") * niveis ** (-col("elasticidade")))
    unidades = np.minimum(col("estoque"), demanda)
    permitido = margem / preco >= config["margem_minima_percentual"]

    def com_pausa(tabela: np.ndarray, valor_pausa=0.0) -> np.ndarray:
        """Acrescenta a coluna da opção 'pausar venda'."""
        return np.hstack([tabela, np.full((len(catalogo), 1), valor_pausa, dtype=tabela.dtype)])

    return InstanciaQ1(
        skus=catalogo["sku"].to_numpy(),
        niveis=niveis,
        preco_atual=col("preco_atual").ravel(),
        estoque=col("estoque").ravel(),
        demanda_base=col("demanda_base_90d").ravel(),
        elasticidade=col("elasticidade").ravel(),
        custo_unitario=col("custo_unitario").ravel(),
        margem=com_pausa(margem),
        unidades=com_pausa(unidades),
        receita=com_pausa(unidades * preco),
        caixa=com_pausa(unidades * margem),
        permitido=com_pausa(permitido, True),
        desconto=com_pausa(np.broadcast_to(niveis < 1.0, preco.shape), False),
        fracao_max_desconto=fracao_max_desconto,
        meta_caixa=float(config["meta_caixa_90d"]),
        divida=float(config["divida_total_90d"]),
        orcamento_promocional=float(config["orcamento_promocional"]),
        margem_minima=float(config["margem_minima_percentual"]),
    )


def otimo_por_decomposicao(inst: InstanciaQ1) -> np.ndarray:
    """Ótimo exato usado como referência para o GA no catálogo completo.

    Como a única restrição que liga os SKUs é 'no máximo K descontos', o ótimo é:
    melhor opção sem desconto para todos + troca, nos K SKUs de maior ganho positivo,
    pela melhor opção com desconto. Custo O(n·A + n log n).
    """
    idx = np.arange(inst.n)
    sem = inst.opcao_sem_desconto
    com = np.where(inst.permitido & inst.desconto, inst.caixa, -np.inf).argmax(axis=1)
    ganho = inst.caixa[idx, com] - inst.caixa[idx, sem]
    ganho[~(inst.permitido & inst.desconto).any(axis=1)] = -np.inf

    politica = sem.copy()
    melhores = np.argsort(-ganho)[: inst.max_descontos]
    melhores = melhores[ganho[melhores] > 0]
    politica[melhores] = com[melhores]
    return politica
