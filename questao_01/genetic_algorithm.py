"""Etapa B — Algoritmo Genético: política de preços para o catálogo completo.

Representação: cromossomo = vetor de n inteiros; gene i = opção de preço do SKU i
(índice em `niveis`, ou `pausar`). Só são sorteadas opções que respeitam o piso de margem.

Operadores:
    inicialização  aleatória entre opções permitidas + 1 indivíduo "status quo" (preço atual)
    seleção        torneio de tamanho t (pressão seletiva ajustável, não depende da escala do fitness)
    crossover      uniforme (genes independentes entre SKUs -> não há "blocos" a preservar)
    mutação        cada gene troca para outra opção permitida com probabilidade pm
    reparo         se houver mais de K descontos, descontos excedentes (sorteados) voltam à
                   melhor opção sem desconto -> toda a população é sempre viável
    elitismo       os E melhores passam intactos para a próxima geração
    parada         G gerações OU `paciencia` gerações sem melhorar o melhor indivíduo

Complexidade (G gerações, P indivíduos, n SKUs, t torneio):
    Tempo:   O(G · P · (n log n + t)) — o fitness é O(P·n) (tabela pré-calculada);
             o reparo ordena n chaves por indivíduo (n log n) e domina para n grande.
    Memória: O(P · n + G) — população, filhos e histórico de fitness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from questao_01.modelo import InstanciaQ1


@dataclass(frozen=True)
class ConfigGA:
    nome: str = "padrao"
    populacao: int = 100
    geracoes: int = 300
    taxa_crossover: float = 0.9
    taxa_mutacao: float | None = None   # None -> 1/n (em média, um gene mutado por filho)
    torneio: int = 3
    elitismo: int = 2
    paciencia: int = 60

    def __post_init__(self):
        if self.populacao < 2 or self.geracoes < 1:
            raise ValueError("populacao >= 2 e geracoes >= 1")
        if not 0 <= self.elitismo < self.populacao:
            raise ValueError("elitismo deve estar em [0, populacao)")
        if self.torneio < 1:
            raise ValueError("torneio >= 1")


@dataclass
class ResultadoGA:
    politica: np.ndarray
    caixa: float
    geracoes: int
    avaliacoes: int
    melhor_por_geracao: list[float] = field(default_factory=list)
    media_por_geracao: list[float] = field(default_factory=list)


def _sortear_opcoes(rng: np.random.Generator, opcoes: list[np.ndarray], linhas: int) -> np.ndarray:
    return np.column_stack([rng.choice(op, size=linhas) for op in opcoes])


def reparar(pop: np.ndarray, inst: InstanciaQ1, rng: np.random.Generator) -> np.ndarray:
    """Mantém no máximo K descontos por indivíduo, escolhendo ao acaso quais descontos ficam."""
    desconto = inst.desconto[np.arange(inst.n), pop]
    if (desconto.sum(axis=1) <= inst.max_descontos).all():
        return pop
    chaves = np.where(desconto, rng.random(pop.shape), np.inf)
    posicao = chaves.argsort(axis=1).argsort(axis=1)          # 0 = desconto que fica primeiro
    excedente = desconto & (posicao >= inst.max_descontos)
    return np.where(excedente, inst.opcao_sem_desconto, pop)


def _torneio(fitness: np.ndarray, tamanho: int, quantidade: int, rng: np.random.Generator) -> np.ndarray:
    competidores = rng.integers(len(fitness), size=(quantidade, tamanho))
    return competidores[np.arange(quantidade), fitness[competidores].argmax(axis=1)]


def executar_ga(inst: InstanciaQ1, cfg: ConfigGA, seed: int) -> ResultadoGA:
    rng = np.random.default_rng(seed)
    opcoes = inst.opcoes
    taxa_mutacao = cfg.taxa_mutacao if cfg.taxa_mutacao is not None else 1 / inst.n
    n_filhos = cfg.populacao - cfg.elitismo

    pop = _sortear_opcoes(rng, opcoes, cfg.populacao)
    pop[0] = inst.politica_atual()
    pop = reparar(pop, inst, rng)

    resultado = ResultadoGA(politica=pop[0], caixa=-np.inf, geracoes=0, avaliacoes=0)
    sem_melhora = 0

    for geracao in range(1, cfg.geracoes + 1):
        fitness = inst.caixa_total(pop)
        resultado.avaliacoes += len(pop)
        ordem = np.argsort(-fitness)

        resultado.melhor_por_geracao.append(float(fitness[ordem[0]]))
        resultado.media_por_geracao.append(float(fitness.mean()))
        resultado.geracoes = geracao
        if fitness[ordem[0]] > resultado.caixa + 1e-9:
            resultado.caixa, resultado.politica = float(fitness[ordem[0]]), pop[ordem[0]].copy()
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= cfg.paciencia or geracao == cfg.geracoes:
            break

        pais_a = pop[_torneio(fitness, cfg.torneio, n_filhos, rng)]
        pais_b = pop[_torneio(fitness, cfg.torneio, n_filhos, rng)]

        cruza = rng.random((n_filhos, 1)) < cfg.taxa_crossover
        mistura = rng.random((n_filhos, inst.n)) < 0.5
        filhos = np.where(cruza & mistura, pais_b, pais_a)

        mutar = rng.random(filhos.shape) < taxa_mutacao
        for i, op in enumerate(opcoes):
            linhas = mutar[:, i]
            filhos[linhas, i] = rng.choice(op, size=int(linhas.sum()))

        pop = np.vstack([pop[ordem[: cfg.elitismo]], reparar(filhos, inst, rng)])

    return resultado
