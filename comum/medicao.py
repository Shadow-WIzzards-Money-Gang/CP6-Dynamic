"""Medição real de tempo (time.perf_counter) e pico de memória (tracemalloc)."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Medicao:
    etapa: str
    segundos: float
    pico_mb: float

    def como_dict(self) -> dict:
        return {"etapa": self.etapa, "segundos": round(self.segundos, 4), "pico_mb": round(self.pico_mb, 3)}


def medir(etapa: str, funcao: Callable[..., Any], *args, **kwargs) -> tuple[Any, Medicao]:
    """Executa `funcao` medindo tempo de parede e pico de memória alocada durante a chamada.

    O tracemalloc acrescenta um pequeno overhead ao tempo; os valores servem para comparar
    etapas e tamanhos de entrada entre si, sempre medidos da mesma forma.
    """
    ja_ativo = tracemalloc.is_tracing()
    if not ja_ativo:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()

    inicio = time.perf_counter()
    resultado = funcao(*args, **kwargs)
    segundos = time.perf_counter() - inicio

    _, pico = tracemalloc.get_traced_memory()
    if not ja_ativo:
        tracemalloc.stop()
    return resultado, Medicao(etapa, segundos, max(pico - base, 0) / 1024**2)
