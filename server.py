from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar
from girth import twopl_mml, threepl_mml
from factor_analyzer import FactorAnalyzer

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("psyche-mcp")

_STATE: dict[str, Any] = {
    "tests": {},
}


def _require_test(test_name: str) -> dict[str, Any]:
    if test_name not in _STATE["tests"]:
        raise RuntimeError(
            f"No existe el test '{test_name}'. Usa 'create_test' primero. "
            f"Tests disponibles: {list(_STATE['tests'].keys())}"
        )
    return _STATE["tests"][test_name]


def _p3pl(theta: np.ndarray, a: float, b: float, c: float = 0.0) -> np.ndarray:
    """Probabilidad de respuesta correcta bajo el modelo 3PL (2PL si c=0)."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def _item_information(item: dict[str, float], theta: float) -> float:
    """Informacion de Fisher de un item en un nivel de theta (formula 3PL)."""
    a, b, c = item["a"], item["b"], item["c"]
    p = float(_p3pl(np.array([theta]), a, b, c)[0])
    p = min(max(p, 1e-6), 1 - 1e-6)
    if c == 0.0:
        return (a ** 2) * p * (1 - p)
    # Informacion 3PL (Lord, 1980): pondera por la asintota de guessing
    q = 1 - p
    return (a ** 2) * (q / p) * ((p - c) / (1 - c)) ** 2


@mcp.tool()
def create_test(test_name: str) -> dict[str, Any]:
    """Crea un nuevo test (banco de items) vacio identificado por test_name."""
    if test_name in _STATE["tests"]:
        raise RuntimeError(f"El test '{test_name}' ya existe.")
    _STATE["tests"][test_name] = {"items": {}}
    return {"status": "ok", "test": test_name}


@mcp.tool()
def add_item(
    test_name: str,
    item_id: str,
    a: float = 1.0,
    b: float = 0.0,
    c: float = 0.0,
    construct: str = "",
) -> dict[str, Any]:
    """Agrega (o sobrescribe) un item al banco de un test, con parametros IRT
    iniciales: a=discriminacion, b=dificultad, c=guessing (0 para 2PL)."""
    test = _require_test(test_name)
    test["items"][item_id] = {"a": a, "b": b, "c": c, "construct": construct}
    return {"status": "ok", "test": test_name, "item": item_id}


@mcp.tool()
def list_items(test_name: str) -> dict[str, Any]:
    """Lista los items de un test con sus parametros IRT actuales."""
    test = _require_test(test_name)
    return {"test": test_name, "items": test["items"]}



@mcp.tool()
def item_information(
    test_name: str,
    item_id: str,
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    n_points: int = 41,
) -> dict[str, Any]:
    """Calcula la curva de informacion de Fisher de un item (formula 3PL,
    se reduce a 2PL si guessing=0) a lo largo del rango de theta."""
    test = _require_test(test_name)
    if item_id not in test["items"]:
        raise RuntimeError(f"No existe el item '{item_id}' en '{test_name}'.")
    item = test["items"][item_id]
    theta = np.linspace(theta_min, theta_max, n_points)
    info = np.array([_item_information(item, t) for t in theta])
    return {
        "test": test_name,
        "item": item_id,
        "theta": theta.tolist(),
        "information": info.tolist(),
    }


@mcp.tool()
def test_information_curve(
    test_name: str,
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    n_points: int = 41,
) -> dict[str, Any]:
    """Suma la informacion de todos los items del test en cada nivel de theta.
    Esta curva es la 'profundidad' real del test: donde es alta, el test mide
    con precision; donde es baja, el test es ciego a esa zona del rasgo."""
    test = _require_test(test_name)
    if not test["items"]:
        raise RuntimeError(f"El test '{test_name}' no tiene items todavia.")
    theta = np.linspace(theta_min, theta_max, n_points)
    total_info = np.array([
        sum(_item_information(item, t) for item in test["items"].values())
        for t in theta
    ])
    se = np.where(total_info > 0, 1.0 / np.sqrt(total_info), np.inf)
    return {
        "test": test_name,
        "theta": theta.tolist(),
        "test_information": total_info.tolist(),
        "standard_error": se.tolist(),
        "peak_theta": float(theta[int(np.argmax(total_info))]),
    }


@mcp.tool()
def calibrate_items(
    test_name: str,
    response_matrix: list[list[int]],
    model: str = "2pl",
) -> dict[str, Any]:
    """Estima parametros IRT (a, b, y c si model='3pl') a partir de una matriz
    real de respuestas (filas=personas, columnas=items, 0/1), usando girth.
    Sobrescribe los parametros de los items existentes, en orden."""
    test = _require_test(test_name)
    data = np.array(response_matrix, dtype=float).T  # girth espera items x personas

    if model == "3pl":
        estimates = threepl_mml(data)
        discrimination = estimates["Discrimination"]
        difficulty = estimates["Difficulty"]
        guessing = estimates.get("Guessing", np.zeros_like(discrimination))
    else:
        estimates = twopl_mml(data)
        discrimination = estimates["Discrimination"]
        difficulty = estimates["Difficulty"]
        guessing = np.zeros_like(discrimination)

    item_ids = list(test["items"].keys())
    if len(item_ids) != data.shape[0]:
        item_ids = [f"item_{i+1}" for i in range(data.shape[0])]

    for i, item_id in enumerate(item_ids):
        existing = test["items"].get(item_id, {"construct": ""})
        test["items"][item_id] = {
            "a": float(discrimination[i]),
            "b": float(difficulty[i]),
            "c": float(guessing[i]),
            "construct": existing.get("construct", ""),
        }

    return {
        "status": "ok",
        "test": test_name,
        "model": model,
        "n_items_calibrated": len(item_ids),
    }


@mcp.tool()
def score_respondent(test_name: str, responses: dict[str, int]) -> dict[str, Any]:
    """Estima el nivel del rasgo latente (theta) de una persona via maxima
    verosimilitud, dado un patron de respuestas 0/1 a items ya calibrados."""
    test = _require_test(test_name)
    for item_id in responses:
        if item_id not in test["items"]:
            raise RuntimeError(f"El item '{item_id}' no existe en '{test_name}'.")

    def neg_log_likelihood(theta: float) -> float:
        ll = 0.0
        for item_id, r in responses.items():
            item = test["items"][item_id]
            p = float(_p3pl(np.array([theta]), item["a"], item["b"], item["c"])[0])
            p = min(max(p, 1e-6), 1 - 1e-6)
            ll += r * np.log(p) + (1 - r) * np.log(1 - p)
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-6, 6), method="bounded")
    theta_hat = result.x

    total_info = sum(_item_information(test["items"][iid], theta_hat) for iid in responses)
    se = 1.0 / np.sqrt(total_info) if total_info > 0 else float("inf")

    return {
        "test": test_name,
        "theta_estimate": float(theta_hat),
        "standard_error": float(se),
        "n_items_answered": len(responses),
    }


@mcp.tool()
def run_adaptive_session(
    test_name: str,
    true_theta: float,
    max_items: int = 10,
    se_stop: float = 0.3,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Simula una sesion de test adaptativo (CAT): en cada paso elige, entre
    los items no usados, el que maximiza la informacion en la estimacion
    actual de theta, simula la respuesta segun 'true_theta', y reestima theta.
    Se detiene por max_items o cuando el error estandar cae bajo 'se_stop'."""
    test = _require_test(test_name)
    if not test["items"]:
        raise RuntimeError(f"El test '{test_name}' no tiene items todavia.")

    rng = np.random.default_rng(seed)
    remaining = dict(test["items"])
    answered: dict[str, int] = {}
    theta_hat = 0.0
    history: list[dict[str, Any]] = []

    def neg_log_likelihood(theta: float) -> float:
        ll = 0.0
        for iid, r in answered.items():
            item = test["items"][iid]
            p = float(_p3pl(np.array([theta]), item["a"], item["b"], item["c"])[0])
            p = min(max(p, 1e-6), 1 - 1e-6)
            ll += r * np.log(p) + (1 - r) * np.log(1 - p)
        return -ll

    for step in range(max_items):
        if not remaining:
            break

        best_id = max(remaining, key=lambda iid: _item_information(remaining[iid], theta_hat))
        item = remaining.pop(best_id)

        p_true = float(_p3pl(np.array([true_theta]), item["a"], item["b"], item["c"])[0])
        response = int(rng.random() < p_true)
        answered[best_id] = response

        result = minimize_scalar(neg_log_likelihood, bounds=(-6, 6), method="bounded")
        theta_hat = result.x

        total_info = sum(_item_information(test["items"][iid], theta_hat) for iid in answered)
        se = 1.0 / np.sqrt(total_info) if total_info > 0 else float("inf")

        history.append({
            "step": step + 1,
            "item": best_id,
            "response": response,
            "theta_estimate": float(theta_hat),
            "standard_error": float(se),
        })

        if se <= se_stop:
            break

    return {
        "test": test_name,
        "true_theta": true_theta,
        "n_items_used": len(history),
        "final_theta": history[-1]["theta_estimate"] if history else None,
        "final_se": history[-1]["standard_error"] if history else None,
        "history": history,
    }


@mcp.tool()
def validate_structure(
    test_name: str,
    response_matrix: list[list[float]],
    n_factors: int = 1,
    rotation: str = "oblimin",
) -> dict[str, Any]:
    """Corre un analisis factorial exploratorio (EFA) sobre datos de respuesta
    reales, para chequear si los items cargan segun los constructos que
    predice el marco teorico del test."""
    test = _require_test(test_name)
    data = np.array(response_matrix, dtype=float)

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(data)
    loadings = fa.loadings_
    variance = fa.get_factor_variance()

    item_ids = list(test["items"].keys())
    if len(item_ids) != loadings.shape[0]:
        item_ids = [f"item_{i+1}" for i in range(loadings.shape[0])]

    return {
        "test": test_name,
        "n_factors": n_factors,
        "loadings": {item_ids[i]: loadings[i].tolist() for i in range(len(item_ids))},
        "variance_explained": variance[1].tolist(),
    }


@mcp.tool()
def reset_bank() -> dict[str, Any]:
    """Borra todos los tests y items."""
    _STATE["tests"] = {}
    return {"status": "ok", "message": "Banco de tests reiniciado"}


if __name__ == "__main__":
    mcp.run()
