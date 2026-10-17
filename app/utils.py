"""
utils.py — Input validation and response helpers.
"""

from fastapi import HTTPException
from typing import Any


def validate_range(value: float, name: str, lo: float, hi: float):
    if not (lo <= value <= hi):
        raise HTTPException(
            status_code=422,
            detail=f"'{name}' must be between {lo} and {hi}. Got: {value}",
        )


def success_response(data: Any, model_name: str) -> dict:
    return {"status": "success", "model": model_name, "prediction": data}


def error_response(message: str) -> dict:
    return {"status": "error", "message": message}
