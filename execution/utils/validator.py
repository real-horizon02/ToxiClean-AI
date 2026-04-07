"""
utils/validator.py
------------------
Shared input validation helpers using Pydantic v2.

Usage:
    from execution.utils.validator import validate_text_input, TextInput
    payload = validate_text_input({"text": "Hello world", "user_id": "u_123"})
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class TextInput(BaseModel):
    """Generic validated text payload for moderation / analysis tasks."""

    text: str = Field(..., min_length=1, max_length=10_000)
    user_id: str = Field(..., min_length=1, max_length=128)
    language: Optional[str] = Field(default="en", pattern=r"^[a-z]{2}$")

    @field_validator("text")
    @classmethod
    def no_null_bytes(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("Null bytes not allowed in text input.")
        return v.strip()

    @field_validator("user_id")
    @classmethod
    def alphanumeric_user_id(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("user_id must be alphanumeric (underscores and dashes allowed).")
        return v


def validate_text_input(data: dict) -> TextInput:
    """
    Parse and validate raw dict into a TextInput model.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    return TextInput(**data)
