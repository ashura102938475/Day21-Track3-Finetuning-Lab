import importlib
import sys


def test_bonus_verifier_is_safe_to_import_with_kernel_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ipykernel", "--f=/tmp/kernel.json"])
    module = importlib.import_module("scripts.bonus_verify")
    assert callable(module.main)


def test_core_verifier_main_accepts_explicit_arguments():
    module = importlib.import_module("scripts.verify")
    assert module.main.__defaults__ == (None,)
