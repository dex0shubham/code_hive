"""
Diagnose why transformers' lazy loader can't resolve RobertaModel.
Prints versions and the real traceback from a direct submodule import.
"""
import sys, traceback

print("python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch import failed:", type(e).__name__, e)

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers import failed:", type(e).__name__, e)

try:
    import safetensors
    print("safetensors:", safetensors.__version__)
except Exception as e:
    print("safetensors import failed:", type(e).__name__, e)

try:
    import tokenizers
    print("tokenizers:", tokenizers.__version__)
except Exception as e:
    print("tokenizers import failed:", type(e).__name__, e)

print()
print("=== direct submodule import (bypasses lazy loader) ===")
try:
    from transformers.models.roberta.modeling_roberta import RobertaModel
    print("RobertaModel direct import: OK")
except Exception:
    traceback.print_exc()

print()
print("=== top-level lazy import ===")
try:
    from transformers import RobertaModel
    print("from transformers import RobertaModel: OK")
except Exception:
    traceback.print_exc()
