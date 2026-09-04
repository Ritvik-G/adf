"""
Shared config and dataset loading.
Paths are resolved relative to this file, so scripts work regardless of CWD.
"""

import json
import os
from dotenv import load_dotenv

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

load_dotenv(os.path.join(ROOT_DIR, ".env"))


def load_dataset():
    """Load train and test datasets"""
    with open(os.path.join(DATA_DIR, "train.json"), "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(DATA_DIR, "test.json"), "r", encoding="utf-8") as f:
        test = json.load(f)
    return train, test


def load_llm_config():
    """Load non-secret LLM config from file, secret API key from environment (.env)"""
    with open(os.path.join(CODE_DIR, "config_llm.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["api_key"] = os.environ.get("GROQ_API_KEY")
    return cfg
