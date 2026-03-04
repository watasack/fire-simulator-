"""
設定管理モジュール
config.yamlの読み込みとPydanticスキーマによる厳密なバリデーションを担当
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from pydantic import ValidationError

from src.config_schema import AppConfig


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    設定ファイルを読み込み、Pydanticスキーマで全フィールドを検証する。

    Args:
        config_path: 設定ファイルのパス

    Returns:
        検証済み設定辞書（全キーが存在することが保証される）

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        ValueError: 必須パラメータが不足・型不正の場合
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    try:
        validated = AppConfig(**raw)
    except ValidationError as e:
        raise ValueError(
            f"Config validation failed ({config_path}):\n{e}"
        ) from e

    return validated.model_dump()


def get_scenario_config(config: Dict[str, Any], scenario: str = 'standard') -> Dict[str, Any]:
    """
    特定のシナリオ設定を取得

    Args:
        config: 設定辞書
        scenario: シナリオ名（デフォルト: 'standard'）

    Returns:
        シナリオ設定辞書
    """
    if scenario not in config['simulation']:
        raise ValueError(f"Unknown scenario: {scenario}")

    return config['simulation'][scenario]
