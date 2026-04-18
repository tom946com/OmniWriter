"""
提示词加载模块
提供统一的 YAML 提示词文件加载功能
"""

from pathlib import Path
from typing import Optional

import yaml


PROMPTS_DIR = Path(__file__).parent


def load_prompt(prompt_name: str) -> str:
    """
    加载 YAML 提示词文件
    
    Args:
        prompt_name: 提示词文件名（不含扩展名），如 "title_decomposer"
        
    Returns:
        提示词内容字符串
        
    Raises:
        FileNotFoundError: 提示词文件不存在
        RuntimeError: 加载失败
    """
    prompt_path = PROMPTS_DIR / f"{prompt_name}.yaml"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)
            return prompt_data.get("content", "")
    except Exception as e:
        raise RuntimeError(f"加载提示词文件失败: {e}")


def load_prompt_with_metadata(prompt_name: str) -> dict:
    """
    加载 YAML 提示词文件，返回完整元数据
    
    Args:
        prompt_name: 提示词文件名（不含扩展名）
        
    Returns:
        包含 name, description, content 等字段的字典
    """
    prompt_path = PROMPTS_DIR / f"{prompt_name}.yaml"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"加载提示词文件失败: {e}")
