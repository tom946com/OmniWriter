"""
模型客户端模块
提供统一的 LLM 和嵌入模型调用接口
"""

import os
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from openai import (
    APIError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from dotenv import load_dotenv

from src.utils.logs import logger

load_dotenv()


class ModelClient:
    """
    统一的模型客户端
    
    支持 LLM 调用和嵌入模型调用，配置从环境变量读取
    """
    
    _instance: Optional["ModelClient"] = None
    _llm: Optional[ChatOpenAI] = None
    _llms: List[Dict[str, Any]] = []
    _current_model_index: int = 0
    
    def __new__(cls) -> "ModelClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._llm is not None:
            return
        
        self._init_llms()
    
    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        value = os.getenv(key, default)
        if value:
            value = value.strip().strip("'").strip('"').strip(",")
        return value
    
    def _init_llms(self):
        base_url = self._get_env("LLM_BASE_URL")
        api_key = self._get_env("LLM_BASE_API_KEY")
        max_tokens = int(self._get_env("DEFAULT_LLM_MAX_TOKENS", "4000"))
        temperature = float(self._get_env("DEFAULT_LLM_TEMPERATURE", "0.7"))
        
        models_config = [
            {
                "name": "DeepSeek-V3.2",
                "llm": ChatOpenAI(
                    model="deepseek-ai/DeepSeek-V3.2",
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            },
            {
                "name": "GLM-5",
                "llm": ChatOpenAI(
                    model="ZhipuAI/GLM-5",
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            },
            {
                "name": "DeepSeek-R1-0528",
                "llm": ChatOpenAI(
                    model="deepseek-ai/DeepSeek-R1-0528",
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            },
        ]
        
        self._llms = models_config
        
        default_name = self._get_env("DEFAULT_LLM_NAME", "GLM-5")
        try:
            self._current_model_index = next(
                i for i, m in enumerate(models_config) if m["name"] == default_name
            )
            self._llm = models_config[self._current_model_index]["llm"]
        except StopIteration:
            self._current_model_index = 0
            self._llm = models_config[0]["llm"]
            logger.warning(
                "default_model_not_found",
                requested_model=default_name,
                using_model=models_config[0]["name"]
            )
    
    def _get_next_model_index(self) -> int:
        total_models = len(self._llms)
        next_index = (self._current_model_index + 1) % total_models
        return next_index
    
    def _switch_to_next_model(self) -> bool:
        try:
            current_model_name = self._llms[self._current_model_index]["name"]
            next_index = self._get_next_model_index()
            next_model_entry = self._llms[next_index]
            
            logger.warning(
                "switching_model",
                from_model=current_model_name,
                to_model=next_model_entry["name"]
            )
            
            self._current_model_index = next_index
            self._llm = next_model_entry["llm"]
            logger.info("model_switched_successfully", new_model=next_model_entry["name"])
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e), exc_info=True)
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def _call_llm_with_retry(self, messages: List) -> Any:
        if not self._llm:
            logger.error("llm_not_initialized")
            raise RuntimeError("llm not initialized")
        
        current_model_name = self._llms[self._current_model_index]["name"]
        
        try:
            response = await self._llm.ainvoke(messages)
            return response
        except (RateLimitError, APITimeoutError, APIError) as e:
            logger.warning(
                "llm_call_failed_retrying",
                model=current_model_name,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_strategy="exponential_backoff"
            )
            raise
        except OpenAIError as e:
            logger.error(
                "llm_call_failed",
                model=current_model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def call_llm(
        self,
        messages: List,
        model_name: Optional[str] = None,
        **model_kwargs,
    ) -> Any:
        if model_name:
            try:
                model_entry = next(m for m in self._llms if m["name"] == model_name)
                self._llm = model_entry["llm"]
                self._current_model_index = next(
                    i for i, m in enumerate(self._llms) if m["name"] == model_name
                )
            except StopIteration:
                logger.error("requested_model_not_found", model_name=model_name)
                raise ValueError(f"model '{model_name}' not found")
        
        total_models = len(self._llms)
        models_tried = 0
        starting_index = self._current_model_index
        last_error = None
                
        while models_tried < total_models:
            current_model_name = self._llms[self._current_model_index]["name"]
            try:
                response = await self._call_llm_with_retry(messages)
                print("llm_call_success")
                return response
            except Exception  as e:
                last_error = e
                models_tried += 1
                
                logger.error(
                    "model_failed_after_retries",
                    model=current_model_name,
                    models_tried=models_tried,
                    total_models=total_models
                )
                
                if models_tried >= total_models:
                    logger.critical(
                        "all_models_failed",
                        models_tried=models_tried,
                        starting_model=self._llms[starting_index]["name"]
                    )
                    break
                
                if not self._switch_to_next_model():
                    logger.error("model_switch_aborted")
                    break
        
        raise RuntimeError(
            f"failed to get response from llm after trying {models_tried} models. last error: {str(last_error)}"
        )
    
    @property
    def llm(self) -> ChatOpenAI:
        return self._llm



model_client = ModelClient()
