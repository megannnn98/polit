"""Загрузка языковой модели: 4-битная квантизация (GPU) или float32 (CPU)."""

from __future__ import annotations

import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Конфигурация загрузки модели."""
    model_id: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 42
    device: str = "auto"


def load_model(config: ModelConfig):
    """Загружает модель и токенизатор.

    На GPU: 4-битная квантизация через BitsAndBytes.
    На CPU: float32 без квантизации.

    Args:
        config: конфигурация модели

    Returns:
        tuple (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = config.device != "cpu" and torch.cuda.is_available()

    if use_cuda:
        from transformers import BitsAndBytesConfig

        logger.info("Loading model %s with 4-bit quantization (CUDA)...", config.model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        logger.info("Loading model %s in float32 (CPU)...", config.model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    model.eval()

    logger.info("Model loaded successfully on %s", model.device)
    return model, tokenizer
