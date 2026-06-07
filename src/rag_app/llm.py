from __future__ import annotations

import os
from typing import Iterable

from langchain.chat_models import init_chat_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import settings
from .schema import SourceResponse


class LLMService:
    SYSTEM_PROMPT = (
        "You are a factual assistant. Use only the provided source documents to answer the user query. "
        "If the answer cannot be found in the documents, say so instead of hallucinating."
    )

    def __init__(self) -> None:
        self.settings = settings
        self.llm = self._load_llm()

    def _load_llm(self):
        if self.settings.LLM_PROVIDER == "openai":
            if self.settings.OPENAI_API_KEY:
                os.environ.setdefault("OPENAI_API_KEY", self.settings.OPENAI_API_KEY)
            if self.settings.OPENAI_API_BASE:
                os.environ.setdefault("OPENAI_API_BASE", self.settings.OPENAI_API_BASE)
            return init_chat_model(
                self.settings.LLM_MODEL,
                model_provider="openai",
                temperature=self.settings.TEMPERATURE,
                max_tokens=self.settings.MAX_TOKENS,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.settings.LOCAL_MODEL_PATH, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(self.settings.LOCAL_MODEL_PATH)
        text_gen = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            truncation=True,
        )

        class HFTextGenerator:
            def __init__(self, pl):
                self.pl = pl

            def __call__(self, prompt: str) -> str:
                out = self.pl(prompt, max_new_tokens=256)
                if isinstance(out, list) and out:
                    # transformers text-generation pipelines typically return a list of dicts
                    first = out[0]
                    return first.get("generated_text", str(first))
                return str(out)

        return HFTextGenerator(text_gen)

    def _build_prompt(self, query: str, sources: Iterable[SourceResponse]) -> str:
        context_blocks = []
        for index, source in enumerate(sources, start=1):
            metadata_parts = []
            if source.metadata.title:
                metadata_parts.append(f"title={source.metadata.title}")
            if source.metadata.source:
                metadata_parts.append(f"source={source.metadata.source}")
            if source.metadata.extra:
                metadata_parts.append(
                    ", ".join(f"{key}={value}" for key, value in source.metadata.extra.items())
                )

            metadata = ", ".join(metadata_parts) if metadata_parts else ""
            context_blocks.append(
                f"Source {index}\n{source.text}\nMetadata: {metadata}".strip()
            )

        context = "\n\n".join(context_blocks)
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer in a concise, factual way and cite the sources if possible."
        )

    def generate_answer(self, query: str, sources: list[SourceResponse]) -> str:
        prompt = self._build_prompt(query, sources)
        if self.settings.LLM_PROVIDER == "openai":
            response = self.llm.invoke(prompt)
            return str(response).strip()

        return self.llm(prompt).strip()
