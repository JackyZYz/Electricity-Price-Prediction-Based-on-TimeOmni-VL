from typing import Dict, Optional

from PIL import Image

from timeomni_vl.models.adapter import BackboneAdapter
from timeomni_vl.utils.text import extract_between_tags


class UnderstandingInferencer:
    def __init__(
        self,
        adapter: BackboneAdapter,
        system_prompt: str = None,
    ):
        self.adapter = adapter
        self.system_prompt = system_prompt or self._default_prompt()

    def infer(
        self,
        image: Image.Image,
        question: str,
    ) -> Dict[str, str]:
        raw = self.adapter.understand(
            image,
            question,
            system_prompt=self.system_prompt,
        )
        cot = self._extract_cot(raw)
        answer = self._extract_answer(raw)
        return {"raw": raw, "cot": cot, "answer": answer}

    def _extract_cot(self, text: str) -> str:
        return extract_between_tags(text, "think")

    def _extract_answer(self, text: str) -> str:
        cot = self._extract_cot(text)
        answer = text.replace(f"<think>{cot}</think>", "").strip()
        return answer if answer else ""

    def _default_prompt(self) -> str:
        return (
            "You are analyzing time-series images (TS-images). "
            "Think step by step inside <think> tags and provide a concise final answer."
        )
