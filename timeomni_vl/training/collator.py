from typing import Any, Dict, List


class TSUMMCollator:
    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images_src = [item.get("image_src") for item in batch]
        images_tgt = [item.get("image_tgt") for item in batch]
        instructions = [item.get("instruction", "") for item in batch]
        cots = [item.get("cot", "") for item in batch]
        answers = [item.get("answer", "") for item in batch]
        tasks = [item.get("task", "generation") for item in batch]

        return {
            "images_src": images_src,
            "images_tgt": images_tgt,
            "instructions": instructions,
            "cots": cots,
            "answers": answers,
            "tasks": tasks,
        }
