from __future__ import annotations

import os
import os.path as osp
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset  # type: ignore

from ..text_base import TextBaseDataset
from ...smp import dump, load
from ...smp.file import get_intermediate_file_path

# Optional dependency for answer extraction
try:
    from evalscope.metrics.math_parser import extract_answer as evalscope_extract_answer
    HAS_EVALSCOPE = True
except ImportError:
    HAS_EVALSCOPE = False


def _default_local_repo_dir(repo_id: str) -> str:
    data_root = os.environ.get("SCIEVAL_DATA_ROOT", "/data/home/scyb546/datasets")
    return osp.join(data_root, repo_id.replace("/", "__"))


def _extract_answer(text: str) -> Optional[str]:
    """Extracts the final answer, preferring the evalscope logic for \boxed{} format."""
    if not isinstance(text, str):
        return None

    if HAS_EVALSCOPE:
        try:
            # Use evalscope's official extractor which handles \boxed{}, etc.
            result = evalscope_extract_answer(text)
            if result is not None:
                return str(result)
        except Exception:
            # Fallback if evalscope extractor fails
            pass

    # Fallback to original regex for [ANSWER] tag or last integer
    tagged = re.search(r"\[ANSWER\]\s*([+-]?[0-9]+)\s*\[/ANSWER\]", text, flags=re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()

    hits = re.findall(r"[+-]?[0-9]+", text)
    if hits:
        return hits[-1]

    return None


def _normalize_aime24_answer(ans: Any) -> str:
    """Normalizes an answer to a 3-digit string, e.g., '025'."""
    if ans is None:
        return ""
    s = str(ans).strip()

    m = re.search(r"[+-]?[0-9]+", s)
    if not m:
        return ""

    digits = m.group(0)
    if digits.startswith(('+', '-')):
        digits = digits[1:]

    if not digits:
        return ""

    digits = digits[-3:]
    return digits.zfill(3)


def _get_column(ds, required: str, fallbacks: Tuple[str, ...] = ()) -> str:
    cols = list(getattr(ds, "column_names", []))
    if required in cols:
        return required
    for name in fallbacks:
        if name in cols:
            return name
    raise KeyError(
        f"Missing required column '{required}' (tried {required!r} + {fallbacks}) in dataset columns: {cols}"
    )


class AIME24(TextBaseDataset):
    TYPE = "TEXT"
    MODALITY = "TEXT"
    HF_DATASET = "HuggingFaceH4/aime_2024"

    def __init__(
        self,
        dataset: str = "AIME24",
        split: str = "train",
        prefer_local: bool = True,
        local_repo_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.prefer_local = prefer_local
        self.local_repo_dir = Path(local_repo_dir).expanduser() if local_repo_dir else None
        self._samples: Dict[str, Dict[str, Any]] = {}
        super().__init__(dataset=dataset)

    @classmethod
    def supported_datasets(cls):
        return ["AIME24"]

    def _resolve_dataset_source(self) -> Tuple[str, Dict[str, Any]]:
        local_path = (
            str(self.local_repo_dir)
            if self.local_repo_dir is not None
            else _default_local_repo_dir(self.HF_DATASET)
        )
        if self.prefer_local and osp.exists(local_path):
            return local_path, {}
        return self.HF_DATASET, {}

    def load_data(self, dataset: str) -> pd.DataFrame:
        source, kwargs = self._resolve_dataset_source()
        ds = load_dataset(source, split=self.split, **kwargs)

        id_col = _get_column(ds, "id", ("index", "qid", "question_id"))
        problem_col = _get_column(ds, "problem", ("question", "prompt"))
        answer_col = _get_column(ds, "answer", ("target", "label"))

        records: List[Dict[str, Any]] = []
        for row in ds:
            qid = str(row.get(id_col, "")).strip()
            problem = str(row.get(problem_col, "")).strip()
            answer = _normalize_aime24_answer(row.get(answer_col))
            records.append({"index": qid, "id": qid, "problem": problem, "answer": answer})

        return pd.DataFrame(records)

    def post_build(self, dataset: str) -> None:
        self._samples = {str(row["id"]): row for row in self.data.to_dict(orient="records")}

    def build_prompt(self, line) -> List[Dict[str, str]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = str(line.get("problem", "")).strip()

        # Use the official evalscope prompt template
        prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        return [dict(type="text", value=prompt)]

    def evaluate(self, eval_file: str, **judge_kwargs) -> Dict[str, Any]:
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception:
                raise TypeError("Predictions must be a path to a JSON/CSV/TSV file or a pandas-compatible object.")

        id_col = self._detect_id_column(data.columns)
        if id_col is None:
            raise KeyError("Prediction file must contain an 'id' or 'index' column.")
        if "prediction" not in data.columns:
            raise KeyError("Prediction file must contain a 'prediction' column.")

        data = data.sort_values(by=id_col)

        per_row: List[Dict[str, Any]] = []
        hits: List[int] = []

        for _, row in data.iterrows():
            qid = str(row[id_col]).strip()
            sample = self._samples.get(qid)
            if sample is None:
                continue

            gt = str(sample.get("answer", "")).strip()
            pred_raw = str(row["prediction"])
            
            # Use the new extraction logic
            pred_extracted = _extract_answer(pred_raw)
            pred_normalized = _normalize_aime24_answer(pred_extracted)
            
            hit = 1 if pred_normalized == gt and gt != "" else 0

            per_row.append({"id": qid, "prediction": pred_normalized, "answer": gt, "hit": hit})
            hits.append(hit)

        acc = sum(hits) / len(hits) if hits else 0.0

        eval_name_result = get_intermediate_file_path(eval_file, "_result")
        dump(pd.DataFrame(per_row), eval_name_result)

        score = {"acc": acc, "total": len(hits)}
        score_file = get_intermediate_file_path(eval_file, "_acc", "json")
        dump(score, score_file)
        return score

    @staticmethod
    def _detect_id_column(columns: Iterable[str]) -> Optional[str]:
        for candidate in ("id", "index", "question_id"):
            if candidate in columns:
                return candidate
        return None