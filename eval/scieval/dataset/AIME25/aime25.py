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


def _default_local_repo_dir(repo_id: str) -> str:
    data_root = os.environ.get("SCIEVAL_DATA_ROOT", "/data/home/scyb546/datasets")
    return osp.join(data_root, repo_id.replace("/", "__"))


def _extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    # Prefer evalscope's official math extractor (handles \\boxed{...} etc.)
    try:
        from evalscope.metrics.math_parser import extract_answer
        from .math_normalize import normalize_answer
        extracted = extract_answer(text)
        filtered_pred = normalize_answer(extracted)
        return filtered_pred
        
    except Exception:
        pass

    # Fallback: last integer found anywhere in the text
    hits = re.findall(r"[+-]?[0-9]+", text)
    if hits:
        return hits[-1]

    return None

def _normalize_aime25_answer(ans: Any) -> str:
    #* AIME25 ground truth is a 1-3 digit integer string.
    if ans is None:
        return ""
    s = str(ans).strip()
    m = re.search(r"^[+-]?[0-9]{1,3}$", s)
    if m:
        return m.group(0)

    #* Fallback for values that might have extra text but contain a number.
    m_fallback = re.search(r"[+-]?[0-9]+", s)
    if m_fallback:
        return m_fallback.group(0)

    return 



def _get_column(ds, required: str, fallbacks: Tuple[str, ...] = ()) -> str:
    #* Prefer the required canonical column name; fall back to a small alias list.
    cols = list(getattr(ds, "column_names", []))
    if required in cols:
        return required
    for name in fallbacks:
        if name in cols:
            return name
    raise KeyError(f"Missing required column '{required}' (tried {required!r} + {fallbacks}) in dataset columns: {cols}")


class AIME25(TextBaseDataset):
    TYPE = "TEXT"
    MODALITY = "TEXT"
    HF_DATASET = "math-ai/aime25"

    def __init__(
        self,
        dataset: str = "AIME25",
        split: str = "test",
        prefer_local: bool = True,
        local_repo_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.prefer_local = prefer_local
        self.local_repo_dir = Path(local_repo_dir).expanduser() if local_repo_dir else None

        #* Map id -> sample row for evaluation lookup.
        self._samples: Dict[str, Dict[str, Any]] = {}
        super().__init__(dataset=dataset)

    @classmethod
    def supported_datasets(cls):
        return ["AIME25"]

    def _resolve_dataset_source(self) -> Tuple[str, Dict[str, Any]]:
        local_path = str(self.local_repo_dir) if self.local_repo_dir is not None else _default_local_repo_dir(self.HF_DATASET)
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
            answer = _normalize_aime25_answer(row.get(answer_col)) #* normalize ground truth answer
            records.append({"index": qid, "id": qid, "problem": problem, "answer": answer})

        return pd.DataFrame(records)

    def post_build(self, dataset: str) -> None:
        #* Build id->row mapping for evaluation.
        self._samples = {str(row["id"]): row for row in self.data.to_dict(orient="records")}


    def build_prompt(self, line) -> List[Dict[str, str]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        problem = str(line.get("problem", "")).strip()

        prompt = (
            "Solve the following math problem step by step. Put your answer inside \\boxed{}.\n\n"
            f"{problem}\n\n"
            "Remember to put your answer inside \\boxed{}."
        )

        return [dict(type="text", value=prompt)]

    def evaluate(self, eval_file: str, **judge_kwargs) -> Dict[str, Any]:
        from evalscope.metrics.math_parser import extract_answer
        from .grader import grade_answer
        data = load(eval_file)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        if not isinstance(data, pd.DataFrame):
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
            pred_extracted = _extract_answer(pred_raw)            
            hit = grade_answer(extract_answer(pred_extracted), gt)

            per_row.append({"id": qid, "prediction": pred_extracted, "answer": gt, "hit": hit})
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
