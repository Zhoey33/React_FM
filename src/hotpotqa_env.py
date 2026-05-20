"""HotPotQA environment wrapper for ReAct-style evaluation.

Provides Search/Lookup/Finish actions over Wikipedia, following
the original ReAct paper (Yao et al., ICLR 2023).
"""

import functools
import logging
import os
import re
import string
import random

import requests as _requests
import wikipedia
import wikipedia.wikipedia as _wiki_mod

logger = logging.getLogger(__name__)

# --- Proxy only for Wikipedia requests (not LLM) ---
_WIKI_PROXY = os.environ.get("WIKI_PROXY", "")
if _WIKI_PROXY:
    _orig_get = _requests.get

    @functools.wraps(_orig_get)
    def _proxied_get(url, **kwargs):
        if "wikipedia.org" in str(url):
            kwargs.setdefault("proxies", {
                "http": _WIKI_PROXY,
                "https": _WIKI_PROXY,
            })
        return _orig_get(url, **kwargs)

    # Monkey-patch only the reference used by the wikipedia library
    _wiki_mod.requests.get = _proxied_get
    logger.info(f"Wikipedia proxy enabled: {_WIKI_PROXY}")


def normalize_answer(s: str) -> str:
    """Normalize answer for EM/F1 evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class HotPotQAEnv:
    """Wikipedia-based environment for HotPotQA ReAct evaluation."""

    def __init__(self, split: str = "validation", num_examples: int = 500, seed: int = 42):
        self.split = split
        self.num_examples = num_examples
        self.seed = seed
        self._examples = []
        self._idx = 0
        self._current = None
        self._page_content = None
        self._lookup_keyword = None
        self._lookup_list = []
        self._lookup_cnt = 0

    def setup(self):
        """Load dataset and sample examples."""
        logger.info("Loading HotPotQA dataset...")
        ds = None
        # Try loading from ModelScope cache (arrow files)
        import os
        from pathlib import Path as _Path
        cache_dir = _Path(os.path.expanduser(
            "~/.cache/modelscope/hub/datasets/OpenDataLab___hotpot_qa/"
            "default-755683ef7995944c/0.0.0/master"
        ))
        arrow_file = cache_dir / f"hotpot_qa-{self.split}.arrow"
        if arrow_file.exists():
            from datasets import Dataset
            ds = Dataset.from_file(str(arrow_file))
            logger.info(f"Loaded from ModelScope cache: {arrow_file}")
        if ds is None:
            try:
                from modelscope.msdatasets import MsDataset
                ds = MsDataset.load('OpenDataLab/HotpotQA', split=self.split)
                logger.info("Loaded from ModelScope (OpenDataLab/HotpotQA)")
            except Exception:
                from datasets import load_dataset
                ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=self.split)
                logger.info("Loaded from HuggingFace (hotpotqa/hotpot_qa)")
        # Deduplicate by _id (OpenDataLab version may have duplicates)
        seen_ids = set()
        unique = []
        for i in range(len(ds)):
            item = ds[i]
            qid = item.get("_id", item.get("id", str(i)))
            if qid not in seen_ids:
                seen_ids.add(qid)
                unique.append(item)
        logger.info(f"HotPotQA: {len(unique)} unique examples from {self.split}")
        rng = random.Random(self.seed)
        rng.shuffle(unique)
        self._examples = unique[:self.num_examples]
        self._idx = 0
        logger.info(f"HotPotQA: sampled {len(self._examples)} examples")

    def reset(self) -> tuple[str, str, dict]:
        """Reset to next question. Returns (observation, task_type, info)."""
        if self._idx >= len(self._examples):
            raise StopIteration("No more examples")
        self._current = self._examples[self._idx]
        self._idx += 1
        self._page_content = None
        self._lookup_keyword = None
        self._lookup_list = []
        self._lookup_cnt = 0

        question = self._current["question"]
        gold_answer = self._current["answer"]
        q_type = self._current.get("type", "unknown")

        obs = f"Question: {question}"
        info = {
            "question": question,
            "gold_answer": gold_answer,
            "question_type": q_type,
            "question_level": self._current.get("level", ""),
        }
        return obs, q_type, info

    def search(self, entity: str) -> str:
        """Search Wikipedia for entity. Return first 5 sentences."""
        try:
            page = wikipedia.page(entity, auto_suggest=False)
            self._page_content = page.content
            self._lookup_keyword = None
            self._lookup_list = []
            self._lookup_cnt = 0
            sentences = []
            for para in self._page_content.split('\n'):
                para = para.strip()
                if not para:
                    continue
                for s in para.split('. '):
                    sentences.append(s.strip())
                    if len(sentences) >= 5:
                        break
                if len(sentences) >= 5:
                    break
            return '. '.join(sentences[:5]) + '.'
        except wikipedia.exceptions.PageError:
            results = wikipedia.search(entity)
            self._page_content = None
            if results:
                return f"Could not find [{entity}]. Similar: {results}"
            return f"Could not find [{entity}]. No similar results."
        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options[:5] if e.options else []
            self._page_content = None
            return f"Could not find [{entity}]. Similar: {options}"
        except Exception as e:
            self._page_content = None
            return f"Search error: {str(e)[:100]}"

    def lookup(self, keyword: str) -> str:
        """Find next sentence containing keyword on current page."""
        if self._page_content is None:
            return "No page loaded. Use Search first."
        if self._lookup_keyword != keyword:
            self._lookup_keyword = keyword
            self._lookup_list = [
                sent.strip() for sent in self._page_content.split('. ')
                if keyword.lower() in sent.lower()
            ]
            self._lookup_cnt = 0
        if self._lookup_cnt >= len(self._lookup_list):
            return "No more results."
        result = self._lookup_list[self._lookup_cnt]
        self._lookup_cnt += 1
        total = len(self._lookup_list)
        return f"(Result {self._lookup_cnt}/{total}) {result}"

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute action. Returns (observation, reward, done, info)."""
        action = action.strip()
        if action.startswith("Search[") and action.endswith("]"):
            entity = action[7:-1]
            obs = self.search(entity)
            return obs, 0.0, False, {}
        elif action.startswith("Lookup[") and action.endswith("]"):
            keyword = action[7:-1]
            obs = self.lookup(keyword)
            return obs, 0.0, False, {}
        elif action.startswith("Finish[") and action.endswith("]"):
            answer = action[7:-1]
            gold = self._current["answer"]
            em = exact_match_score(answer, gold)
            f1 = f1_score(answer, gold)
            return answer, em, True, {"em": em, "f1": f1, "gold_answer": gold}
        else:
            return "Invalid action. Use Search[entity], Lookup[keyword], or Finish[answer].", 0.0, False, {}

    def skip(self):
        """Skip current example."""
        self._idx += 1

    def close(self):
        pass

    @property
    def total_episodes(self) -> int:
        return len(self._examples)

    @property
    def current_gold_answer(self) -> str:
        if self._current:
            return self._current["answer"]
        return ""
