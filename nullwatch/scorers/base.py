from abc import ABC, abstractmethod

from ..models import Eval


class BaseScorer(ABC):
    @property
    @abstractmethod
    def eval_key(self) -> str: ...

    @property
    @abstractmethod
    def scorer_name(self) -> str: ...

    @abstractmethod
    def score(self, run_id: str, **kwargs) -> Eval: ...
