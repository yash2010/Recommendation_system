from abc import ABC, abstractmethod

class BaseExpander(ABC):

    @abstractmethod
    def expand(self, query: str) -> str:
        raise NotImplementedError
    
    def should_expand(self, query: str)->str:
        vague_words = {
            "something", "anything", "good", "great", "nice",
            "interesting", "fun", "feel", "mood", "vibe",
            "like", "similar", "kind", "type", "sort"}
        words = query.lower().split()
        if len(words) <= 5:
            return True
        if any(word in vague_words for word in words):
            return True

        return False