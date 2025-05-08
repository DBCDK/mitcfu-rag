#!/usr/bin/env python
"""
:mod:`mitcfu_rag.rag -- interface for rag models

All rag models must inherit from this class and implement the abstractmethods
"""
from typing import Generator, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel, confloat, StringConstraints, conint
from typing import List, Annotated

@dataclass
class Reference():
    id: str
    article_headline: str
    article_link: str
    score: float
    text: str
    chunk: str
    
    def __str__(self):
        return f"""
id: {self.id} \n
{self.article_headline} \n
{self.article_link} \n
""{self.text}""
                """
    
    def __hash__(self):
        return hash(self.article_link)
    
    def __eq__(self, another_reference):
        if not isinstance(another_reference, Reference):
            raise TypeError('Can only compare two References')
        if self.article_link == another_reference.article_link:
            return True
        else:
            return False
    
    def __ne__(self, another_reference):
        if not isinstance(another_reference, Reference):
            raise TypeError('Can only compare two References')
        if self.article_link != another_reference.article_link:
            return True
        else:
            return False

class RAG(ABC):

    def __call__(self, messages: list[str], *args, **kwargs) -> str:
        return self.get_response(messages)

    @abstractmethod
    def get_response(self, messages: list[dict[str, Any]], *args, **kwargs) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        pass

    @abstractmethod
    def stream_response(self, messages: list[dict[str, Any]], *args, **kwargs) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        pass

    @abstractmethod
    def evaluate(self, messages: list[str]) -> tuple[list[Reference], str]:
        """
        Takes a list of chat messages as input and returns retrieved references given to the generator
        and the generated response for evaluation.
        """
        pass
        
class Parser(ABC):

    def __call__(self, messages: list[str], *args, **kwargs) -> list[str]:
        return self.preprocess(messages)
    
    @abstractmethod
    def preprocess(self, messages: list[str], *args, **kwargs) -> list[str]:
        pass
    
class Retriever(ABC):

    def __call__(self, messages: list[str], *args, **kwargs):
        return self.retrieve(messages)

    @abstractmethod
    def retrieve(self, messages: list[str], *args, **kwargs)  -> tuple[list[float], list[Reference]]:
        """"
        returns similarity scores and a list of references.
        """
        pass

class Ensembler(ABC):
    retrievers: list[Retriever]
    
    #to use ensemblers with the same interface as retrievers
    def __call__(self, retrievers: list[Retriever], *args, **kwargs) -> tuple[list[float], list[Reference]]:
        return None, self.ensemble(retrievers)

    @abstractmethod
    def ensemble(self, messages: list[str], *args, **kwargs) -> list[Reference]:
        """
        Ensembles the results of a list of retrievers using the retrieve function.
        Retrievers are taken defined in the class initialization.
        """
        pass
    
    @abstractmethod
    def ensemble_by_ranked_docs(self, ref_lists: list[list[Reference]], *args, **kwargs) -> list[Reference]:
        """
        Ensembles the lists of ranked references from different retrievers.
        This function is useful if retrievers will have further input parameters 
        than the default ones defined in the retrieve function in the Retriever class.
        """
        pass
    
    def retrieve(self, messages: list[str], *args, **kwargs) -> tuple[None, list[Reference]]:
        """
        For testing/ comparing different retrieval methods to each other,
        class needs to be compatible with Retriever.retrieve(), 
        returning a tuple of (None (instead of scores), reranked references).
        """
        return None, self.ensemble(messages, *args, **kwargs)
  
    
class Generator(ABC):
    
    def __call__(self, references: list[Reference], query: str, *args, **kwargs) -> str:
        return self.generate(references, query)

    @abstractmethod
    def generate(self, references: list[Reference], query: str, *args, **kwargs) -> str:
        pass

class Validator(ABC):
    
    def __call__(self, generated_response: str, references: list[Reference], query: str, *args, **kwargs) -> bool:
        return self.validate(generated_response, references, query)

    @abstractmethod
    def validate(self, generated_response: str, references: list[Reference], query: str, *args, **kwargs) -> bool:
        pass

    def validate_references(self, references: list[Reference], query: str, limit: int, *args, **kwargs) -> bool:
        """
        Validates if the references are relevant to the query.
        """
        pass

class Summarizer(ABC):
    
    def __call__(self, current_summary: str, query: str, answer: str, *args, **kwargs) -> str:
        return self.summarize(current_summary, query, answer)

    @abstractmethod
    def summarize(self, current_summary: str, query: str, answer: str, *args, **kwargs) -> str:
        pass

# Grammer til llm:
"""
response_str = self.client.text_generation(
            prompt,
            max_new_tokens=400,
            grammar={"type": "json", "value": AnswerWithSource.model_json_schema()},
            temperature=0.1,
            return_full_text=False,
        )
"""

# Grammar til validator, der skal returnere en score mellem 1 og 4:
class AnswerWithNumber(BaseModel):
    score: Annotated[int, conint(ge=1, le=4)]

# Grammar til en generator, der for hver kilde skal returnere kildens titel, en genereret beskrivelse,
# et link til kilden og en score mellem 1 og 4:
class SourcesWithScore(BaseModel):
    titel: Annotated[str, StringConstraints(max_length=50)]
    beskrivelse: Annotated[str, StringConstraints(max_length=125)]
    link: Annotated[str, StringConstraints(max_length=50)]
    score: Annotated[int, conint(ge=1, le=4)]

# Grammar til en generator, der skal returnere svaret på brugerens spørgsmål og en liste af kilder:
class AnswerWithSource(BaseModel):
    dit_svar: Annotated[str, StringConstraints(max_length=250)]
    kilder: List[Annotated[str, StringConstraints(max_length=50)]]
