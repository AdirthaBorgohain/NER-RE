import os
import spacy
from scispacy.candidate_generation import DEFAULT_PATHS, DEFAULT_KNOWLEDGE_BASES
from scispacy.candidate_generation import (CandidateGenerator, LinkerPaths)
from scispacy.linking_utils import KnowledgeBase
from scispacy.linking import *

from scripts.create_linker import main

linker_paths = LinkerPaths(
    ann_index="indexes/nmslib_index.bin",
    tfidf_vectorizer="indexes/tfidf_vectorizer.joblib",
    tfidf_vectors="indexes/tfidf_vectors_sparse.npz",
    concept_aliases_list="indexes/concept_aliases.json",
)


class CustomKnowledgeBase(KnowledgeBase):
    def __init__(self, file_path: str = "assets/custom_kb.jsonl",):
        super().__init__(file_path)


DEFAULT_PATHS["entity_linker"] = linker_paths
DEFAULT_KNOWLEDGE_BASES["entity_linker"] = CustomKnowledgeBase


class EntityLinker:
    def __init__(self) -> None:
        self.__init_indexes()
        self.__nlp = spacy.load("en_ner_bionlp13cg_md")
        self.__nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "entity_linker",
                                                       "filter_for_definitions": False, "threshold": "0.5"})

    def __init_indexes(self):
        if not os.path.isdir('indexes/'):
            main(kb_path='assets/custom_kb.jsonl', output_path='indexes/')

    def get_id(self, entity) -> str:
        doc = self.__nlp(entity)
        try:
            ents = doc.ents[0]
            possible_ents = [ent for ent in ents._.kb_ents]
            most_likely_ent = possible_ents[0]
            return most_likely_ent[0]
        except:
            return ""


if __name__ == '__main__':
    linker = EntityLinker()
    entity_name = "1 Sarcosine 8 Isoleucine Angiotensin II"
    concept_id = linker.get_id(entity_name)
    print(f"\nConcept ID for {entity_name}: {concept_id}")
