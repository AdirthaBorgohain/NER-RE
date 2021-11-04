import os
import spacy
from scispacy.linking import *
from EL.scripts.create_linker import main
from scispacy.linking_utils import KnowledgeBase
from scispacy.candidate_generation import DEFAULT_PATHS, DEFAULT_KNOWLEDGE_BASES, LinkerPaths

if not os.path.isdir('EL/indexes/'):
    main(kb_path='EL/assets/custom_kb.jsonl', output_path='EL/indexes/')

linker_paths = LinkerPaths(
    ann_index="EL/indexes/nmslib_index.bin",
    tfidf_vectorizer="EL/indexes/tfidf_vectorizer.joblib",
    tfidf_vectors="EL/indexes/tfidf_vectors_sparse.npz",
    concept_aliases_list="EL/indexes/concept_aliases.json",
)


class CustomKnowledgeBase(KnowledgeBase):
    def __init__(self, file_path: str = "EL/assets/custom_kb.jsonl", ):
        super().__init__(file_path)


DEFAULT_PATHS["entity_linker"] = linker_paths
DEFAULT_KNOWLEDGE_BASES["entity_linker"] = CustomKnowledgeBase


def attach_linker(spacy_model: spacy.language):
    spacy_model.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "entity_linker",
                                                    "filter_for_definitions": False, "threshold": "0.5"})
    return spacy_model
