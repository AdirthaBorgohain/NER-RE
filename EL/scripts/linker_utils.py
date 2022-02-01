import os
import spacy
from EL import configs
from EL.scripts.linking import *
from EL.scripts.create_linker import main
from EL.scripts.linking_utils import KnowledgeBase
from EL.scripts.candidate_generation import DEFAULT_PATHS, DEFAULT_KNOWLEDGE_BASES, LinkerPaths

if not os.path.isdir(configs['index_path']):
    main(kb_path=configs['kb_path'], output_path=configs['index_path'])

linker_paths = LinkerPaths(
    ann_index=configs['ann_index'],
    tfidf_vectorizer=configs['tfidf_vectorizer'],
    tfidf_vectors=configs['tfidf_vectors'],
    concept_aliases_list=configs['concept_aliases_list'],
)


class CustomKnowledgeBase(KnowledgeBase):
    def __init__(self, file_path: str = configs['kb_path'], ):
        super().__init__(file_path)


DEFAULT_PATHS["entity_linker"] = linker_paths
DEFAULT_KNOWLEDGE_BASES["entity_linker"] = CustomKnowledgeBase


def attach_linker(spacy_model: spacy.language):
    spacy_model.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "entity_linker",
                                                    "filter_for_definitions": False, "threshold": "0.5"})
    return spacy_model


def get_linker():
    linker = EntityLinker(resolve_abbreviations=False, name="custom_linker", filter_for_definitions=False,
                          threshold=0.5, linker_name="entity_linker")
    return linker
