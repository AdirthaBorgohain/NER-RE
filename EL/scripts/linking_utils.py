import json
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional, Set

from EL.scripts.file_cache import cached_path


class Entity(NamedTuple):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    types: List[str] = []
    definition: Optional[str] = None

    def __repr__(self):

        rep = ""
        num_aliases = len(self.aliases)
        rep = rep + f"CUI: {self.concept_id}, Name: {self.canonical_name}\n"
        rep = rep + f"Definition: {self.definition}\n"
        rep = rep + f"TUI(s): {', '.join(self.types)}\n"
        if num_aliases > 10:
            rep = (
                    rep
                    + f"Aliases (abbreviated, total: {num_aliases}): \n\t {', '.join(self.aliases[:10])}"
            )
        else:
            rep = (
                    rep + f"Aliases: (total: {num_aliases}): \n\t {', '.join(self.aliases)}"
            )
        return rep


class KnowledgeBase:
    """
    A class representing two commonly needed views of a Knowledge Base:
    1. A mapping from concept_id to an Entity NamedTuple with more information.
    2. A mapping from aliases to the sets of concept ids for which they are aliases.

    Parameters
    ----------
    file_path: str, required.
        The file path to the json/jsonl representation of the KB to load.
    """

    def __init__(
            self,
            file_path: str = None,
    ):
        if file_path is None:
            raise ValueError(
                "Do not use the default arguments to KnowledgeBase. "
                "Instead, use a subclass (e.g UmlsKnowledgeBase) or pass a path to a kb."
            )
        if file_path.endswith("jsonl"):
            raw = (json.loads(line) for line in open(cached_path(file_path)))
        else:
            raw = json.load(open(cached_path(file_path)))

        alias_to_cuis: Dict[str, Set[str]] = defaultdict(set)
        self.cui_to_entity: Dict[str, Entity] = {}

        for concept in raw:
            unique_aliases = set(concept["aliases"])
            unique_aliases.add(concept["canonical_name"])
            for alias in unique_aliases:
                alias_to_cuis[alias].add(concept["concept_id"])
            self.cui_to_entity[concept["concept_id"]] = Entity(**concept)

        self.alias_to_cuis: Dict[str, Set[str]] = {**alias_to_cuis}
