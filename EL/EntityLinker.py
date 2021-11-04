import os
import spacy
from scripts.linker_utils import attach_linker


class EntityLinker:
    def __init__(self) -> None:
        self.__nlp = spacy.load("en_ner_bionlp13cg_md")
        self.__nlp = attach_linker(self.__nlp)

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
