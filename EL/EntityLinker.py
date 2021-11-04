import os
import spacy
from scripts.linker_utils import attach_linker


class EntityLinker:
    def __init__(self) -> None:
        self.__nlp = spacy.load("en_ner_bionlp13cg_md")
        self.__nlp = attach_linker(self.__nlp)

    def get_id(self, entity) -> str:
        doc = self.__nlp(entity)
        linker = self.__nlp.get_pipe('scispacy_linker')
        print(linker.kb.cui_to_entity["C0000107"].aliases)
        try:
            ents = doc.ents[0]
            print('ents: ', ents)
            possible_ents = [ent for ent in ents._.kb_ents]
            most_likely_ent = possible_ents[0]
            return most_likely_ent[0]
        except:
            return ""


if __name__ == '__main__':
    linker = EntityLinker()
    entity_name = "BMP-6"
    concept_id = linker.get_id(entity_name)
    print(f"\nConcept ID for {entity_name}: {concept_id}")
