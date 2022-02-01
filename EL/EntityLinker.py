import spacy
from scripts.linker_utils import attach_linker, get_linker


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
    entity_name = "1 Sarcosine 8 Isoleucine Angiotensin II"
    nlp = spacy.load("en_ner_bionlp13cg_md")
    doc = nlp(entity_name)
    linker = get_linker()
    linked_doc = linker(doc)
    print('Entities: ', linked_doc.ents)
    print('Entity Details: ', linker.kb.cui_to_entity[linked_doc.ents[0]._.kb_ents[0][0]])