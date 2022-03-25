import spacy
from EL.scripts.linker_utils import attach_linker, get_linker


class EntityLinker:
    def __init__(self) -> None:
        self.__nlp = spacy.blank("en")
        self.__linker = get_linker()

    def get_ent_info(self, entity_text, entity_label) -> str:
        try:
            ent_doc = self.__nlp.make_doc(entity_name)
            ent_doc.set_ents(
                [ent_doc.char_span(0, len(entity_name), label=entity_label)])
            linked_doc = self.__linker(ent_doc)
            print('Entities: ', linked_doc.ents)
            return self.__linker.kb.cui_to_entity[linked_doc.ents[0]._.kb_ents[0][0]]
        except:
            return ""


if __name__ == '__main__':
    entity_name = "1 Sarcosine 8 Isoleucine Angiotensin II"
    linker = EntityLinker()
    print(linker.get_ent_info(entity_text=entity_name, entity_label='GGP'))
