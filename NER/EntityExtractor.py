import spacy
from utils import load_yaml_file


class EntityExtractor:
    def __init__(self):
        self.__project_config = load_yaml_file('./project.yml')
        self.__ner = spacy.load(self.__project_config.vars['trained_model'])

    def get_predictions(self, text: str):
        for doc in self.__ner.pipe(text, disable=["tagger", "parser"]):
            print([(ent.text, ent.label_) for ent in doc.ents])


if __name__ == "__main__":
    extractor = EntityExtractor()
    extractor.get_predictions(text=["CASP1 also activates proinflammatory interleukins, IL1B and IL18, via proteolysis",
                                    "(2) How does the NLRP3 caspase 1 IL 1b axis in the cartilaginous endplates of patients with Modic changes compare with control (trauma) specimens"])
