import yaml
import spacy

# from scripts.rel_pipe import make_relation_extractor, score_relations
# from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# helper functions and classes
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EntityExtractor:
    def __init__(self):
        self.__project_config = self.load_yaml_file('./project.yml')
        self.__ner = spacy.load(self.__project_config.vars['trained_model'])

    @staticmethod
    def load_yaml_file(file_name):
        with open(file_name) as file:
            doc_dict = yaml.full_load(file)
        return DotDict(doc_dict)

    def get_predictions(self, text: str):
        for doc in self.__ner.pipe(text, disable=["tagger", "parser"]):
            print([(ent.text, ent.label_) for ent in doc.ents])


if __name__ == "__main__":
    extractor = EntityExtractor()
    extractor.get_predictions(text=["CASP1 also activates proinflammatory interleukins, IL1B and IL18, via proteolysis",
                                    "(2) How does the NLRP3 caspase 1 IL 1b axis in the cartilaginous endplates of patients with Modic changes compare with control (trauma) specimens"])
