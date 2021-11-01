import yaml
import spacy
import scispacy

from scripts.rel_pipe import make_relation_extractor, score_relations
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# helper functions and classes
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RelationExtractor:
    def __init__(self):
        self.__project_config = self.load_yaml_file('./project.yml')
        self.__ner = spacy.load("en_ner_craft_md")
        self.__re = spacy.load(self.__project_config.vars['trained_model'])

    @staticmethod
    def load_yaml_file(file_name):
        with open(file_name) as file:
            doc_dict = yaml.full_load(file)
        return DotDict(doc_dict)

    def get_predictions(self, text: str, threshold: float = 0.4):
        doc = self.__ner(text)
        print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")
        for name, proc in self.__re.pipeline:
            doc = proc(doc)

        for value, rel_dict in doc._.rel.items():
            for sent in doc.sents:
                for e in sent.ents:
                    for b in sent.ents:
                        if e.start == value[0] and b.start == value[1]:
                            if rel_dict['Binds'] >= threshold or rel_dict['Regulates'] >= threshold:
                                print(f"entities: {e.text, b.text} --> predicted relation: {rel_dict}")
                            # print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")


if __name__ == "__main__":
    extractor = RelationExtractor()
    extractor.get_predictions(text="CASP1 also activates proinflammatory interleukins, IL1B and IL18, via proteolysis")
