import spacy
from utils import load_yaml_file
from RE.scripts.rel_pipe import make_relation_extractor, score_relations
from RE.scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


class RelationExtractor:
    def __init__(self):
        self.__model = spacy.load('../NER/training/model-best', exclude=['tagger', 'parser'])
        self.__re_project_config = load_yaml_file('./project.yml')
        self.__re = spacy.load(self.__re_project_config.vars['trained_model'], exclude=['tagger', 'parser'])
        self.__model.add_pipe("tok2vec", name="re_tok2vec", source=self.__re)
        self.__model.add_pipe("relation_extractor", source=self.__re, last=True)

    def get_predictions(self, text: str, threshold: float = 0.4):
        doc = self.__model(text)
        print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")

        for value, rel_dict in doc._.rel.items():
            for e in doc.ents:
                for b in doc.ents:
                    if e.start == value[0] and b.start == value[1]:
                        if rel_dict['Binds'] >= threshold:
                            print(f"entities: {e.text, b.text} --> predicted relation: Binds")
                        elif rel_dict['Regulates'] >= threshold:
                            print(f"entities: {e.text, b.text} --> predicted relation: Regulates")


if __name__ == "__main__":
    extractor = RelationExtractor()
    extractor.get_predictions(
        text="The formation of a stable tetrameric DCoH-HNF-1 alpha complex, which required the dimerization domain of HNF-1 alpha, did not change the DNA binding characteristics of HNF-1 alpha, but enhanced its transcriptional activity.")
