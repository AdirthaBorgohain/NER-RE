import spacy
import yaml

from EL.scripts.linker_utils import attach_linker
from RE.scripts.rel_pipe import make_relation_extractor, score_relations
from RE.scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# helper functions and classes
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PipelineModel:
    def __init__(self) -> None:
        self._pipeline_config = self.load_yaml_file('./pipeline.yml')
        self._model = spacy.load(self._pipeline_config.vars['ner_model'])
        self._model = attach_linker(spacy_model=self._model)
        self._re = spacy.load(self._pipeline_config.vars['re_model'])
        self._model.add_pipe(
            self._re.component_names[0], name=f"re_{self._re.component_names[0]}", source=self._re)
        self._model.add_pipe("relation_extractor",
                             source=self._re, last=True)

    @staticmethod
    def load_yaml_file(file_name):
        with open(file_name) as file:
            doc_dict = yaml.full_load(file)
        return DotDict(doc_dict)

    def get_predictions(self, text: str, threshold: float = 0.4):
        doc = self._model(text)
        print(f"Text: {text}\n")
        print(f"Extracted Entities -> {[(e.text, e.label_) for e in doc.ents]}\n")
        linker = self._model.get_pipe("scispacy_linker")
        print(f"Linked Entities in Knowledge Base ->")
        for entity in doc.ents:
            for ent in entity._.kb_ents:
                print(f"Entity: {entity}, {linker.kb.cui_to_entity[ent[0]]}")

        print("\nExtracted Relations ->")
        for value, rel_dict in doc._.rel.items():
            for e in doc.ents:
                for b in doc.ents:
                    if e.start == value[0] and b.start == value[1]:
                        if rel_dict['Binds'] >= threshold:
                            print(
                                f"{e.text, b.text} --> Predicted Relation: Binds")
                        elif rel_dict['Regulates'] >= threshold:
                            print(
                                f"{e.text, b.text} --> Predicted Relation: Regulates")


if __name__ == "__main__":
    pipeline = PipelineModel()
    sample_text = "The up-regulation of RNA was characteristic of an early inducible gene, with " \
                  "maximal upregulation two hours after the addition of BMP-6 and returned to baseline after 24 hours."
    pipeline.get_predictions(text=sample_text)
