import spacy


# helper functions and classes
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NER_RE:
    def __init__(self) -> None:
        self.__pipeline_config = self.load_yaml_file('./pipeline.yml')
        self.__ner = spacy.load(self.__pipeline_config.vars['ner_model'])
        self.__re = spacy.load(self.__pipeline_config.vars['re_model'])

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
                                print(
                                    f"entities: {e.text, b.text} --> predicted relation: {rel_dict}")


if __name__ == "__main__":
    pipeline = NER_RE()
    sample_text = ""
    pipeline.get_predictions(text=sample_text)
