import json
import typer
import warnings
from wasabi import Printer
from pathlib import Path

import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
msg = Printer()


def main(json_loc_train: Path, json_loc_dev: Path, json_loc_test: Path, train_file: Path, dev_file: Path,
         test_file: Path):
    docs = {"train": [], "dev": [], "test": []}
    file_paths = {"train": train_file, "dev": dev_file, "test": test_file}

    for set_type, json_loc in zip(['train', 'dev', 'test'], [json_loc_train, json_loc_dev, json_loc_test]):
        with open(json_loc, 'r', encoding='utf8') as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                doc = nlp.make_doc(example['text'])
                ents = []
                for s in example['spans']:
                    if all([s.get(t) for t in ["start", "end", "label"]]):
                        span = doc.char_span(s["start"], s["end"], label=s["label"])
                        if span is None:
                            warn = f"Skipping entity [{s['start']}, {s['end']}, {s['label']}] in the following text " \
                                   f"because the character span '{doc.text[s['start']:s['end']]}' does not align with " \
                                   f"token boundaries: \n\n{repr(doc.text)}\n"
                            msg.warn(warn)
                        else:
                            ents.append(span)
                doc.ents = ents
                docs[set_type].append(doc)

        docbin = DocBin(docs=docs[set_type], store_user_data=True)
        docbin.to_disk(file_paths[set_type])
        msg.info(
            f"{len(docs[set_type])} NER examples generated "
        )


if __name__ == "__main__":
    typer.run(main)
