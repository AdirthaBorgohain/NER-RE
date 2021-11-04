## Relation Extraction Module

- To run the whole flow:
	* `spacy project run all_gpu (Using GPU)`
	* `spacy project run all (Using CPU)`

P.S. : If using CPU, model will not use transformers and will be trained using tok2vec instead.


- All runnable commands are mentioned in project.yaml


- For training, three annotated files are needed: 
	* annotations_train.jsonl
	* annotations_dev.jsonl
	* annotations_test.jsonl
