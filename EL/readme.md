## Entity Linker Module

- To generate all indexes and vectors from Knowledge Base JSONL file:
	* `python scripts/create_linker.py --kb_path assets/custom_kb.jsonl --output_path indexes/`


- Format of each JSONL object in Knowledge Base JSONL file (sample file is present in assets/custom_kb.jsonl):
	{
		"concept_id": "The ID for the concept",
		"canonical_name": "MyEntity",
		"aliases": ["List of alternative ways to refer to the entity"],
		"definition": "Longer form def of entity", # optional
		"types": ["The type of the entity"] # optional
	}
