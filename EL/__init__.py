import os
from pathlib import Path

base_dir = Path(__file__).parent.absolute().parent

configs = dict()

configs['kb_path'] = os.path.join(base_dir, 'EL/assets/custom_kb.jsonl')
configs['index_path'] = os.path.join(base_dir, 'EL/indexes/')
configs['ann_index'] = os.path.join(configs['index_path'], 'nmslib_index.bin')
configs['tfidf_vectorizer'] = os.path.join(configs['index_path'], 'tfidf_vectorizer.joblib')
configs['tfidf_vectors'] = os.path.join(configs['index_path'], 'tfidf_vectors_sparse.npz')
configs['concept_aliases_list'] = os.path.join(configs['index_path'], 'concept_aliases.json')


