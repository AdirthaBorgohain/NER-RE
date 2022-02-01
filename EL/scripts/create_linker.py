import os
import argparse

from EL.scripts.linking_utils import KnowledgeBase
from EL.scripts.candidate_generation import create_tfidf_ann_index


def main(kb_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    kb = KnowledgeBase(kb_path)
    create_tfidf_ann_index(output_path, kb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kb_path',
        help="Path to the KB file."
    )
    parser.add_argument(
        '--output_path',
        help="Path to the output directory."
    )

    args = parser.parse_args()
    main(args.kb_path, args.output_path)
