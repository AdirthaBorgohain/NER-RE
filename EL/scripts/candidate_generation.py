import re
import json
import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, NamedTuple, Type

import scipy
import numpy
import joblib
import nmslib
from nmslib.dist import FloatIndex
from sklearn.feature_extraction.text import TfidfVectorizer

from EL.scripts.file_cache import cached_path
from EL.scripts.linking_utils import KnowledgeBase


class LinkerPaths(NamedTuple):
    """
    Encapsulates all the (possibly remote) paths to data for a scispacy CandidateGenerator.
    ann_index: str
        Path to the approximate nearest neighbours index.
    tfidf_vectorizer: str
        Path to the joblib serialized sklearn TfidfVectorizer.
    tfidf_vectors: str
        Path to the float-16 encoded tf-idf vectors for the entities in the KB.
    concept_aliases_list: str
        Path to the indices mapping concepts to aliases in the index.
    """

    ann_index: str
    tfidf_vectorizer: str
    tfidf_vectors: str
    concept_aliases_list: str


DEFAULT_PATHS: Dict[str, LinkerPaths] = {}
DEFAULT_KNOWLEDGE_BASES: Dict[str, Type[KnowledgeBase]] = {}


class MentionCandidate(NamedTuple):
    """
    A data class representing a candidate entity that a mention may be linked to.

    Parameters
    ----------
    concept_id : str, required.
        The canonical concept id in the KB.
    aliases : List[str], required.
        The aliases that caused this entity to be linked.
    similarities : List[float], required.
        The cosine similarities from the mention text to the alias in tf-idf space.

    """

    concept_id: str
    aliases: List[str]
    similarities: List[float]


def load_approximate_nearest_neighbours_index(
        linker_paths: LinkerPaths,
        ef_search: int = 200,
) -> FloatIndex:
    """
    Load an approximate nearest neighbours index from disk.

    Parameters
    ----------
    linker_paths: LinkerPaths, required.
        Contains the paths to the data required for the entity linker.
    ef_search: int, optional (default = 200)
        Controls speed performance at query time. Max value is 2000,
        but reducing to around ~100 will increase query speed by an order
        of magnitude for a small performance hit.
    """
    concept_alias_tfidfs = scipy.sparse.load_npz(
        cached_path(linker_paths.tfidf_vectors)
    ).astype(numpy.float32)
    ann_index = nmslib.init(
        method="hnsw",
        space="cosinesimil_sparse",
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )
    ann_index.addDataPointBatch(concept_alias_tfidfs)
    ann_index.loadIndex(cached_path(linker_paths.ann_index))
    query_time_params = {"efSearch": ef_search}
    ann_index.setQueryTimeParams(query_time_params)

    return ann_index


class CandidateGenerator:
    """
    A candidate generator for entity linking to a KnowledgeBase. Currently, two defaults are available:
     - Unified Medical Language System (UMLS).
     - Medical Subject Headings (MESH).

    To use these configured default KBs, pass the `name` parameter, either 'umls' or 'mesh'.

    It uses a sklearn.TfidfVectorizer to embed mention text into a sparse embedding of character 3-grams.
    These are then compared via cosine distance in a pre-indexed approximate nearest neighbours index of
    a subset of all entities and aliases in the KB.

    Once the K nearest neighbours have been retrieved, they are canonicalized to their KB canonical ids.
    This step is required because the index also includes entity aliases, which map to a particular canonical
    entity. This point is important for two reasons:

    1. K nearest neighbours will return a list of Y possible neighbours, where Y < K, because the entity ids
    are canonicalized.

    2. A single string may be an alias for multiple canonical entities. For example, "Jefferson County" may be an
    alias for both the canonical ids "Jefferson County, Iowa" and "Jefferson County, Texas". These are completely
    valid and important aliases to include, but it means that using the candidate generator to implement a naive
    k-nn baseline linker results in very poor performance, because there are multiple entities for some strings
    which have an exact char3-gram match, as these entities contain the same alias string. This situation results
    in multiple entities returned with a distance of 0.0, because they exactly match an alias, making a k-nn
    baseline effectively a random choice between these candidates. However, this doesn't matter if you have a
    classifier on top of the candidate generator, as is intended!

    Parameters
    ----------
    ann_index: FloatIndex
        An nmslib approximate nearest neighbours index.
    tfidf_vectorizer: TfidfVectorizer
        The vectorizer used to encode mentions.
    ann_concept_aliases_list: List[str]
        A list of strings, mapping the indices used in the ann_index to possible KB mentions.
        This is essentially used a lookup between the ann index and actual mention strings.
    kb: KnowledgeBase
        A class representing canonical concepts from the knowledge graph.
    verbose: bool
        Setting to true will print extra information about the generated candidates.
    ef_search: int
        The efs search parameter used in the index. This substantially effects runtime speed
        (higher is slower but slightly more accurate). Note that this parameter is ignored
        if a preconstructed ann_index is passed.
    name: str, optional (default = None)
        The name of the pretrained entity linker to load. Must be one of 'umls' or 'mesh'.
    """

    def __init__(
            self,
            ann_index: FloatIndex = None,
            tfidf_vectorizer: TfidfVectorizer = None,
            ann_concept_aliases_list: List[str] = None,
            kb: KnowledgeBase = None,
            verbose: bool = False,
            ef_search: int = 200,
            name: str = None,
    ) -> None:

        if name is not None and any(
                [ann_index, tfidf_vectorizer, ann_concept_aliases_list, kb]
        ):
            raise ValueError(
                "You cannot pass both a name argument and other constuctor arguments."
            )

        # Set the name to the default, after we have checked
        # the compatibility with the args above.
        if name is None:
            name = "entity_linker"

        linker_paths = DEFAULT_PATHS.get(name)

        self.ann_index = ann_index or load_approximate_nearest_neighbours_index(
            linker_paths=linker_paths, ef_search=ef_search
        )
        self.vectorizer = tfidf_vectorizer or joblib.load(
            cached_path(linker_paths.tfidf_vectorizer)
        )
        self.ann_concept_aliases_list = ann_concept_aliases_list or json.load(
            open(cached_path(linker_paths.concept_aliases_list))
        )

        self.kb = kb or DEFAULT_KNOWLEDGE_BASES[name]()
        self.verbose = verbose

        # TODO(Mark): Remove in scispacy v1.0.
        self.umls = self.kb

    def nmslib_knn_with_zero_vectors(
            self, vectors: numpy.ndarray, k: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
        This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
        - remove empty vectors from `vectors`.
        - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
        a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
        - extend the list `neighbors` with `None`s in place of empty vectors.
        - return the extended list of neighbors and distances.
        """
        empty_vectors_boolean_flags = numpy.array(vectors.sum(axis=1) != 0).reshape(-1)
        empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
        if self.verbose:
            print(f"Number of empty vectors: {empty_vectors_count}")

        # init extended_neighbors with a list of Nones
        extended_neighbors = numpy.empty(
            (len(empty_vectors_boolean_flags),), dtype=object
        )
        extended_distances = numpy.empty(
            (len(empty_vectors_boolean_flags),), dtype=object
        )

        if vectors.shape[0] - empty_vectors_count == 0:
            return extended_neighbors, extended_distances

        # remove empty vectors before calling `ann_index.knnQueryBatch`
        vectors = vectors[empty_vectors_boolean_flags]

        # call `knnQueryBatch` to get neighbors
        original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k)

        neighbors, distances = zip(
            *[(x[0].tolist(), x[1].tolist()) for x in original_neighbours]
        )
        neighbors = list(neighbors)
        distances = list(distances)

        # neighbors need to be converted to an np.array of objects instead of ndarray of dimensions len(vectors)xk
        # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
        # returns an np.array of objects
        neighbors.append([])
        distances.append([])
        # interleave `neighbors` and Nones in `extended_neighbors`
        extended_neighbors[empty_vectors_boolean_flags] = numpy.array(neighbors, dtype='object')[:-1]
        extended_distances[empty_vectors_boolean_flags] = numpy.array(distances, dtype='object')[:-1]

        return extended_neighbors, extended_distances

    def __call__(
            self, mention_texts: List[str], k: int
    ) -> List[List[MentionCandidate]]:
        """
        Given a list of mention texts, returns a list of candidate neighbors.

        NOTE: Because we include canonical name aliases in the ann index, the list
        of candidates returned will not necessarily be of length k for each candidate,
        because we then map these to canonical ids only.

        NOTE: For a given mention, the returned candidate list might be empty, which implies that
        the tfidf vector for this mention was all zeros (i.e there were no 3 gram overlaps). This
        happens reasonably rarely, but does occasionally.
        Parameters
        ----------
        mention_texts: List[str], required.
            The list of mention strings to generate candidates for.
        k: int, required.
            The number of ann neighbours to look up.
            Note that the number returned may differ due to aliases.

        Returns
        -------
        A list of MentionCandidate objects per mention containing KB concept_ids and aliases
        and distances which were mapped to. Note that these are lists for each concept id,
        because the index contains aliases which are canonicalized, so multiple values may map
        to the same canonical id.
        """
        if self.verbose:
            print(f"Generating candidates for {len(mention_texts)} mentions")

        # tfidf vectorizer crashes on an empty array, so we return early here
        if mention_texts == []:
            return []

        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = datetime.datetime.now()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        if self.verbose:
            print(f"Finding neighbors took {total_time.total_seconds()} seconds")
        batch_mention_candidates = []
        for neighbors, distances in zip(batch_neighbors, batch_distances):
            if neighbors is None:
                neighbors = []
            if distances is None:
                distances = []

            concept_to_mentions: Dict[str, List[str]] = defaultdict(list)
            concept_to_similarities: Dict[str, List[float]] = defaultdict(list)
            for neighbor_index, distance in zip(neighbors, distances):
                mention = self.ann_concept_aliases_list[neighbor_index]
                concepts_for_mention = self.kb.alias_to_cuis[mention]
                for concept_id in concepts_for_mention:
                    concept_to_mentions[concept_id].append(mention)
                    concept_to_similarities[concept_id].append(1.0 - distance)

            mention_candidates = [
                MentionCandidate(concept, mentions, concept_to_similarities[concept])
                for concept, mentions in concept_to_mentions.items()
            ]
            mention_candidates = sorted(mention_candidates, key=lambda c: c.concept_id)

            batch_mention_candidates.append(mention_candidates)

        return batch_mention_candidates


regex_compiled = re.compile(r"\s+|'|-")


def preprocess(word):
    word = word.lower()
    word = regex_compiled.sub("", word)
    return word


def create_tfidf_ann_index(
        out_path: str, kb: KnowledgeBase = None
) -> Tuple[List[str], TfidfVectorizer, FloatIndex]:
    """
    Build tfidf vectorizer and ann index.

    Parameters
    ----------
    out_path: str, required.
        The path where the various model pieces will be saved.
    kb : KnowledgeBase, optional.
        The kb items to generate the index and vectors for.

    """
    tfidf_vectorizer_path = f"{out_path}/tfidf_vectorizer.joblib"
    ann_index_path = f"{out_path}/nmslib_index.bin"
    tfidf_vectors_path = f"{out_path}/tfidf_vectors_sparse.npz"
    uml_concept_aliases_path = f"{out_path}/concept_aliases.json"

    kb = kb

    # nmslib hyperparameters (very important)
    # guide: https://github.com/nmslib/nmslib/blob/master/python_bindings/parameters.md
    # Default values resulted in very low recall.

    # set to the maximum recommended value. Improves recall at the expense of longer indexing time.
    # TODO: This variable name is so hot because I don't actually know what this parameter does.
    m_parameter = 100
    # `C` for Construction. Set to the maximum recommended value
    # Improves recall at the expense of longer indexing time
    construction = 2000
    num_threads = 60  # set based on the machine
    index_params = {
        "M": m_parameter,
        "indexThreadQty": num_threads,
        "efConstruction": construction,
        "post": 0,
    }

    print(
        f"No tfidf vectorizer on {tfidf_vectorizer_path} or ann index on {ann_index_path}"
    )
    concept_aliases = list(kb.alias_to_cuis.keys())

    # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
    # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
    # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
    # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
    print(f"Fitting tfidf vectorizer on {len(concept_aliases)} aliases")

    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=10, dtype=numpy.float32, preprocessor=preprocess
    )

    # tfidf_vectorizer = TfidfVectorizer(
    #     analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=numpy.float32
    # )
    start_time = datetime.datetime.now()
    concept_alias_tfidfs = tfidf_vectorizer.fit_transform(concept_aliases)
    print(f"Saving tfidf vectorizer to {tfidf_vectorizer_path}")
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"Fitting and saving vectorizer took {total_time.total_seconds()} seconds")

    print("Finding empty (all zeros) tfidf vectors")
    empty_tfidfs_boolean_flags = numpy.array(
        concept_alias_tfidfs.sum(axis=1) != 0
    ).reshape(-1)
    number_of_non_empty_tfidfs = sum(empty_tfidfs_boolean_flags == False)  # noqa: E712
    total_number_of_tfidfs = numpy.size(concept_alias_tfidfs, 0)

    print(
        f"Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty"
    )
    # remove empty tfidf vectors, otherwise nmslib will crash
    concept_aliases = [
        alias
        for alias, flag in zip(concept_aliases, empty_tfidfs_boolean_flags)
        if flag
    ]
    concept_alias_tfidfs = concept_alias_tfidfs[empty_tfidfs_boolean_flags]
    assert len(concept_aliases) == numpy.size(concept_alias_tfidfs, 0)

    print(
        f"Saving list of concept ids and tfidfs vectors to {uml_concept_aliases_path} and {tfidf_vectors_path}"
    )
    json.dump(concept_aliases, open(uml_concept_aliases_path, "w"))
    scipy.sparse.save_npz(
        tfidf_vectors_path, concept_alias_tfidfs.astype(numpy.float16)
    )

    print(f"Fitting ann index on {len(concept_aliases)} aliases (takes 2 hours)")
    start_time = datetime.datetime.now()
    ann_index = nmslib.init(
        method="hnsw",
        space="cosinesimil_sparse",
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )
    ann_index.addDataPointBatch(concept_alias_tfidfs)
    ann_index.createIndex(index_params, print_progress=True)
    ann_index.saveIndex(ann_index_path)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Fitting ann index took {elapsed_time.total_seconds()} seconds")

    return concept_aliases, tfidf_vectorizer, ann_index
