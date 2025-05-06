"""Module for measuring constructs in text using construct-text similarity.

Author: Daniel Low
License: Apache 2.0.
"""

import concurrent.futures
import datetime
import gzip
import inspect
import json
import logging
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

import dill
import numpy as np
import pandas as pd
import torch  # Add explicit torch import
import tqdm
from IPython.display import HTML, display
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from . import lexicon
from .utils.logger_config import setup_logger
from .utils.tokenizer import spacy_tokenizer

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# Set up the logger
logger = setup_logger()


# Add this helper function to handle MPS tensor conversion
def ensure_numpy(tensor):
    """Convert PyTorch tensor to numpy array, safely handling MPS/CUDA tensors."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def process_document(
    doc_id: str,
    docs_embeddings_d: Dict[str, np.ndarray],
    construct_embeddings_all: Dict[str, np.ndarray],
    constructs: List[str],
    construct_representation: str,
    summary_stat: List[str],
    skip_nan: bool,
    doc_id_col_name: str,
) -> Tuple[Optional[pd.DataFrame], Dict[str, np.ndarray]]:
    """Process a document and compute cosine similarities between the document and each construct.

    Args:
            doc_id (str): The ID of the document.
            docs_embeddings_d (Dict[str, np.ndarray]): A dictionary mapping document IDs to their embeddings.
            construct_embeddings_all (Dict[str, np.ndarray]): A dictionary mapping construct names to their embeddings.
            constructs (List[str]): A list of construct names.
            construct_representation (str): The representation of the constructs.
            summary_stat (List[str]): A list of summary statistics to compute.
            skip_nan (bool): Whether to skip documents with no embeddings.
            doc_id_col_name (str): The name of the column for the document ID.

    Returns:
            Tuple[Optional[pd.DataFrame], Dict[str, np.ndarray]]:
                    - feature_vectors_doc_df (Optional[pd.DataFrame]): The feature vectors for the document.
                    - cosine_scores_docs_all (Dict[str, np.ndarray]): The cosine scores for each construct (columns are document tokens, rows are lexicon tokens).
    """
    doc_token_embeddings_i = docs_embeddings_d.get(doc_id)  # embeddings for a document

    # Convert tensors to numpy arrays if they are torch tensors
    if doc_token_embeddings_i is not None:
        doc_token_embeddings_i = [ensure_numpy(emb) for emb in doc_token_embeddings_i]
        doc_token_embeddings_i = np.array(doc_token_embeddings_i, dtype=float)

    if skip_nan and doc_token_embeddings_i is None:
        return None, {}

    feature_vectors_doc = [doc_id]
    feature_vectors_doc_col_names = [doc_id_col_name]
    cosine_scores_docs_all = {}

    # compute cosine similarity between each construct and this document
    for construct in constructs:
        construct_embeddings = construct_embeddings_all.get(construct)  # embeddings for a construct

        # cosine similarity
        if construct_representation.startswith("word_"):
            assert len(construct_embeddings.shape) == 1
            if doc_token_embeddings_i.shape[0] == 0:  # happens when there is an empty str
                doc_token_embeddings_i = [np.zeros(construct_embeddings.shape[0])]
            cosine_scores_docs_i = cosine_similarity([construct_embeddings], doc_token_embeddings_i)
        else:
            # cosine similarity between embedding of construct and document
            cosine_scores_docs_i = cosine_similarity(construct_embeddings, doc_token_embeddings_i)

        cosine_scores_docs_all[str(doc_id) + "_" + construct] = cosine_scores_docs_i

        # all summary stats for a single construct will be concatenated side by side
        summary_stats_doc_i = []
        summary_stats_name_doc_i = []

        for stat in summary_stat:
            function = getattr(np, stat)  # e.g. np.max
            doc_sim_stat = function(cosine_scores_docs_i)
            summary_stats_doc_i.append(doc_sim_stat)
            summary_stats_name_doc_i.append(construct + "_" + stat)

        feature_vectors_doc.extend(summary_stats_doc_i)
        feature_vectors_doc_col_names.extend(summary_stats_name_doc_i)

    feature_vectors_doc_df = pd.DataFrame([feature_vectors_doc], columns=feature_vectors_doc_col_names)
    return feature_vectors_doc_df, cosine_scores_docs_all


def measure(
    lexicon_dict: Dict[str, List[str]],
    documents: List[str],
    documents_df: Optional[pd.DataFrame] = None,
    construct_representation: str = "lexicon",
    document_representation: str = "clause",
    summary_stat: List[str] = ["max"],
    count_if_exact_match: Union[bool, str] = "sum",
    lexicon_counts: Optional[pd.DataFrame] = None,
    similarity_threshold: Optional[float] = 0.30,
    minmaxscaler: Optional[Tuple[int, int]] = None,
    return_cosine_similarity: bool = True,
    embeddings_model: str = "all-MiniLM-L6-v2",
    doc_encoding_batch_size: int = 2048,
    load_document_embeddings: Optional[str] = None,
    save_lexicon_embeddings: bool = True,
    stored_embeddings_path: Optional[str] = None,
    save_doc_embeddings: bool = False,
    save_partial_doc_embeddings: bool = True,
    skip_nan: bool = False,
    remove_stat_name_from_col_name: bool = False,
    doc_id_col_name: str = "document_id",
    save_dir: Optional[str] = None,
    save_append_to_filename: Optional[str] = None,
    verbose: bool = True,
    disable_gpu: bool = False,  # option to disable GPU
    use_parallel: bool = True,  # option to disable parallel processing
    skip_batching: bool = False,  # option to skip batch processing altogether
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, np.ndarray]]]:
    """Measure the similarity between constructs and documents.

    Args:
            lexicon_dict (Dict[str, List[str]]): Mapping construct names to lists of tokens.
            documents (List[str]): List of strings.
            documents_df (Optional[pd.DataFrame], optional): Concatenate the output DF to this dataframe (should have the same amount of rows). Defaults to None.
            construct_representation (str, optional): How to represent constructs. Possible values: "lexicon", "word_lexicon", "avg_lexicon", "weighted_avg_lexicon". Defaults to "lexicon".
            document_representation (str, optional): How to represent documents. Possible values: "unigram", "clause", "sentence", "document". Defaults to "clause".
            summary_stat (List[str], optional): List of summary statistics to compute. Possible values: "max", "min", "mean", "sum", "std". Defaults to ["max"].
            count_if_exact_match (Union[bool, str], optional): If a document contains a lexicon token, 'replace' returns count (positive int replaces cosine similarity), False returns cosine similarity to find similar tokens that are not in lexicon, 'sum' returns the sum of counts and cosine similarity. Defaults to 'sum'.
            lexicon_counts (Optional[pd.DataFrame], optional): Only used if count_if_exact_match is True; if you want to compute counts in a specific way use your_lexicon.extract() and input the counts dataframe here. Defaults to None.
            similarity_threshold (Optional[float], optional): Avoid using very low cosine similarities by setting a threshold below which to replace with NaN. Should be tested given it will depend on the embeddings. Recommended possible values: None (do not apply) or float [0.3,0.7]. Defaults to 0.30.
            minmaxscaler (Optional[Tuple[int, int]], optional): Range to scale summary statistics. Possible values: (int, int) or None. Defaults to None.
            return_cosine_similarity (bool, optional): Whether to return cosine similarity. Can occupy a lot of memory if you have many documents and many tokens per document. Defaults to True.
            embeddings_model (str, optional): Name of sentence embeddings model. Possible values: see "Models" here: https://huggingface.co/sentence-transformers and here (click All models upper right corner of table): https://www.sbert.net/docs/sentence_transformer/pretrained_models.html. Defaults to "all-MiniLM-L6-v2".
            doc_encoding_batch_size (int, optional): How many documents to include in a batch for embedding encode. Depends on size of embedding and memory. Defaults to 2048.
            load_document_embeddings (Optional[str], optional): Path to load document embeddings. Possible values: None, file path str. Defaults to None.
            save_lexicon_embeddings (bool, optional): Whether to save lexicon embeddings. Defaults to True.
            stored_embeddings_path (Optional[str], optional): Path to pickle of stored embeddings. Defaults to None.
            save_doc_embeddings (bool, optional): Whether to save document embeddings. WARNING: Can be an extremely heavy file (16GB for 5000 conversations, 1 embedding per phrase), which will take time to save and load. Defaults to False.
            save_partial_doc_embeddings (bool, optional): Whether to save partial document embeddings. Defaults to True.
            skip_nan (bool, optional): Whether to skip documents with no embeddings. Defaults to False.
            remove_stat_name_from_col_name (bool, optional): Whether to remove summary stat name from column name. Defaults to False.
            doc_id_col_name (str, optional): Name of doc_id column. Defaults to "document_id".
            save_dir (Optional[str], optional): Directory to save the extracted features (will save with relevant filenames and name of lexicon). Defaults to None.
            save_append_to_filename (Optional[str], optional): Append this to filename. Defaults to None.
            verbose (bool, optional): Whether to print progress (True) or just warnings (False). Defaults to True.
            disable_gpu (bool, optional): Whether to disable GPU usage and force CPU. Useful for MPS issues. Defaults to False.

    Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, np.ndarray]]]:
                    - If return_cosine_similarity is True:
                            - feature_vectors_all (pd.DataFrame): The feature vectors for the document.
                            - lexicon_dict_final_order (Dict[str, List[str]]): Final order of lexicon constructs.
                            - cosine_scores_docs_all (Dict[str, np.ndarray]): The cosine scores for each construct.
                    - If return_cosine_similarity is False:
                            - feature_vectors_all (pd.DataFrame): The feature vectors for the document.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    # Force CPU if requested
    if disable_gpu:
        logger.info("Disabling GPU/MPS usage as requested. Using CPU only.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.enabled = False

    embeddings_model_clean = embeddings_model.split("/")[-1]

    logger.info("\n =============================================")

    if isinstance(documents_df, pd.DataFrame):
        try:
            assert documents_df.shape[0] == len(documents)
        except AssertionError:
            raise ValueError("documents_df should have the same amount of rows as the length of documents")

    # Make sure summary_stat is a single statistic, and recommend using ['max']
    if count_if_exact_match:
        try:
            assert len(summary_stat) == 1
        except AssertionError as e:
            raise ValueError(
                f"Error {e}: when count_if_exact_match is different than False, summary_stat should be a single statistic like ['max'], because we'll replace cosine similarities with counts >=1."
            )

        try:
            assert summary_stat[0] == "max"
        except AssertionError:
            logging.warning(
                "when count_if_exact_match is True, we recommend summary_stat be ['max'] to max cosine similarity (approximate semantic match up to value of 1) with counts (definite exact match of value 1 or higher). If you replaced mean or std cosine similarities with counts, this is harder to interpret how similarities and counts are related."
            )

    # Embed construct: construct_embeddings_d
    # ================================================================================================
    # Concatenate all tokens so we don't vectorize the same token multiple times
    lexicon_tokens_concat = [item for sublist in lexicon_dict.values() for item in sublist]

    if stored_embeddings_path is not None:
        logger.info("Loading existing lexicon token embeddings...")
        stored_embeddings = dill.load(open(stored_embeddings_path, "rb"))
        # If you need to encode new tokens:
        tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]

    else:
        os.makedirs(f"./data/embeddings/", exist_ok=True)
        stored_embeddings_path = f"./data/embeddings/{embeddings_model_clean}.pickle"
        # Try loading default dir
        try:
            logger.warning(f"stored_embeddings_path is None. Checking if {stored_embeddings_path} exists...")
            stored_embeddings = dill.load(open(stored_embeddings_path, "rb"))
            logger.info(f"Loaded existing lexicon token embeddings from: {stored_embeddings_path}")

        except Exception as e:
            # If you need to encode new tokens:
            stored_embeddings = {}
            tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]
            logger.warning(
                f"Error {e}. Did not find it. Extracting all lexicon token embeddings from scratch instead of loading stored embeddings..."
            )

    # Set device strategy
    device = "cpu" if disable_gpu else None
    sentence_embedding_model = SentenceTransformer(
        embeddings_model, device=device
    )  # load embedding with device preference

    logger.info(f"Default input sequence length for {embeddings_model}: {sentence_embedding_model.max_seq_length}")
    if device:
        logger.info(f"Using device: {device}")
    else:
        logger.info(f"Using default device")

    tokens_to_encode = [n for n in lexicon_tokens_concat if n not in stored_embeddings.keys()]

    # Encode new tokens
    if tokens_to_encode != []:
        logger.info(f"Encoding {len(tokens_to_encode)} new construct tokens...")
        embeddings = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True, show_progress_bar=True)

        # Convert tensors to numpy immediately
        embeddings_d = dict(zip(tokens_to_encode, [ensure_numpy(emb) for emb in embeddings]))
        stored_embeddings.update(embeddings_d)

        # save pickle of embeddings
        if save_lexicon_embeddings:
            logger.info(f"Saving lexicon token embeddings here: {stored_embeddings_path}")
            with open(stored_embeddings_path, "wb") as handle:
                dill.dump(stored_embeddings, handle, protocol=dill.HIGHEST_PROTOCOL)

    # Process stored embeddings to ensure all are numpy arrays
    for key in stored_embeddings:
        stored_embeddings[key] = ensure_numpy(stored_embeddings[key])

    construct_embeddings_d = {}
    lexicon_dict_final_order = {}
    for construct, tokens in lexicon_dict.items():
        construct_embeddings_d[construct] = []
        lexicon_dict_final_order[construct] = []
        for token in tokens:
            construct_embeddings_d[construct].append(ensure_numpy(stored_embeddings.get(token)))
            lexicon_dict_final_order[construct].append(token)

    # Average embeddings for a single construct
    constructs = lexicon_dict.keys()
    if construct_representation == "avg_lexicon":
        for construct in constructs:
            construct_embeddings_list = construct_embeddings_d.get(construct)
            construct_embeddings_avg = np.mean(construct_embeddings_list, axis=0)
            construct_embeddings_avg = np.array(construct_embeddings_avg, dtype=float)
            construct_embeddings_d[construct] = construct_embeddings_avg

    # Embed documents: docs_embeddings_d
    # ================================================================================================

    # Tokenize documents
    if construct_representation == "document":
        docs_tokenized = documents.copy()
    else:
        logger.info("Tokenizing documents...")

        docs_tokenized = spacy_tokenizer(
            documents,
            method=document_representation,
            lowercase=False,
            # display_tree=False,
            remove_punct=False,
            clause_remove_conj=True,
        )

    # Encoding documents
    ts = datetime.datetime.utcnow().strftime("%y-%m-%dT%H-%M-%S")
    docs_embeddings_d = {}
    i_str_all = []
    logger.info("Encoding all document tokens...")

    if save_dir:
        if save_dir[-1] == "/":
            save_dir = (
                save_dir
                + f'cts-scores_count-{count_if_exact_match}_thresh-{str(similarity_threshold).replace(".","")}_{save_append_to_filename}_{lexicon.generate_timestamp(format="%y-%m-%dT%H-%M-%S")}/'
            )
        else:
            save_dir = (
                save_dir
                + f'/cts-scores_count-{count_if_exact_match}_thresh-{str(similarity_threshold).replace(".","")}_{save_append_to_filename}_{lexicon.generate_timestamp(format="%y-%m-%dT%H-%M-%S")}/'
            )
        os.makedirs(save_dir, exist_ok=True)

    # OPTION 1: Non-batched processing approach to avoid MPS issues
    def encode_without_batching(docs_tokenized):
        """Encode documents without batching to avoid MPS memory issues."""
        docs_embeddings_d = {}
        for i, list_of_clauses in tqdm.tqdm(enumerate(docs_tokenized), desc="Encoding documents"):
            # Process each document separately to avoid batch memory issues
            embeddings = []
            for clause in list_of_clauses:
                # Encode one clause at a time
                emb = sentence_embedding_model.encode(clause, convert_to_tensor=False)  # directly get numpy
                embeddings.append(emb)
            docs_embeddings_d[i] = embeddings
        return docs_embeddings_d

    if skip_batching or doc_encoding_batch_size <= 1:
        logger.info("Using non-batched encoding approach to avoid MPS issues")
        docs_embeddings_d = encode_without_batching(docs_tokenized)
    else:
        # OPTION 2: Standard batched approach with tensor conversion
        # Flatten the list of lists into a single list while keeping track of the keys
        flattened_docs = []
        keys = []
        for i, list_of_clauses in enumerate(docs_tokenized):
            flattened_docs.extend(list_of_clauses)
            keys.extend([i] * len(list_of_clauses))

        # Encode in batches
        encoded_batches = []
        for i in tqdm.tqdm(range(0, len(flattened_docs), doc_encoding_batch_size), desc="Encoding batches"):
            batch = flattened_docs[i : i + doc_encoding_batch_size]
            encoded_batch = sentence_embedding_model.encode(batch, convert_to_tensor=True)

            # Convert tensors to numpy immediately after encoding
            if isinstance(encoded_batch, torch.Tensor):
                encoded_batch = [ensure_numpy(encoded_batch[j]) for j in range(len(encoded_batch))]
            else:
                encoded_batch = [ensure_numpy(emb) for emb in encoded_batch]

            encoded_batches.extend(encoded_batch)

        # Store embeddings in the dictionary
        docs_embeddings_d = {}
        current_index = 0
        for i, list_of_clauses in enumerate(docs_tokenized):
            docs_embeddings_d[i] = encoded_batches[current_index : current_index + len(list_of_clauses)]
            current_index += len(list_of_clauses)

            if save_doc_embeddings and save_partial_doc_embeddings and i % 500 == 0:
                # save partial ones in case it fails during the process
                i_str = str(i).zfill(5)
                i_str_all.append(i_str)

                with open(
                    save_dir
                    + f"embeddings_{embeddings_model_clean}_docs_{document_representation}_with-interaction_{ts}_part-{i_str}.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(docs_embeddings_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save final one
    if save_doc_embeddings:
        logger.info(
            "Saving document embeddings here: "
            + save_dir
            + f"embeddings_{embeddings_model_clean}_docs_{document_representation}_with-interaction_{ts}.pickle"
        )
        with open(
            save_dir
            + f"embeddings_{embeddings_model_clean}_docs_{document_representation}_with-interaction_{ts}.pickle",
            "wb",
        ) as handle:
            pickle.dump(docs_embeddings_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # remove partial ones
    if save_doc_embeddings and save_partial_doc_embeddings:
        for i_str in i_str_all:
            os.remove(
                save_dir
                + f"embeddings_{embeddings_model_clean}_docs_{document_representation}_with-interaction_{ts}_part-{i_str}.pickle"
            )

    # Compute cosine similarity
    # ================================================================================================
    feature_vectors_all = []
    cosine_scores_docs_all = {}

    logger.info(
        f"computing similarity between {len(constructs)} constructs and {len(docs_embeddings_d.keys())} documents..."
    )

    # Use the user-specified option for parallel processing
    if use_parallel:
        # parallelized processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_document,
                    doc_id,
                    docs_embeddings_d,
                    construct_embeddings_d,
                    constructs,
                    construct_representation,
                    summary_stat,
                    skip_nan,
                    doc_id_col_name,
                )
                for doc_id in docs_embeddings_d.keys()
            ]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), desc="Processing documents"):
                doc_result, doc_cosine_scores = future.result()
                if doc_result is not None:
                    feature_vectors_all.append(doc_result)
                    cosine_scores_docs_all.update(doc_cosine_scores)
    else:
        # sequential processing for debugging
        for doc_id in tqdm.tqdm(docs_embeddings_d.keys(), desc="Processing documents"):
            doc_result, doc_cosine_scores = process_document(
                doc_id,
                docs_embeddings_d,
                construct_embeddings_d,
                constructs,
                construct_representation,
                summary_stat,
                skip_nan,
                doc_id_col_name,
            )
            if doc_result is not None:
                feature_vectors_all.append(doc_result)
                cosine_scores_docs_all.update(doc_cosine_scores)

    feature_vectors_all = pd.concat(feature_vectors_all).reset_index(drop=True)

    # Scale between 0 and 1 to follow output range of other classification models.
    if minmaxscaler is not None:
        scaler = MinMaxScaler()
        feature_cols = [col for col in feature_vectors_all.columns if any(string in col for string in summary_stat)]
        feature_vectors_all[feature_cols] = scaler.fit_transform(feature_vectors_all[feature_cols].values)

    if remove_stat_name_from_col_name:
        for stat in summary_stat:
            feature_vectors_all.columns = [n.replace(f"_{stat}", "") for n in feature_vectors_all.columns]

    if similarity_threshold is not None:
        # Set all values below threshold to 0
        # feature_vectors_all = feature_vectors_all[counts.columns]
        feature_vectors_all[feature_vectors_all < similarity_threshold] = 0

    # Count lexicon in documents
    # =================================================================================================
    if count_if_exact_match:
        if not isinstance(lexicon_counts, pd.DataFrame):
            # if you did not provide the counts DF, lexicon_counts will be None, and it will be generated here
            logger.info("Adding lexicon tokens to a new lexicon to count exact matches...")
            new_lexicon = lexicon.Lexicon()
            for c in lexicon_dict.keys():
                new_lexicon.add(c, section="tokens", value=lexicon_dict[c], verbose=False)
            logger.info("Lemmatizing tokens...")
            new_lexicon = lexicon.lemmatize_tokens(new_lexicon)

            logger.info("Counting exact matches lexicon.extract()...")
            lexicon_counts, matches_by_construct, matches_doc2construct, matches_construct2doc = new_lexicon.extract(
                documents,
                normalize=False,
            )

        lexicon_counts = lexicon_counts[list(lexicon_dict.keys())]
        if not remove_stat_name_from_col_name:
            lexicon_counts.columns = [n + f"_{summary_stat[0]}" for n in lexicon_counts.columns]

        # display(counts)
        # display(feature_vectors_all)
        if count_if_exact_match == "sum":
            feature_vectors_all[lexicon_counts.columns] = feature_vectors_all[lexicon_counts.columns] + lexicon_counts

        elif count_if_exact_match == "replace":
            feature_vectors_all = lexicon_counts.where(lexicon_counts >= 1, feature_vectors_all)

    # add documents
    construct_columns = list(feature_vectors_all.columns[1:])
    feature_vectors_all["document"] = documents
    feature_vectors_all["documents_tokenized"] = docs_tokenized
    # reorder
    feature_vectors_all = feature_vectors_all[["document", "documents_tokenized"] + construct_columns]

    feature_vectors_all.reset_index(drop=True, inplace=True)

    if isinstance(documents_df, pd.DataFrame):
        documents_df.reset_index(drop=True, inplace=True)
        feature_vectors_all = pd.concat([documents_df, feature_vectors_all], axis=1)

    if save_append_to_filename:
        save_append_to_filename = "_" + save_append_to_filename
    else:
        save_append_to_filename = ""

    if save_dir:
        # write a txt with arguments
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        arguments = {arg: values[arg] for arg in args}

        exclude = [
            "documents",
            "documents_df",
            "cosine_scores_docs_all",
            "lexicon_counts",
            "list_of_clauses",
            "keys",
            "encoded_batches",
        ]
        with open(save_dir + f"arguments_log{save_append_to_filename}.txt", "w") as file:
            for name, value in arguments.items():
                if name not in exclude:
                    file.write(f"{name}: {value}\n")

        feature_vectors_all.to_csv(save_dir + "cts_scores.csv", index=False)
        # save lexicon_dict_final_order as json with indent
        with open(save_dir + f"lexicon_dict_final_order{save_append_to_filename}.json", "w", encoding="utf-8") as f:
            json.dump(lexicon_dict_final_order, f, ensure_ascii=False, indent=4)

        # save cosine_scores_docs_all as compressed pickle
        with gzip.open(save_dir + f"cosine_similarities{save_append_to_filename}.pkl.gz", "wb") as f:
            pickle.dump(cosine_scores_docs_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    if return_cosine_similarity:
        # Fix: Return only two values as expected in the calling code
        return feature_vectors_all, cosine_scores_docs_all
    else:
        return feature_vectors_all


def get_highest_similarity_phrase(
    doc_id: str,
    construct: str,
    documents: List[str],
    documents_tokenized: Dict[str, List[str]],
    cosine_similarities: Dict[str, np.ndarray],
    lexicon_dict_final_order: Dict[str, List[str]],
) -> Tuple[str, str, float]:
    """Find and display the highest similarity between a document token and a construct token.

    Args:
            doc_id (str): The ID of the document.
            construct (str): The name of the construct.
            documents (List[str]): List of documents.
            documents_tokenized (Dict[str, List[str]]): A dictionary mapping document IDs to lists of tokenized document tokens.
            cosine_similarities (Dict[str, np.ndarray]): A dictionary mapping document-construct pairs to their cosine similarity matrices.
            lexicon_dict_final_order (Dict[str, List[str]]): A dictionary mapping construct names to lists of lexicon tokens.

    Returns:
            Tuple[str, str, float]:
                    - most_similar_lexicon_token (str): The lexicon token with the highest cosine similarity.
                    - most_similar_document_token (str): The document token with the highest cosine similarity.
                    - highest_similarity (float): The highest cosine similarity value.
    """
    # You can verify similarity here: https://github.com/danielmlow/tutorials/blob/main/text/semantic_similarity.ipynb
    doc_id_construct = f"{doc_id}_{construct}"
    similarities_for_1_doc = cosine_similarities[doc_id_construct]
    most_similar_lexicon_token_per_doc_token_index = np.argmax(
        similarities_for_1_doc, axis=0
    )  # top similarities for each token in the document
    highest_similarity = np.max(similarities_for_1_doc)
    highest_similarity = np.round(highest_similarity, 2)
    most_similar_document_token_index = np.argmax(np.max(similarities_for_1_doc, axis=0))
    most_similar_lexicon_token_index = most_similar_lexicon_token_per_doc_token_index[most_similar_document_token_index]
    most_similar_document_token = documents_tokenized[doc_id][most_similar_document_token_index]
    most_similar_lexicon_token = lexicon_dict_final_order.get(construct)[most_similar_lexicon_token_index]

    # Display
    print(
        f"The construct '{construct}' through its token '{most_similar_lexicon_token}' had the highest cosine similarity ({highest_similarity}) with the following document token:\n'{most_similar_document_token}'"
    )
    document = documents[doc_id]
    document = document.replace(most_similar_document_token, f"<mark>{most_similar_document_token}</mark>")
    display(HTML(document))

    return most_similar_lexicon_token, most_similar_document_token, highest_similarity
