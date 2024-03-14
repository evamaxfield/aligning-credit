#!/usr/bin/env python

import logging

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from skops import io as skio

from .data import DATA_FILES_DIR
from .types import DeveloperDetails

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEV_AUTHOR_EM_CLASSIFIER_PATH = DATA_FILES_DIR / "dev-author-em-model.skops"
DEV_AUTHOR_EM_EMBEDDING_MODEL = "microsoft/deberta-v3-base"
DEV_AUTHOR_EM_TEMPLATE = """
### Developer Details

username: {dev_username}
name: {dev_name}
email: {dev_email}

---

### Author Details

name: {author_name}
""".strip()

###############################################################################


def _load_dev_author_em_model() -> LogisticRegressionCV:
    """
    Load the author EM model.

    Returns
    -------
    LogisticRegressionCV
        The author EM model
    """
    log.debug(f"Loading author EM model from {DEV_AUTHOR_EM_CLASSIFIER_PATH}")
    return skio.load(DEV_AUTHOR_EM_CLASSIFIER_PATH, trust=True)

def _load_dev_author_em_embedding_model() -> SentenceTransformer:
    """
    Load the author EM embedding model.

    Returns
    -------
    SentenceTransformer
        The author EM embedding model
    """
    log.debug(f"Loading author EM embedding model from {DEV_AUTHOR_EM_EMBEDDING_MODEL}")
    return SentenceTransformer(DEV_AUTHOR_EM_EMBEDDING_MODEL)


def match_devs_and_authors(
    devs: list[DeveloperDetails],
    authors: list[str],
    loaded_dev_author_em_model: LogisticRegressionCV | None = None,
    loaded_embedding_model: SentenceTransformer | None = None,
) -> dict[str, str]:
    """
    Embed developers and authors.

    Parameters
    ----------
    devs : list[DeveloperDetails]
        The developers to embed.
    authors : list[str]
        The authors to embed.
    loaded_dev_author_em_model : LogisticRegressionCV, optional
        The loaded author EM model, by default None
    loaded_embedding_model : SentenceTransformer, optional
        The loaded embedding model, by default None
    
    Returns
    -------
    dict[str, str]
        The predicted matches
    """
    # If no loaded classifer, load the model
    if loaded_dev_author_em_model is None:
        clf = _load_dev_author_em_model()
    else:
        clf = loaded_dev_author_em_model
    
    # If no loaded embedding model, load the model
    if loaded_embedding_model is None:
        embed_model = _load_dev_author_em_embedding_model()
    else:
        embed_model = loaded_embedding_model

    # Create pairs of developers and authors and fill in the template
    pairs = {}
    pair_strs = []
    for dev_details in devs:
        pairs[dev_details.username] = {}
        for author in authors:
            filled_template = DEV_AUTHOR_EM_TEMPLATE.format(
                dev_username=dev_details.username,
                dev_name=dev_details.name,
                dev_email=dev_details.email,
                author_name=author,
            )
            pairs[dev_details.username][author] = filled_template
            pair_strs.append(filled_template)

    # Create embeddings for all of the pairs
    log.debug("Creating embeddings for all dev-author pairs")
    pair_embeddings = embed_model.encode(pair_strs)
