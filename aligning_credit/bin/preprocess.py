#!/usr/bin/env python

import logging

import numpy as np
import pandas as pd
import typer

from aligning_credit import code, plos
from aligning_credit.bin.typer_utils import setup_logger
from aligning_credit.data import DATA_FILES_DIR

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# Filtering PLOS XML
FILTERED_PLOS_SUCCESS_FILE = DATA_FILES_DIR / "filtered-corpus.parquet"
FILTERED_PLOS_ERRORED_FILE = DATA_FILES_DIR / "filtered-corpus-errored.parquet"

# Extracting Acknowledged Members
ACKNOWLEDGED_MEMBERS_SUCCESS_FILE = DATA_FILES_DIR / "acknowledged-members.parquet"
ACKNOWLEDGED_MEMBERS_ERRORED_FILE = (
    DATA_FILES_DIR / "acknowledged-members-errored.parquet"
)

# Repository checking
REPO_CHECK_SUCCESS_FILE = DATA_FILES_DIR / "repo-check-corpus.parquet"
REPO_CHECK_ERRORED_FILE = DATA_FILES_DIR / "repo-check-errored.parquet"

# Repository contributors
REPO_CONTRIB_SUCCESS_FILE = DATA_FILES_DIR / "repo-contributors-successful.parquet"
REPO_CONTRIB_ERRORED_FILE = DATA_FILES_DIR / "repo-contributors-errored.parquet"

# Matched contributors and paper members
MATCHED_REPO_CONTRIB_TO_PAPER_MEMBERS_FILE = (
    DATA_FILES_DIR / "matched-repo-contributors-to-paper-members.parquet"
)


###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def filter_plos_xml(
    sample: int = -1,
    random_state: int = 12,
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Load the unprocessed corpus
    df = plos._load_unprocessed_corpus(
        sample=None if sample == -1 else sample,
        random_state=random_state,
    )

    # Filter the XML files
    results = plos._process_plos_xml_files(df)

    # Store these first pass results
    log.info(f"Storing successful results to '{FILTERED_PLOS_SUCCESS_FILE}'")
    results.successful_results.to_parquet(FILTERED_PLOS_SUCCESS_FILE)
    log.info(f"Storing errored results to '{FILTERED_PLOS_ERRORED_FILE}'")
    results.errored_results.to_parquet(FILTERED_PLOS_ERRORED_FILE)

    # Success Stats
    print("Results from XML Filtering:")
    print(f"Number of successful papers: {results.successful_results.doi.nunique()}")
    print()
    print(
        f"Total number of author-paper contributions: {len(results.successful_results)}"
    )
    print()
    print("Repo Hosts:")
    print(
        results.successful_results.groupby("doi").first().repository_host.value_counts()
    )
    print()
    print("-" * 80)
    print()

    # Error Stats
    if len(results.errored_results) > 0:
        print("Errored Results:")
        print()
        print("Error Steps:")
        print(results.errored_results["step"].value_counts())
        print()
        print("Error Values:")
        print(results.errored_results["error"].value_counts())


@app.command()
def extract_acknowledged_members(
    sample: int = -1,
    random_state: int = 12,
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Load prior results
    filtered_plos_corpus = pd.read_parquet(FILTERED_PLOS_SUCCESS_FILE)

    # Sample if desired
    if sample != -1:
        np.random.seed(random_state)
        dois = np.random.choice(
            filtered_plos_corpus.doi.unique(), sample, replace=False
        )
        filtered_plos_corpus = filtered_plos_corpus[filtered_plos_corpus.doi.isin(dois)]

    # Extract the acknowledged members
    results = plos._extract_acknowledged_members(filtered_plos_corpus)

    # Store the results
    log.info(f"Storing successful results to '{ACKNOWLEDGED_MEMBERS_SUCCESS_FILE}'")
    results.successful_results.to_parquet(ACKNOWLEDGED_MEMBERS_SUCCESS_FILE)
    log.info(f"Storing errored results to '{ACKNOWLEDGED_MEMBERS_ERRORED_FILE}'")
    results.errored_results.to_parquet(ACKNOWLEDGED_MEMBERS_ERRORED_FILE)

    # Success Stats
    print("Results from Acknowledged Members Extraction:")
    print(f"Number of successful papers: {results.successful_results.doi.nunique()}")
    print()
    print("Distribution of Member Positions:")
    print(results.successful_results["position"].value_counts())
    print()
    print("-" * 80)
    print()

    # Error Stats
    if len(results.errored_results) > 0:
        print("Errored Results:")
        print()
        print("Error Steps:")
        print(results.errored_results["step"].value_counts())
        print()
        print("Error Values:")
        print(results.errored_results["error"].value_counts())
    else:
        print("No errored results!")


@app.command()
def filter_repositories(
    sample: int = -1,
    random_state: int = 12,
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Load prior results
    filtered_plos_corpus = pd.read_parquet(ACKNOWLEDGED_MEMBERS_SUCCESS_FILE)

    # Sample if desired
    if sample != -1:
        np.random.seed(random_state)
        dois = np.random.choice(
            filtered_plos_corpus.doi.unique(), sample, replace=False
        )
        filtered_plos_corpus = filtered_plos_corpus[filtered_plos_corpus.doi.isin(dois)]

    # Filter the repositories
    results = code._check_and_filter_repositories(filtered_plos_corpus)

    # Store the results
    log.info(f"Storing successful results to '{REPO_CHECK_SUCCESS_FILE}'")
    results.successful_results.to_parquet(REPO_CHECK_SUCCESS_FILE)
    log.info(f"Storing errored results to '{REPO_CHECK_ERRORED_FILE}'")
    results.errored_results.to_parquet(REPO_CHECK_ERRORED_FILE)

    # Success Stats
    print("Results from Repository Filtering:")
    print(f"Number of successful papers: {results.successful_results.doi.nunique()}")
    print()
    print(
        f"Total number of member-paper contributions: {len(results.successful_results)}"
    )
    print()
    print("Distribution of Stargazers:")
    print(
        results.successful_results.groupby("doi")
        .first()
        .repository_stargazers_count.describe()
    )
    print()
    print("Distribution of Open Issues:")
    print(
        results.successful_results.groupby("doi")
        .first()
        .repository_open_issues_count.describe()
    )
    print()
    print("Repositories Licenses:")
    print(
        results.successful_results.groupby("doi")
        .first()
        .repository_license.value_counts()
    )
    print()
    print("-" * 80)
    print()

    # Error Stats
    if len(results.errored_results) > 0:
        print("Errored Results:")
        print()
        print("Error Steps:")
        print(results.errored_results["step"].value_counts())
        print()
        print("Error Values:")
        print(results.errored_results["error"].value_counts())


@app.command()
def get_repository_contributors(
    sample: int = -1,
    random_state: int = 12,
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Load prior results
    filtered_plos_corpus = pd.read_parquet(REPO_CHECK_SUCCESS_FILE)

    # Sample if desired
    if sample != -1:
        np.random.seed(random_state)
        dois = np.random.choice(
            filtered_plos_corpus.doi.unique(), sample, replace=False
        )
        filtered_plos_corpus = filtered_plos_corpus[filtered_plos_corpus.doi.isin(dois)]

    # Get the repository contributors
    results = code._get_repository_contributors(filtered_plos_corpus)

    # Store the results
    log.info(f"Storing successful results to '{REPO_CONTRIB_SUCCESS_FILE}'")
    results.successful_results.to_parquet(REPO_CONTRIB_SUCCESS_FILE)
    log.info(f"Storing errored results to '{REPO_CONTRIB_ERRORED_FILE}'")
    results.errored_results.to_parquet(REPO_CONTRIB_ERRORED_FILE)

    # Success Stats
    print("Results from Repository Contributors:")
    print(f"Number of successful papers: {results.successful_results.doi.nunique()}")
    print()
    print(f"Total number of contributors: {len(results.successful_results)}")
    print()
    print("Distribution of Contributors:")
    print(results.successful_results.groupby("doi").apply(len).describe())
    print()
    print("-" * 80)
    print()

    # Error Stats
    if len(results.errored_results) > 0:
        print("Errored Results:")
        print()
        print("Error Steps:")
        print(results.errored_results["step"].value_counts())
        print()
        print("Error Values:")
        print(results.errored_results["error"].value_counts())
    else:
        print("No errored results!")


@app.command()
def match_repository_contributors_to_authors(
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Load prior results
    repo_checked_plos_corpus = pd.read_parquet(REPO_CHECK_SUCCESS_FILE)
    repo_contributors = pd.read_parquet(REPO_CONTRIB_SUCCESS_FILE)

    # Match the repository contributors to the authors
    df = code._match_repository_contributors_to_authors(
        authors_df=repo_checked_plos_corpus,
        contributors_df=repo_contributors,
    )

    # Store the results
    log.info(
        f"Storing matched results to '{MATCHED_REPO_CONTRIB_TO_PAPER_MEMBERS_FILE}'"
    )
    df.to_parquet(MATCHED_REPO_CONTRIB_TO_PAPER_MEMBERS_FILE)

    # Stats
    print("Results from Matching Repository Contributors to Authors:")
    print(f"Number of successful papers: {df.doi.nunique()}")
    print()
    print(f"Total number of project contributors (authors or devs): {len(df)}")
    print()
    print("Distribution of Known-Member-Dev Classifications:")
    print(df.known_member_dev_classification.value_counts())
    print()
    print("-" * 80)
    print()


@app.command()
def all_steps(
    sample: int = -1,
    random_state: int = 12,
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Filter PLOS XML
    log.info("Beginning PLOS XML Filtering")
    filter_plos_xml(sample=sample, random_state=random_state, debug=debug)
    print()

    # Extract Acknowledged Members
    log.info("Beginning Acknowledged Members Extraction")
    extract_acknowledged_members(debug=debug)
    print()

    # Filter Repositories
    log.info("Beginning Repository Filtering")
    filter_repositories(debug=debug)
    print()

    # Get Repository Contributors
    log.info("Beginning Repository Contributors")
    get_repository_contributors(debug=debug)
    print()

    # Match Repository Contributors to Authors
    log.info("Beginning Matching Repository Contributors to Authors")
    match_repository_contributors_to_authors(debug=debug)
    print()


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
