#!/usr/bin/env python

import logging
import os
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from ghapi.all import GhApi
from tqdm import tqdm

from .types import ErrorResult, SuccessAndErroredResults

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _filter_repositories_in_plos_dataset(df: pd.DataFrame) -> SuccessAndErroredResults:
    """
    Process the PLOS corpus to check for repository existance and if the repository has
    common coding language files.

    Parameters
    ----------
    df : pd.DataFrame
        The PLOS corpus.

    Returns
    -------
    SuccessAndErroredResults
        The processed PLOS corpus (in success and errored dataframes)
    """
    # Load env
    load_dotenv()
    gh_api = GhApi(token=os.environ["GITHUB_TOKEN"])

    # Get rate limit status of GitHub API
    rate_limit = gh_api.rate_limit.get()
    current_core_limit = rate_limit["resources"]["core"]["remaining"]
    if current_core_limit <= 100:
        raise ValueError(
            f"GitHub API rate limit is too low: {current_core_limit}. "
            f"Ensure that you have an up-to-date API key."
        )

    # Filter out non-GitHub repositories from the dataframe
    # Log how many were filtered out
    n_non_github = len(df[df.repository_host != "github"])
    log.info(f"Filtering out {n_non_github} non-GitHub repositories")
    df = df[df.repository_host == "github"].copy()

    # Filtering out non-Repository repositories from the dataframe
    # Log how many were filtered out
    n_non_repo = len(df[df.repository_name.isna()])
    log.info(f"Filtering out {n_non_repo} organization URLs")
    df = df[~df.repository_name.isna()].copy()

    # Store final results
    successful_results = []
    errored_results = []

    # Iter through corpus, grouping by DOI
    # (as there are multiple rows with all the same data except authors)
    for _, group in tqdm(
        df.groupby("doi"),
        desc="Checking repository details",
        total=df.doi.nunique(),
    ):
        # Sleep to be nice to APIs
        time.sleep(0.5)

        # Get first row to use for data extraction
        # The other rows have the same repository data so we can just use the first
        row = group.iloc[0]

        # Construct the repo path
        repo_path = f"{row.repository_owner}/{row.repository_name}"

        # Check if the repository exists
        try:
            # Call to ecosyste.ms API
            resp = requests.get(
                f"https://repos.ecosyste.ms/api/v1/hosts/github/repositories/"
                f"{repo_path}"
            )
            resp.raise_for_status()

            # Keep some of the data
            data = resp.json()
            group["repository_stargazers_count"] = data["stargazers_count"]
            group["repository_open_issues_count"] = data["open_issues_count"]
            group["repository_forks_count"] = data["forks_count"]
            group["repository_most_recent_push_datetime"] = data["pushed_at"]
            group["repository_license"] = data["license"]
            group["repository_data_cache_datetime"] = datetime.now().isoformat()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                errored_results.append(
                    ErrorResult(
                        jats_xml_path=row.jats_xml_path,
                        step="Repository existance check",
                        error="Repository not found",
                    ).to_dict()
                )
                continue
            else:
                errored_results.append(
                    ErrorResult(
                        jats_xml_path=row.jats_xml_path,
                        step="Repository existance check",
                        error=f"Error with Ecosyste.ms API service: {e}",
                    ).to_dict()
                )
                continue

        # Check repository languages
        try:
            repo_languages = gh_api.repos.list_languages(
                row.repository_owner,
                row.repository_name,
            )
        except Exception as e:
            errored_results.append(
                ErrorResult(
                    jats_xml_path=row.jats_xml_path,
                    step="Repository language check",
                    error=f"Error with GitHub API service: {e}",
                ).to_dict()
            )
            continue

        # Check languages
        if all(
            lang not in repo_languages
            for lang in [
                "Python",
                "R",
                "RMarkdown",
                "Jupyter Notebook",
                "Ruby",
                "C",
                "C++",
                "Java",
                "Go",
                "JavaScript",
                "TypeScript",
                "Rust",
                "Julia",
            ]
        ):
            errored_results.append(
                ErrorResult(
                    jats_xml_path=row.jats_xml_path,
                    step="Repository language check",
                    error="No common coding languages found",
                ).to_dict()
            )
            continue

        # Add languages to new repository data
        languages_string = ""
        for lang in repo_languages:
            languages_string += f"{lang}:{repo_languages[lang]};"
        group["repository_languages"] = languages_string

        # Add to successful results
        successful_results.append(group)

    # Return the results
    return SuccessAndErroredResults(
        successful_results=pd.concat(successful_results),
        errored_results=pd.DataFrame(errored_results),
    )

def _match_repository_contributors_to_authors(
    df: pd.DataFrame,
) -> SuccessAndErroredResults:
    # Load env
    load_dotenv()
    gh_api = GhApi(token=os.environ["GITHUB_TOKEN"])

    # Get rate limit status of GitHub API
    rate_limit = gh_api.rate_limit.get()
    current_core_limit = rate_limit["resources"]["core"]["remaining"]
    if current_core_limit <= 100:
        raise ValueError(
            f"GitHub API rate limit is too low: {current_core_limit}. "
            f"Ensure that you have an up-to-date API key."
        )
    
    # Store final results
    successful_results = []
    errored_results = []

    # Iter through corpus, grouping by DOI
    # (as there are multiple rows with all the same data except authors)
    for _, group in tqdm(
        df.groupby("doi"),
        desc="Matching repository contributors to authors",
        total=df.doi.nunique(),
    ):
        # Sleep to be nice to APIs
        time.sleep(0.5)

        