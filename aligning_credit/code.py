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

from . import ml
from .types import DeveloperDetails, ErrorResult, SuccessAndErroredResults

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _load_github_api_and_check_rate_limit() -> GhApi:
    """
    Load the GitHub API and check the rate limit.

    Returns
    -------
    GhApi
        The GitHub API
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

    return gh_api


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
    gh_api = _load_github_api_and_check_rate_limit()

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
                        identifier=repo_path,
                        step="Repository existance check",
                        error="Repository not found",
                    ).to_dict()
                )
                continue
            else:
                errored_results.append(
                    ErrorResult(
                        identifier=repo_path,
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
                    identifier=repo_path,
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
                    identifier=repo_path,
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


def _get_repository_contributors(
    df: pd.DataFrame,
) -> SuccessAndErroredResults:
    """
    Get the contributors to the repositories in the PLOS corpus.

    Parameters
    ----------
    df : pd.DataFrame
        The PLOS corpus.

    Returns
    -------
    SuccessAndErroredResults
        The successful and errored results
    """
    gh_api = _load_github_api_and_check_rate_limit()

    # Iter over dataframe, grouping by DOI
    # (as there are multiple rows with all the same data except authors)
    # For each group, call to the list_contributors API and get the usernames
    # Then, for each username, call to the get_user API and get the name and email
    # Store the results in a new dataframe
    # Return the new dataframe
    successful_results = []
    errored_results = []

    for _, group in tqdm(
        df.groupby("doi"),
        desc="Getting repository contributors",
        total=df.doi.nunique(),
    ):
        # Sleep to be nice to APIs
        time.sleep(0.5)

        # Get first row to use for data extraction
        # The other rows have the same repository data so we can just use the first
        row = group.iloc[0]

        # Get the contributors
        try:
            contributors = gh_api.repos.list_contributors(
                row.repository_owner,
                row.repository_name,
            )
        except Exception as e:
            errored_results.append(
                ErrorResult(
                    identifier=row.repository_name,
                    step="Repository contributors check",
                    error=f"Error with GitHub API service: {e}",
                ).to_dict()
            )
            continue

        if len(contributors) == 0:
            errored_results.append(
                ErrorResult(
                    identifier=row.repository_name,
                    step="Repository contributors check",
                    error="No contributors found",
                ).to_dict()
            )
            continue

        # Get the contributors' details
        for contributor in tqdm(
            contributors,
            desc="Getting repository contributor details",
            total=len(contributors),
            leave=False,
        ):
            try:
                # Sleep to be nice to APIs
                time.sleep(0.25)

                user = gh_api.users.get_by_username(contributor.login)
                successful_results.append(
                    {
                        "doi": row.doi,
                        "repository_owner": row.repository_owner,
                        "repository_name": row.repository_name,
                        "repository_contributor_username": user.login,
                        "repository_contributor_name": user.name,
                        "repository_contributor_email": user.email,
                    }
                )
            except Exception as e:
                errored_results.append(
                    ErrorResult(
                        identifier=contributor.login,
                        step="Repository contributor details check",
                        error=f"Error with GitHub API service: {e}",
                    ).to_dict()
                )

    return SuccessAndErroredResults(
        successful_results=pd.DataFrame(successful_results),
        errored_results=pd.DataFrame(errored_results),
    )


class AuthorDevClassification:
    dev_author = "dev_author"
    dev_not_author = "dev_not_author"
    author_not_dev = "author_not_dev"


def _match_repository_contributors_to_authors(
    authors_df: pd.DataFrame,
    contributors_df: pd.DataFrame,
) -> pd.DataFrame:
    # Load models once
    clf = ml._load_dev_author_em_model()
    embed_model = ml._load_dev_author_em_embedding_model()

    # Iter over contributors dataframe, grouping by DOI
    # For each group, find the matching group (by DOI) in the authors dataframe
    # Then, use the ML model to predict the matches
    # Store the results in a new dataframe
    results = []
    for _, contributor_group in tqdm(
        contributors_df.groupby("doi"),
        desc="Matching repository contributors to authors",
        total=contributors_df.doi.nunique(),
    ):
        # Get the matching group
        author_group = authors_df[authors_df.doi == contributor_group.doi.iloc[0]]

        # Get the matches
        matches = ml.match_devs_and_authors(
            devs=[
                DeveloperDetails(
                    username=row.repository_contributor_username,
                    name=row.repository_contributor_name,
                    email=row.repository_contributor_email,
                )
                for _, row in contributor_group.iterrows()
            ],
            authors=author_group.full_name.tolist(),
            loaded_dev_author_em_model=clf,
            loaded_embedding_model=embed_model,
        )

        # Create rows for everyone, but

        # for authors-who-were-matched-to-devs,
        # record both their full_name and their dev details

        # for authors-who-were-not-matched-to-devs,
        # record their full_name and None for dev details

        # for devs-who-were-not-matched-to-authors,
        # record None for author details
        for dev_username, author_full_name in matches.items():
            results.append(
                {
                    **author_group.loc[author_group.full_name == author_full_name]
                    .iloc[0]
                    .to_dict(),
                    "repository_contributor_username": dev_username,
                    "repository_contributor_name": contributor_group[
                        contributor_group.repository_contributor_username
                        == dev_username
                    ].repository_contributor_name.iloc[0],
                    "repository_contributor_email": contributor_group[
                        contributor_group.repository_contributor_username
                        == dev_username
                    ].repository_contributor_email.iloc[0],
                    "author_dev_classification": AuthorDevClassification.dev_author,
                }
            )

        for _, row in contributor_group.iterrows():
            if row.repository_contributor_username not in matches:
                results.append(
                    {
                        **author_group.iloc[0].to_dict(),
                        # Remove author details
                        "full_name": None,
                        "email": None,
                        "affliation": None,
                        "roles": None,
                        # Add dev details
                        "repository_contributor_username": (
                            row.repository_contributor_username
                        ),
                        "repository_contributor_name": (
                            row.repository_contributor_name
                        ),
                        "repository_contributor_email": (
                            row.repository_contributor_email
                        ),
                        "author_dev_classification": (
                            AuthorDevClassification.dev_not_author
                        ),
                    }
                )

        for author_full_name in author_group.full_name.tolist():
            if author_full_name not in matches.values():
                results.append(
                    {
                        **author_group.loc[author_group.full_name == author_full_name]
                        .iloc[0]
                        .to_dict(),
                        # Empty dev details
                        "repository_contributor_username": None,
                        "repository_contributor_name": None,
                        "repository_contributor_email": None,
                        "author_dev_classification": (
                            AuthorDevClassification.author_not_dev
                        ),
                    }
                )

    return pd.DataFrame(results)
