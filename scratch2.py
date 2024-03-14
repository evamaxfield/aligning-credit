import pandas as pd

from aligning_credit import code

if __name__ == "__main__":
    df = pd.read_parquet("processed-plos-corpus.parquet")
    repo_contrib_results = code._get_repository_contributors(df.sample(10))

    repo_contrib_results.successful_results.to_parquet(
        "repo-contributors-successful.parquet"
    )
    repo_contrib_results.errored_results.to_parquet("repo-contributors-errored.parquet")

    print(repo_contrib_results.successful_results.sample(3))

    print()
    print()
    print("-" * 80)
    print()

    print(len(repo_contrib_results.errored_results))
    if len(repo_contrib_results.errored_results) > 0:
        print(repo_contrib_results.errored_results.step.value_counts())
        print(repo_contrib_results.errored_results.error.value_counts())

    df = code._match_repository_contributors_to_authors(
        df, repo_contrib_results.successful_results
    )

    print(df)

    df.to_parquet("matched-repo-contributors-to-authors.parquet")

    print(df.sample(3))
    print(df.shape)
    print(df.columns)

    print(df.author_dev_classification.value_counts())
