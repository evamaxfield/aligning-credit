from aligning_credit.data import plos

if __name__ == "__main__":
    df = plos._load_unprocessed_corpus(sample=None)
    results = plos._first_pass_xml_filter_corpus(df)
    print("First pass successful results:")
    print("-" * 80)
    print(
        f"Number of successful papers after first pass: "
        f"{results.successful_results.doi.nunique()}"
    )
    print()
    print()

    print("First pass errored results:")
    print("-" * 80)
    print(results.errored_results["step"].value_counts())
    print()
    print(results.errored_results["error"].value_counts())
    print()
    print()
    print("=" * 80)
    print()
    print()

    results = plos._second_pass_repository_checks(results.successful_results)

    # Store the results
    results.successful_results.to_parquet("processed-plos-corpus.parquet")

    print("Second pass successful results:")
    print("-" * 80)
    print(
        f"Number of successful papers after second pass: "
        f"{results.successful_results.doi.nunique()}"
    )
    print("Repo Basic Stats:")
    print(
        results.successful_results.groupby("doi")
        .first()
        .repository_stargazers_count.describe()
    )
    print()
    print(
        results.successful_results.groupby("doi")
        .first()
        .repository_open_issues_count.describe()
    )
    print()
    print(
        results.successful_results.groupby("doi")
        .first()
        .repository_license.value_counts()
    )
    print()
    print()

    print("Second pass errored results:")
    print("-" * 80)
    print(results.errored_results["step"].value_counts())
    print()
    print(results.errored_results["error"].value_counts())
    print()
    print()
    print("=" * 80)
