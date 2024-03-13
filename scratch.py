from aligning_credit.data import plos

if __name__ == "__main__":
    df = plos._load_unprocessed_corpus(sample=2000)
    results = plos._first_pass_xml_filter_corpus(df)
    print("First pass successful results:")
    print("-" * 80)
    print(
        f"Number of successful repos after first pass: "
        f"{len(results.successful_results)}"
    )
    print(results.successful_results["repository_host"].value_counts())
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
    print("Second pass successful results:")
    print("-" * 80)
    print(
        f"Number of successful repos after second pass: "
        f"{len(results.successful_results)}"
    )
    print("Repo Basic Stats:")
    print(results.successful_results["stars_count"].describe())
    print()
    print(results.successful_results["open_issues_count"].describe())
    print()

    print("Second pass errored results:")
    print("-" * 80)
    print(results.errored_results["step"].value_counts())
    print()
    print(results.errored_results["error"].value_counts())
    print()
    print()
    print("=" * 80)

    # Store the results
    results.successful_results.to_parquet("plos-sample.parquet")
