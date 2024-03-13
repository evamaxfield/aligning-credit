from aligning_credit.data import plos

if __name__ == "__main__":
    df = plos._load_unprocessed_corpus(sample=500)
    results = plos._first_pass_xml_filter_corpus(df)
    print(results.successful_results.sample(10))
    print(len(results.successful_results))

    print()
    print()
    print(results.successful_results["repository_host"].value_counts())
    print()
    print()

    print(results.errored_results.sample(10))
    print(results.errored_results["step"].value_counts())
    print(results.errored_results["error"].value_counts())
    print(results.errored_results["extra_data"].value_counts())
    print(results.errored_results.sample(1).iloc[0].jats_xml_path)

    # df = plos._second_pass_repository_checks(df)
    # print(df.sample(10))
    # print(len(df))
    results.successful_results.to_parquet("plos_sample.parquet")
