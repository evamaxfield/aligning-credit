#!/usr/bin/env python

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: N817

import pandas as pd
from allofplos.corpus.plos_corpus import get_corpus_dir
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def load_unprocessed_corpus(
    sample: int | None = 10000, random_state: int = 12
) -> pd.DataFrame:
    """
    Load the PLOS corpus.

    Returns
    -------
    pd.DataFrame
        The PLOS corpus.
    """
    # Get the corpus dir
    corpus_dir = get_corpus_dir()

    # Get all files
    files = Path(corpus_dir).resolve(strict=True).glob("*.xml")

    # Return sample if requested
    if sample:
        return (
            pd.DataFrame(
                files,
                columns=["jats_xml_path"],
            )
            .sample(
                sample,
                random_state=random_state,
            )
            .astype({"jats_xml_path": str})
        )

    # Return the full set
    return pd.DataFrame(files, columns=["jats_xml_path"]).astype({"jats_xml_path": str})


@dataclass
class Author(DataClassJsonMixin):
    full_name: str
    email: str
    affliation: str
    roles: str | None


@dataclass
class ArticleProcessingResult(DataClassJsonMixin):
    journal_name: str
    journal_pmc_id: str
    doi: str
    full_title: str
    short_title: str
    abstract: str
    disciplines: str
    repository_url: str
    authors: list[Author]
    acknowledgement_statement: str
    funding_statement: str
    funding_sources: str
    publish_date: str


def _get_repository_url(root: ET.Element, jats_xml_filepath: str) -> str | None:
    # Before anything else, check for data availability
    # Example:
    # <custom-meta id="data-availability">
    #     <meta-name>Data Availability</meta-name>
    #     <meta-value>A TimeTeller R package is available from GitHub at <ext-link ext-link-type="uri" xlink:href="https://github.com/VadimVasilyev1994/TimeTeller" xlink:type="simple">https://github.com/VadimVasilyev1994/TimeTeller</ext-link>. All data used except that of Bjarnasson et al. is publicly available from the sites given in the Supplementary Information. The Bjarnasson et al. data that is used is available with the R package.</meta-value>  # noqa: E501
    # </custom-meta>
    data_availability = root.find(".//custom-meta[@id='data-availability']")
    if data_availability is None:
        log.debug(f"No data availability found for: '{jats_xml_filepath}'")
        return None

    # Check if URL within data availability points to github.com / github.io
    # If so, store the URL from the ext-link text
    ext_link = data_availability.find(".//ext-link[@ext-link-type='uri']")
    if ext_link is not None:
        repository_url = ext_link.text

        # Check if the URL points to github.com or github.io
        if isinstance(repository_url, str) and all(
            domain not in repository_url for domain in ["github.com", "github.io"]
        ):
            log.debug(
                f"External link found within data-availability-block "
                f"for: '{jats_xml_filepath}' but it does not point to GitHub"
            )
            return None

        return repository_url

    log.debug(
        f"No external link found within "
        f"data-availability-block for: '{jats_xml_filepath}'"
    )
    return None


@dataclass
class JournalInfo:
    journal_name: str
    journal_pmc_id: str


def _get_journal_info(root: ET.Element, jats_xml_filepath: str) -> JournalInfo | None:
    # Get the Journal Name "nlm-ta"
    # Example:
    # <journal-id journal-id-type="nlm-ta">PLoS Negl Trop Dis</journal-id>
    journal_container = root.find(".//journal-id[@journal-id-type='nlm-ta']")
    if journal_container is None:
        print(f"No journal name found for: '{jats_xml_filepath}'")
        return None
    if journal_container.text is None:
        print(f"No journal name found for: '{jats_xml_filepath}'")
        return None
    journal_name = journal_container.text

    # Get the Journal PMC ID
    # Example:
    # <journal-id journal-id-type="pmc">plosntds</journal-id>
    journal_pmc_id_container = root.find(".//journal-id[@journal-id-type='pmc']")
    if journal_pmc_id_container is None:
        print(f"No journal PMC ID found for: '{jats_xml_filepath}'")
        return None
    if journal_pmc_id_container.text is None:
        print(f"No journal PMC ID found for: '{jats_xml_filepath}'")
        return None
    journal_pmc_id = journal_pmc_id_container.text

    return JournalInfo(journal_name=journal_name, journal_pmc_id=journal_pmc_id)


@dataclass
class ArticleBasicInfo:
    doi: str
    disciplines: str
    publish_date: str


def _get_article_basic_info(
    root: ET.Element, jats_xml_filepath: str
) -> ArticleBasicInfo | None:
    # Get the DOI
    # Example:
    # <article-id pub-id-type="doi">10.1371/journal.pntd.0002114</article-id>
    doi_container = root.find(".//article-id[@pub-id-type='doi']")
    if doi_container is None:
        print(f"No DOI found for: '{jats_xml_filepath}'")
        return None
    if doi_container.text is None:
        print(f"No DOI found for: '{jats_xml_filepath}'")
        return None
    doi = doi_container.text

    # Get the top level disciplines
    # Example:
    # <subj-group subj-group-type="Discipline-v3">
    # <subject>Biology and life sciences</subject><subj-group><subject>Chronobiology</subject><subj-group><subject>Circadian rhythms</subject></subj-group></subj-group></subj-group><subj-group subj-group-type="Discipline-v3">  # noqa: E501
    # <subject>Research and analysis methods</subject><subj-group><subject>Bioassays and physiological analysis</subject><subj-group><subject>Microarrays</subject></subj-group></subj-group></subj-group>  # noqa: E501
    # Should be: Biology and life sciences, Medicine and health sciences, Computer and information sciences, Physical sciences  # noqa: E501
    discipline_containers = root.findall(
        ".//subj-group[@subj-group-type='Discipline-v3']"
    )
    if len(discipline_containers) == 0:
        print(f"No disciplines found for: '{jats_xml_filepath}'")
        return None

    disciplines_list = []
    for container in discipline_containers:
        subject_container = container.find(".//subject")
        if subject_container is not None:
            if subject_container.text is not None:
                disciplines_list.append(subject_container.text)
    disciplines = ";".join(set(disciplines_list))

    # Get the published date
    # Example:
    # <pub-date pub-type="epub">
    # <day>29</day>
    # <month>2</month>
    # <year>2024</year>
    # </pub-date>
    pub_date_container = root.find(".//pub-date[@pub-type='epub']")
    if pub_date_container is None:
        print(f"No publish date found for: '{jats_xml_filepath}'")
        return None

    pub_year = pub_date_container.find(".//year")
    pub_month = pub_date_container.find(".//month")
    pub_day = pub_date_container.find(".//day")
    if pub_year is None or pub_month is None or pub_day is None:
        print(f"Invalid publish date found for: '{jats_xml_filepath}'")
        return None
    if pub_year.text is None or pub_month.text is None or pub_day.text is None:
        print(f"Invalid publish date found for: '{jats_xml_filepath}'")
        return None
    publish_date = (
        datetime(
            year=int(pub_year.text),
            month=int(pub_month.text),
            day=int(pub_day.text),
        )
        .date()
        .isoformat()
    )

    return ArticleBasicInfo(
        doi=doi,
        disciplines=disciplines,
        publish_date=publish_date,
    )


@dataclass
class ArticleTitleAndAbstract:
    full_title: str
    short_title: str
    abstract: str


def _get_article_title_and_abstract(
    root: ET.Element, jats_xml_filepath: str
) -> ArticleTitleAndAbstract | None:
    # Get the full title
    # Example:
    # <article-title>TimeTeller: A tool to probe the circadian clock as a multigene dynamical system</article-title>  # noqa: E501
    full_title_container = root.find(".//article-title")
    if full_title_container is None:
        print(f"No full title found for: '{jats_xml_filepath}'")
        return None
    if full_title_container.text is None:
        print(f"No full title found for: '{jats_xml_filepath}'")
        return None
    full_title = full_title_container.text

    # Get the short title
    # Example:
    # <alt-title alt-title-type="running-head">TimeTeller</alt-title>
    short_title_container = root.find(".//alt-title[@alt-title-type='running-head']")
    if short_title_container is None:
        print(f"No short title found for: '{jats_xml_filepath}'")
        return None
    if short_title_container.text is None:
        print(f"No short title found for: '{jats_xml_filepath}'")
        return None
    short_title = short_title_container.text

    # Get the abstract
    # Example:
    # <abstract>
    #       <p>Recent studies have established that the circadian clock influences onset, progression and therapeutic outcomes in a number of diseases including cancer and heart diseases. Therefore, there is a need for tools to measure the functional state of the molecular circadian clock and its downstream targets in patients. Moreover, the clock is a multi-dimensional stochastic oscillator and there are few tools for analysing it as a noisy multigene dynamical system. In this paper we consider the methodology behind TimeTeller, a machine learning tool that analyses the clock as a noisy multigene dynamical system and aims to estimate circadian clock function from a single transcriptome by modelling the multi-dimensional state of the clock. We demonstrate its potential for clock systems assessment by applying it to mouse, baboon and human microarray and RNA-seq data and show how to visualise and quantify the global structure of the clock, quantitatively stratify individual transcriptomic samples by clock dysfunction and globally compare clocks across individuals, conditions and tissues thus highlighting its potential relevance for advancing circadian medicine.</p>  # noqa: E501
    # </abstract>
    # Note: there are two "abstract" tags, one for the full and one for the abstract-type="summary"  # noqa: E501
    # We want to select the first, and full, abstract
    abstract_container = root.find(".//abstract")
    if abstract_container is None:
        print(f"No abstract found for: '{jats_xml_filepath}'")
        return None
    if abstract_container.text is None:
        print(f"No abstract found for: '{jats_xml_filepath}'")
        return None
    abstract = abstract_container.text

    return ArticleTitleAndAbstract(
        full_title=full_title,
        short_title=short_title,
        abstract=abstract,
    )


@dataclass
class AcknowledgementAndFundingInfo:
    acknowledgement_statement: str
    funding_statement: str
    funding_sources: str


def _get_ack_and_funding_info(
    root: ET.Element, jats_xml_filepath: str
) -> AcknowledgementAndFundingInfo | None:
    # Get the acknowledgement statement
    # Example:
    # <ack>
    # <p>We thank the anonymous reviewers for their helpful comments.</p>
    # </ack>
    acknowledgement_statement_container = root.find(".//ack")
    if acknowledgement_statement_container is None:
        print(f"No acknowledgement statement found for: '{jats_xml_filepath}'")
        return None
    if acknowledgement_statement_container.text is None:
        print(f"No acknowledgement statement found for: '{jats_xml_filepath}'")
        return None
    acknowledgement_statement = acknowledgement_statement_container.text

    # Get the funding sources
    # Example:
    # <funding-group>
    # <award-group id="award001">
    # <funding-source>
    # <institution-wrap>
    # <institution-id institution-id-type="funder-id">http://dx.doi.org/10.13039/501100000266</institution-id>  # noqa: E501
    # <institution>Engineering and Physical Sciences Research Council</institution>
    # </institution-wrap>
    # </funding-source>
    # <award-id>EP/F500378/1</award-id>
    # <principal-award-recipient>
    # <name name-style="western">
    # <surname>Vlachou</surname> <given-names>Denise</given-names></name>
    # </principal-award-recipient>
    # </award-group>
    # <award-group id="award002">
    # <funding-source>
    # <institution-wrap>
    # <institution-id institution-id-type="funder-id">http://dx.doi.org/10.13039/501100000266</institution-id>  # noqa: E501
    # <institution>Engineering and Physical Sciences Research Council</institution>
    # </institution-wrap>
    # </funding-source>
    funding_group = root.findall(".//funding-group")
    if len(funding_group) == 0:
        print(f"No funding sources found for: '{jats_xml_filepath}'")
        return None

    funding_sources_list = []
    for source in funding_group:
        institution_container = source.find(".//institution")
        if institution_container is not None:
            if institution_container.text is not None:
                funding_sources_list.append(institution_container.text)
    funding_sources = ";".join(set(funding_sources_list))

    # Get the full funding statement
    # Example:
    # <funding-statement>This work was supported by the UK Engineering &amp; Physical Sciences Research Council (EPSRC) (MOAC Doctoral Training Centre grant number EP/F500378/1 for DV and EP/P019811/1 to DAR), by the UK Biotechnology and Biological Sciences Research Council (BB/K003097/1 to DAR), by Cancer Research UK and EPSRC (C53561/A19933 to MV, RD &amp; DAR), by the Anna-Liisa Farquharson Chair in Renal Cell Cancer Research (to GAB) and the UK Medical Research Council Doctoral Training Partnership (MR/N014294/1 for LU and VV). The funders played no role in study design, data collection and analysis, the decision to publish, or the preparation of the manuscript.</funding-statement>  # noqa: E501
    funding_statement_container = root.find(".//funding-statement")
    if funding_statement_container is None:
        print(f"No funding statement found for: '{jats_xml_filepath}'")
        return None
    if funding_statement_container.text is None:
        print(f"No funding statement found for: '{jats_xml_filepath}'")
        return None
    funding_statement = funding_statement_container.text

    return AcknowledgementAndFundingInfo(
        acknowledgement_statement=acknowledgement_statement,
        funding_statement=funding_statement,
        funding_sources=funding_sources,
    )


def _get_authors(root: ET.Element, jats_xml_filepath: str) -> list[Author] | None:
    # Get the authors
    # Example:
    # <contrib-group>
    # <contrib contrib-type="author" equal-contrib="yes" xlink:type="simple">
    # <name name-style="western">
    # <surname>Vlachou</surname> <given-names>Denise</given-names></name>
    # <role content-type="http://credit.niso.org/contributor-roles/data-curation/">Data curation</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/methodology/">Methodology</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/software/">Software</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/validation/">Validation</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/writing-original-draft/">Writing - original draft</role>  # noqa: E501
    # <xref ref-type="aff" rid="aff001"><sup>1</sup></xref>
    # <xref ref-type="fn" rid="currentaff001"><sup>¤a</sup></xref>
    # </contrib>
    # <contrib contrib-type="author" equal-contrib="yes" xlink:type="simple">
    # <name name-style="western">
    # <surname>Veretennikova</surname> <given-names>Maria</given-names></name>
    # <role content-type="http://credit.niso.org/contributor-roles/data-curation/">Data curation</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/investigation/">Investigation</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/software/">Software</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/validation/">Validation</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/visualization/">Visualization</role>  # noqa: E501
    # <role content-type="http://credit.niso.org/contributor-roles/writing-review-editing/">Writing - review &amp; editing</role>  # noqa: E501
    # <xref ref-type="aff" rid="aff001"><sup>1</sup></xref>
    # <xref ref-type="fn" rid="currentaff002"><sup>¤b</sup></xref>
    # </contrib>
    # AND LATER
    # <aff id="aff001">
    # <label>1</label>
    # <addr-line>Mathematics Institute &amp; Zeeman Institute for Systems Biology and Infectious Disease Epidemiology Research, University of Warwick, Coventry, United Kingdom</addr-line>  # noqa: E501
    # </aff>
    # <aff id="aff002">
    # <label>2</label>
    # <addr-line>Division of Biomedical Sciences, Warwick Medical School, University of Warwick, Coventry, United Kingdom</addr-line>  # noqa: E501
    # </aff>
    # <aff id="aff003">
    # <label>3</label>
    # <addr-line>Odette Cancer Centre, Sunnybrook Health Sciences Centre, Toronto, Ontario, Canada</addr-line>  # noqa: E501
    # </aff>
    # <aff id="aff004">
    # <label>4</label>
    # <addr-line>Department of Statistics, University of Warwick, Coventry, United Kingdom</addr-line>  # noqa: E501
    # </aff>
    # <aff id="aff005">
    # <label>5</label>
    # <addr-line>UPR “Chronotherapy, Cancer and Transplantation”, Medical School, Paris-Saclay University, Medical Oncology Department, Paul Brousse Hospital, Villejuif, France</addr-line>  # noqa: E501
    # </aff>
    author_containers = root.findall(".//contrib[@contrib-type='author']")
    if len(author_containers) == 0:
        print(f"No authors found for: '{jats_xml_filepath}'")
        return None

    authors = []
    for author in author_containers:
        name_container = author.find(".//name")
        email_container = author.find(".//email")
        affliation_xref_container = author.find(".//xref[@ref-type='aff']")
        if (
            name_container is None
            or email_container is None
            or affliation_xref_container is None
        ):
            print(f"Invalid author found for: '{jats_xml_filepath}'")
            continue

        if (
            name_container.text is None
            or email_container.text is None
            or "rid" not in affliation_xref_container.attrib
        ):
            print(f"Invalid author found for: '{jats_xml_filepath}'")
            continue
        full_name = name_container.text
        email = email_container.text
        affliation_id = affliation_xref_container.attrib["rid"]

        # Get the affliation
        affliation_container = root.find(f".//aff[@id='{affliation_id}']")
        if affliation_container is None:
            print(f"No affliation found for: '{jats_xml_filepath}'")
            continue
        # Get the addr-line
        addr_line_container = affliation_container.find(".//addr-line")
        if addr_line_container is None:
            print(f"No addr-line found for: '{jats_xml_filepath}'")
            continue
        if addr_line_container.text is None:
            print(f"No addr-line found for: '{jats_xml_filepath}'")
            continue
        affliation = addr_line_container.text

        roles_containers = author.findall(".//role")
        if len(roles_containers) == 0:
            roles = None
        else:
            role_texts = []
            for role in roles_containers:
                if role.text is not None:
                    role_texts.append(role.text)
            roles = ";".join(role_texts)

        authors.append(
            Author(
                full_name=full_name,
                email=email,
                affliation=affliation,
                roles=roles,
            ),
        )

    return authors


def process_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the PLOS corpus to retrieve basic information from each article.

    Parameters
    ----------
    df : pd.DataFrame
        The PLOS corpus.

    Returns
    -------
    pd.DataFrame
        The processed PLOS corpus.
    """
    # Store final results
    results = []

    # Process each row of the provided dataframe, load the XML and extract relevant info
    for jats_xml_filepath in tqdm(df.jats_xml_path, desc="Processing PLOS Corpus"):
        # Load the XML
        tree = ET.parse(jats_xml_filepath)
        root = tree.getroot()

        # Get the repository URL
        repository_url = _get_repository_url(root, jats_xml_filepath)
        if repository_url is None:
            continue

        # Get the journal info
        journal_info = _get_journal_info(root, jats_xml_filepath)
        if journal_info is None:
            continue

        # Get the article info
        article_basic_info = _get_article_basic_info(root, jats_xml_filepath)
        if article_basic_info is None:
            continue

        # Get the article title and abstract
        article_title_and_abstract = _get_article_title_and_abstract(
            root, jats_xml_filepath
        )
        if article_title_and_abstract is None:
            continue

        # Get the acknowledgement and funding info
        ack_and_funding_info = _get_ack_and_funding_info(root, jats_xml_filepath)
        if ack_and_funding_info is None:
            continue

        # Get the authors
        authors = _get_authors(root, jats_xml_filepath)
        if authors is None:
            continue

        # Store the results
        results.append(
            ArticleProcessingResult(
                journal_name=journal_info.journal_name,
                journal_pmc_id=journal_info.journal_pmc_id,
                doi=article_basic_info.doi,
                full_title=article_title_and_abstract.full_title,
                short_title=article_title_and_abstract.short_title,
                abstract=article_title_and_abstract.abstract,
                disciplines=article_basic_info.disciplines,
                repository_url=repository_url,
                authors=authors,
                acknowledgement_statement=ack_and_funding_info.acknowledgement_statement,
                funding_statement=ack_and_funding_info.funding_statement,
                funding_sources=ack_and_funding_info.funding_sources,
                publish_date=article_basic_info.publish_date,
            )
        )

    # For each result, convert to author long format dataframe
    per_author_results = []
    for result in results:
        for author in result.authors:
            per_author_results.append(
                {
                    "journal_name": result.journal_name,
                    "journal_pmc_id": result.journal_pmc_id,
                    "doi": result.doi,
                    "full_title": result.full_title,
                    "short_title": result.short_title,
                    "abstract": result.abstract,
                    "disciplines": result.disciplines,
                    "repository_url": result.repository_url,
                    "acknowledgement_statement": result.acknowledgement_statement,
                    "funding_statement": result.funding_statement,
                    "funding_sources": result.funding_sources,
                    "publish_date": result.publish_date,
                    "full_name": author.full_name,
                    "email": author.email,
                    "affliation": author.affliation,
                    "roles": author.roles,
                }
            )

    # Return the results
    return pd.DataFrame(per_author_results)
