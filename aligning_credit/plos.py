#!/usr/bin/env python

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: N817

import pandas as pd
import tldextract
from allofplos.corpus.plos_corpus import get_corpus_dir
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

from .types import ErrorResult, SuccessAndErroredResults
from .utils.code_hosts import CodeHostResult, parse_code_host_url

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _load_unprocessed_corpus(
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
    email: str | None
    affliation: str | None
    roles: str | None


@dataclass
class ArticleProcessingResult(DataClassJsonMixin):
    journal_name: str
    journal_pmc_id: str | None
    doi: str
    full_title: str
    short_title: str | None
    abstract: str
    disciplines: str | None
    repository_details: CodeHostResult
    authors: list[Author]
    acknowledgement_statement: str | None
    funding_statement: str | None
    funding_sources: str | None
    publish_date: str


def _get_repository_url(
    root: ET.Element,
    jats_xml_filepath: str,
) -> CodeHostResult | ErrorResult:
    # Before anything else, check for data availability
    # Example:
    # <custom-meta id="data-availability">
    #     <meta-name>Data Availability</meta-name>
    #     <meta-value>A TimeTeller R package is available from GitHub at <ext-link ext-link-type="uri" xlink:href="https://github.com/VadimVasilyev1994/TimeTeller" xlink:type="simple">https://github.com/VadimVasilyev1994/TimeTeller</ext-link>. All data used except that of Bjarnasson et al. is publicly available from the sites given in the Supplementary Information. The Bjarnasson et al. data that is used is available with the R package.</meta-value>  # noqa: E501
    # </custom-meta>
    data_availability = root.find(".//custom-meta[@id='data-availability']")
    if data_availability is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Data Availability",
            error="No data-availability block found",
        )

    # Check if data availablity points to code host
    ext_link = data_availability.find(".//ext-link[@ext-link-type='uri']")
    if ext_link is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Data Availability",
            error="No link found within data-availability-block",
        )

    # Get the url
    repository_url = ext_link.text

    # Check for text
    if not isinstance(repository_url, str):
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Data Availability",
            error="No link found within data-availability-block",
        )

    # Attempt to parse url
    try:
        return parse_code_host_url(repository_url)
    except ValueError as e:
        # Extract components of URL
        ext = tldextract.extract(repository_url)

        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Data Availability",
            error=str(e),
            extra_data=f"{ext.subdomain}.{ext.domain}.{ext.suffix}",
        )


@dataclass
class JournalInfo:
    journal_name: str
    journal_pmc_id: str | None


def _get_journal_info(
    root: ET.Element,
    jats_xml_filepath: str,
) -> JournalInfo | ErrorResult:
    # Get the Journal Name
    # Example:
    # <journal-id journal-id-type="nlm-ta">PLoS Negl Trop Dis</journal-id>
    journal_container = root.find(".//journal-id[@journal-id-type='nlm-ta']")
    if journal_container is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Journal Parsing",
            error="No journal ID found",
        )
    if journal_container.text is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Journal Parsing",
            error="No journal ID found",
        )

    journal_name = journal_container.text

    # Get the Journal PMC ID
    # Example:
    # <journal-id journal-id-type="pmc">plosntds</journal-id>
    journal_pmc_id_container = root.find(".//journal-id[@journal-id-type='pmc']")
    if journal_pmc_id_container is None:
        journal_pmc_id = None
    else:
        journal_pmc_id = journal_pmc_id_container.text

    return JournalInfo(journal_name=journal_name, journal_pmc_id=journal_pmc_id)


@dataclass
class ArticleBasicInfo:
    doi: str
    disciplines: str | None
    publish_date: str


def _get_article_basic_info(
    root: ET.Element, jats_xml_filepath: str
) -> ArticleBasicInfo | ErrorResult:
    # Get the DOI
    # Example:
    # <article-id pub-id-type="doi">10.1371/journal.pntd.0002114</article-id>
    doi_container = root.find(".//article-id[@pub-id-type='doi']")
    if doi_container is None or doi_container.text is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="DOI Parsing",
            error="No article DOI found",
        )
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
    disciplines_list = []
    for container in discipline_containers:
        subject_container = container.find(".//subject")
        if subject_container is not None:
            if subject_container.text is not None:
                disciplines_list.append(subject_container.text)
    disciplines: str | None = ";".join(set(disciplines_list))

    # If disciplines is empty, set to None instead of empty string
    if disciplines == "":
        disciplines = None

    # Get the published date
    # Example:
    # <pub-date pub-type="epub">
    # <day>29</day>
    # <month>2</month>
    # <year>2024</year>
    # </pub-date>
    pub_date_container = root.find(".//pub-date[@pub-type='epub']")
    if pub_date_container is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Publish Date Parsing",
            error="No publish date found",
        )

    pub_year = pub_date_container.find(".//year")
    pub_month = pub_date_container.find(".//month")
    pub_day = pub_date_container.find(".//day")
    if (
        pub_year is None
        or pub_month is None
        or pub_day is None
        or pub_year.text is None
        or pub_month.text is None
        or pub_day.text is None
    ):
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Publish Date Parsing",
            error="No publish date found",
        )

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
    short_title: str | None
    abstract: str


def _get_article_title_and_abstract(
    root: ET.Element, jats_xml_filepath: str
) -> ArticleTitleAndAbstract | ErrorResult:
    # Get the full title
    # Example:
    # <article-title>TimeTeller: A tool to probe the circadian clock as a multigene dynamical system</article-title>  # noqa: E501
    full_title_container = root.find(".//article-title")
    if full_title_container is None or full_title_container.text is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Article Title Parsing",
            error="No article title found",
        )

    full_title = full_title_container.text

    # Get the short title
    # Example:
    # <alt-title alt-title-type="running-head">TimeTeller</alt-title>
    short_title_container = root.find(".//alt-title[@alt-title-type='running-head']")
    if short_title_container is None or short_title_container.text is None:
        short_title = None
    else:
        short_title = short_title_container.text

    # Get the abstract
    # Example:
    # <abstract>
    #       <p>Recent studies have established that the circadian clock influences onset, progression and therapeutic outcomes in a number of diseases including cancer and heart diseases. Therefore, there is a need for tools to measure the functional state of the molecular circadian clock and its downstream targets in patients. Moreover, the clock is a multi-dimensional stochastic oscillator and there are few tools for analysing it as a noisy multigene dynamical system. In this paper we consider the methodology behind TimeTeller, a machine learning tool that analyses the clock as a noisy multigene dynamical system and aims to estimate circadian clock function from a single transcriptome by modelling the multi-dimensional state of the clock. We demonstrate its potential for clock systems assessment by applying it to mouse, baboon and human microarray and RNA-seq data and show how to visualise and quantify the global structure of the clock, quantitatively stratify individual transcriptomic samples by clock dysfunction and globally compare clocks across individuals, conditions and tissues thus highlighting its potential relevance for advancing circadian medicine.</p>  # noqa: E501
    # </abstract>
    # Note: there are two "abstract" tags, one for the full and one for the abstract-type="summary"  # noqa: E501
    # We want to select the first, and full, abstract
    abstract_container = root.find(".//abstract")
    if abstract_container is None or abstract_container.text is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Abstract Parsing",
            error="No abstract found",
        )
    abstract = " ".join(abstract_container.itertext()).strip()

    return ArticleTitleAndAbstract(
        full_title=full_title,
        short_title=short_title,
        abstract=abstract,
    )


@dataclass
class AcknowledgementAndFundingInfo:
    acknowledgement_statement: str | None
    funding_statement: str | None
    funding_sources: str | None


def _get_ack_and_funding_info(root: ET.Element) -> AcknowledgementAndFundingInfo:
    # Get the acknowledgement statement
    # Example:
    # <ack>
    # <p>We thank the anonymous reviewers for their helpful comments.</p>
    # </ack>
    acknowledgement_statement_container = root.find(".//ack")
    if (
        acknowledgement_statement_container is None
        or acknowledgement_statement_container.text is None
    ):
        acknowledgement_statement = None
    else:
        acknowledgement_statement = " ".join(
            acknowledgement_statement_container.itertext()
        ).strip()

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
    funding_sources_list = []
    for source in funding_group:
        institution_container = source.find(".//institution")
        if institution_container is not None:
            if institution_container.text is not None:
                funding_sources_list.append(institution_container.text)
    funding_sources: str | None = ";".join(set(funding_sources_list))

    # If funding sources is empty, set to None instead of empty string
    if funding_sources == "":
        funding_sources = None

    # Get the full funding statement
    # Example:
    # <funding-statement>This work was supported by the UK Engineering &amp; Physical Sciences Research Council (EPSRC) (MOAC Doctoral Training Centre grant number EP/F500378/1 for DV and EP/P019811/1 to DAR), by the UK Biotechnology and Biological Sciences Research Council (BB/K003097/1 to DAR), by Cancer Research UK and EPSRC (C53561/A19933 to MV, RD &amp; DAR), by the Anna-Liisa Farquharson Chair in Renal Cell Cancer Research (to GAB) and the UK Medical Research Council Doctoral Training Partnership (MR/N014294/1 for LU and VV). The funders played no role in study design, data collection and analysis, the decision to publish, or the preparation of the manuscript.</funding-statement>  # noqa: E501
    funding_statement_container = root.find(".//funding-statement")
    if funding_statement_container is None or funding_statement_container.text is None:
        funding_statement = None
    else:
        funding_statement = funding_statement_container.text

    return AcknowledgementAndFundingInfo(
        acknowledgement_statement=acknowledgement_statement,
        funding_statement=funding_statement,
        funding_sources=funding_sources,
    )


def _get_authors(  # noqa: C901
    root: ET.Element,
    jats_xml_filepath: str,
) -> list[Author] | ErrorResult:
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
    # Get the first contrib-group
    contrib_group = root.find(".//contrib-group")
    if contrib_group is None:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Authors Parsing",
            error="No contrib-group found",
        )

    author_containers = contrib_group.findall(".//contrib[@contrib-type='author']")
    if len(author_containers) == 0:
        return ErrorResult(
            jats_xml_path=jats_xml_filepath,
            step="Authors Parsing",
            error="No author found",
        )

    authors = []
    for author in author_containers:
        given_names_container = author.find(".//given-names")
        surname_container = author.find(".//surname")
        if (
            given_names_container is None
            or given_names_container.text is None
            or surname_container is None
            or surname_container.text is None
        ):
            return ErrorResult(
                jats_xml_path=jats_xml_filepath,
                step="Authors Parsing",
                error="No author name found",
            )

        full_name = f"{given_names_container.text} {surname_container.text}"

        email_container = author.find(".//email")
        affliation_xref_container = author.find(".//xref[@ref-type='aff']")
        if (
            email_container is None
            or affliation_xref_container is None
            or email_container.text is None
            or "rid" not in affliation_xref_container.attrib
        ):
            email = None
            affliation = None
        else:
            email = email_container.text
            affliation_id = affliation_xref_container.attrib["rid"]

            # Get the affliation
            affliation_container = root.find(f".//aff[@id='{affliation_id}']")
            if affliation_container is None:
                affliation = None
            else:
                # Get the addr-line
                addr_line_container = affliation_container.find(".//addr-line")
                if addr_line_container is None or addr_line_container.text is None:
                    affliation = None
                else:
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


def _process_plos_xml_files(  # noqa: C901
    df: pd.DataFrame,
) -> SuccessAndErroredResults:
    """
    Process the PLOS corpus to retrieve basic information from each article.

    Parameters
    ----------
    df : pd.DataFrame
        The PLOS corpus.

    Returns
    -------
    SuccessAndErroredResults
        The processed PLOS corpus (in success and errored dataframes)
    """
    # Store final results
    successful_results = []
    errored_results = []

    # Process each row of the provided dataframe, load the XML and extract relevant info
    for jats_xml_filepath in tqdm(df.jats_xml_path, desc="Processing PLOS Corpus"):
        # Load the XML
        try:
            tree = ET.parse(jats_xml_filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            log.error(f"Error parsing XML file: '{jats_xml_filepath}': {e}")
            errored_results.append(
                ErrorResult(
                    jats_xml_path=jats_xml_filepath,
                    step="XML Parsing",
                    error=str(e),
                ).to_dict()
            )
            continue

        # Get the repository URL
        repository_details = _get_repository_url(root, jats_xml_filepath)
        if isinstance(repository_details, ErrorResult):
            errored_results.append(repository_details.to_dict())
            continue

        # Get the journal info
        journal_info = _get_journal_info(root, jats_xml_filepath)
        if isinstance(journal_info, ErrorResult):
            errored_results.append(journal_info.to_dict())
            continue

        # Get the article info
        article_basic_info = _get_article_basic_info(root, jats_xml_filepath)
        if isinstance(article_basic_info, ErrorResult):
            errored_results.append(article_basic_info.to_dict())
            continue

        # Get the article title and abstract
        article_title_and_abstract = _get_article_title_and_abstract(
            root, jats_xml_filepath
        )
        if isinstance(article_title_and_abstract, ErrorResult):
            errored_results.append(article_title_and_abstract.to_dict())
            continue

        # Get the acknowledgement and funding info
        ack_and_funding_info = _get_ack_and_funding_info(root)
        if isinstance(ack_and_funding_info, ErrorResult):
            errored_results.append(ack_and_funding_info.to_dict())
            continue

        # Get the authors
        authors = _get_authors(root, jats_xml_filepath)
        if isinstance(authors, ErrorResult):
            errored_results.append(authors.to_dict())
            continue

        # Store the results
        successful_results.append(
            ArticleProcessingResult(
                journal_name=journal_info.journal_name,
                journal_pmc_id=journal_info.journal_pmc_id,
                doi=article_basic_info.doi,
                full_title=article_title_and_abstract.full_title,
                short_title=article_title_and_abstract.short_title,
                abstract=article_title_and_abstract.abstract,
                disciplines=article_basic_info.disciplines,
                repository_details=repository_details,
                authors=authors,
                acknowledgement_statement=ack_and_funding_info.acknowledgement_statement,
                funding_statement=ack_and_funding_info.funding_statement,
                funding_sources=ack_and_funding_info.funding_sources,
                publish_date=article_basic_info.publish_date,
            )
        )

    # For each result, convert to author long format dataframe
    per_author_results = []
    for result in successful_results:
        for author in result.authors:
            per_author_results.append(
                {
                    "jats_xml_path": jats_xml_filepath,
                    "journal_name": result.journal_name,
                    "journal_pmc_id": result.journal_pmc_id,
                    "doi": result.doi,
                    "full_title": result.full_title,
                    "short_title": result.short_title,
                    "abstract": result.abstract,
                    "disciplines": result.disciplines,
                    "repository_host": result.repository_details.host,
                    "repository_owner": result.repository_details.owner,
                    "repository_name": result.repository_details.name,
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
    return SuccessAndErroredResults(
        successful_results=pd.DataFrame(per_author_results),
        errored_results=pd.DataFrame(errored_results),
    )
