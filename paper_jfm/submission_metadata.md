# Journal of Futures Markets — Submission Metadata

## Journal
Journal of Futures Markets

## Article type
Research Article

## Title
How Much Does Limit Order Book Microstructure Add Beyond Quote Geometry? Evidence from Cryptocurrency Limit Order Books

## Running title
LOB Information Beyond Quotes

## Author
Jude Kriel Ramcharitar

## Affiliation
Independent Researcher

## Email
j.k.ramcharitar@gmail.com

## ORCID
Required by the journal. Connect or enter the author's existing ORCID in the Wiley submission system.

## Abstract
We measure the incremental predictive information in limit order book (LOB) microstructure after conditioning on quote side, distance from mid, and holding horizon. Using 269,330 10-level LOB snapshots from Kraken for BTC/USD, ETH/USD, and SOL/USD, we predict a replay-defined crossing outcome at 300- and 900-second horizons against a 16-cell lookup reference using quote geometry alone. Under a forecasting-ordered split, gradient-boosted trees using 63 microstructure features improve AUC by 0.0480 on BTC/USD and 0.0596 on ETH/USD, with paired block-bootstrap 95% confidence intervals of [0.0400, 0.0551] and [0.0427, 0.0751]. The SOL/USD gap is -0.0105 with a 95% interval of [-0.0283, 0.0075], and is not distinguishable from zero. An exploratory blocked diagnostic produces positive gaps across all three assets and all ten block rotations, but is not interpreted as prospective forecasting evidence. The results show robust incremental predictive information beyond quote geometry for BTC/USD and ETH/USD, while highlighting sensitivity to temporal distribution shift and pipeline validation.

## Keywords
limit order book; market microstructure; crossing probability; gradient boosting; predictive information; reproducibility

## JEL
G12; G14; C45; C61

## Funding
No funding was received for this work.

## Conflict of interest
The author declares no conflicts of interest and holds no positions in any asset discussed in the manuscript.

## Data availability
The analysis code, collection pipeline, label builder, model code, verification code, bootstrap script, and diagnostics are publicly available at https://github.com/juderamcharitar/latency-compensating-mm. The raw market-data archive is not currently public and is available from the author pending deposit in a persistent archive.

## Preprint / prior dissemination
Earlier working versions of this research have been disseminated through preprint or research repositories and the author's website. The manuscript is not under consideration by another journal. Journal of Futures Markets explicitly permits prior posting on preprint servers.

## AI disclosure
Generative AI tools were used during implementation, diagnostic analysis, manuscript revision, and submission preparation. Claude (Anthropic) was used for implementation assistance, diagnostics, and drafting/revision. ChatGPT (OpenAI) was used for manuscript revision, submission preparation, and implementation assistance for the dependence-aware bootstrap analysis. All substantive research decisions, verification of reported results, citation checking, and responsibility for the final content remained with the author. This disclosure is included in the Methods section of the anonymized manuscript.

## Files to upload
1. `main_anonymized.tex` — designation: Main Document – LaTeX .tex File
2. A compiled PDF of `main_anonymized.tex` — designation: Main Document – LaTeX PDF
3. `title_page.tex` or a compiled title-page file — separate Title Page
4. `cover_letter.md` or converted PDF/DOCX — optional Cover Letter

## Important portal choices
- Peer review: double-anonymized; do not upload the named Quantitative Finance PDF as the reviewer manuscript.
- Special issue: No, unless deliberately submitting to a currently open special issue.
- Open Access: optional only after acceptance; choose the standard subscription route if avoiding publication charges.
- Data sharing: journal expects data sharing; use the data-availability statement above and answer the portal accurately.
