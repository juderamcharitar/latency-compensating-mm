# Quantitative Finance Submission Metadata

## Article type
Original Article

## Title
How Much Does Limit Order Book Microstructure Add Beyond Quote Geometry? Evidence from Cryptocurrency Limit Order Books

## Author
Jude Kriel Ramcharitar

## Affiliation
Independent Researcher

## Corresponding author email
j.k.ramcharitar@gmail.com

## Abstract
We measure the incremental predictive information in limit order book (LOB) microstructure after conditioning on quote side, distance from mid, and holding horizon. Using 269,330 depth-10 Kraken snapshots for BTC/USD, ETH/USD, and SOL/USD over 70.5 days, we predict a replay-defined crossing outcome at 300- and 900-second horizons against a 16-cell lookup benchmark using only quote geometry. Under a snapshot-grouped forecasting-ordered split with a 990-second embargo, gradient-boosted trees using 63 microstructure features improve AUC by 0.0480 on BTC/USD and 0.0596 on ETH/USD. Paired one-hour block-bootstrap 95% confidence intervals are [0.0400, 0.0551] and [0.0427, 0.0751]. For SOL/USD, the gap is -0.0105 with a 95% interval of [-0.0283, 0.0075]. An exploratory blocked diagnostic produces positive gaps on all three assets across all ten block rotations, but is interpreted as an information-content test rather than prospective forecasting because training observations may post-date test observations. The results provide evidence that LOB state contains incremental predictive information beyond quote geometry for BTC/USD and ETH/USD, while showing that split design and pipeline validation materially affect conclusions.

Word count: 169

## Keywords
limit order book; market microstructure; crossing probability; gradient boosting; predictive information; reproducibility

## JEL codes
G12; G14; C45; C61

## Funding
No external funding.

## Conflicts of interest
The author declares no conflicts of interest and holds no positions in any asset discussed in the manuscript.

## Ethics statement
Not applicable. The study uses market data and does not involve human participants, animals, or personally identifiable information.

## Data availability statement
The collection pipeline, label builder, model code, verification code, and bootstrap analysis are publicly available in the project GitHub repository. The underlying raw market-data archive is not currently deposited in a persistent public repository and is available from the author subject to practical transfer constraints. The manuscript distinguishes the replay-defined crossing outcome from realised executions.

## Code availability
https://github.com/juderamcharitar/latency-compensating-mm

## Prior dissemination
Earlier working versions have appeared on GitHub, SSRN, Zenodo, the author's personal website, ResearchGate, and ORCID. A preprint submission to arXiv is pending moderation. Version 5.0 supersedes those earlier working versions for journal submission.

## Exclusive submission statement
This manuscript is not under consideration by another journal.

## Generative AI disclosure
Claude (Anthropic) was used for implementation assistance, diagnostic analysis, and drafting/revision. ChatGPT (OpenAI) was used for manuscript revision, submission preparation, and implementation of the dependence-aware bootstrap analysis. The author designed and directed the study, made all substantive research decisions, verified the reported results, and accepts full responsibility for the manuscript, code, citations, and conclusions.

## Suggested short title
Incremental Information in Limit Order Book Microstructure

## Suggested subject areas
Market microstructure; liquidity modelling; financial econometrics; market dynamics and prediction; machine learning in finance

## Files to upload
1. Main manuscript PDF or source package generated from `main.tex` and `refs.bib`.
2. Cover letter (`cover_letter.md`, converted to PDF/DOCX only if the portal requires a file upload).
3. Any supplementary code/data statement requested by the portal.

## Final pre-submit checks
- Replace the manuscript abstract with the 169-word submission abstract if the portal/manuscript checker enforces the 200-word limit.
- Compile the V5 LaTeX source with `refs.bib` and check for unresolved citations/references.
- Confirm final page count is within the journal's stated typical 35-page limit.
- Confirm every citation in the text resolves to the bibliography.
- Confirm the current AI disclosure matches the submission form answer.
- Confirm prior dissemination is disclosed consistently in the cover letter and portal.
