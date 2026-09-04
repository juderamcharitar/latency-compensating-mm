September 4, 2026

Editors
Quantitative Finance

Dear Editors,

Please consider my manuscript, “How Much Does Limit Order Book Microstructure Add Beyond Quote Geometry? Evidence from Cryptocurrency Limit Order Books,” for publication as an Original Article in Quantitative Finance.

The paper asks a narrow market-microstructure question: after conditioning on quote side, distance from mid, and holding horizon, how much incremental predictive information remains in observable limit order book state? Using 269,330 depth-10 Kraken snapshots across BTC/USD, ETH/USD, and SOL/USD, I compare gradient-boosted and recurrent models with an explicit lookup benchmark based only on quote geometry. In the primary forecasting-ordered evaluation, microstructure features improve AUC by 0.0480 on BTC/USD and 0.0596 on ETH/USD, with dependence-aware paired block-bootstrap 95% confidence intervals of [0.0400, 0.0551] and [0.0427, 0.0751]. The corresponding SOL/USD effect is not distinguishable from zero. A secondary blocked analysis is reported only as an information-content diagnostic because it is not prospective.

I believe the manuscript fits Quantitative Finance particularly well because it sits directly at the intersection of market microstructure, liquidity modelling, financial econometrics, and market prediction. Its main contribution is not another high-performing LOB model, but an interpretable measurement of what book state adds beyond information mechanically available from quote placement. The paper also documents validation checks that detected several plausible-looking but incorrect results during development, illustrating how temporal splitting, target construction, and simple baselines can materially change empirical conclusions.

Earlier working versions of this project have been disseminated publicly through GitHub, SSRN, Zenodo, a personal website, ResearchGate, and ORCID, and a preprint submission to arXiv is currently pending moderation. Those earlier versions should not be treated as the submitted manuscript. The research question, labels, empirical claims, and validation framework were substantially revised after defects were identified in the earlier pipeline. The present Version 5.0 is the journal-submission candidate and supersedes those prior versions. The manuscript is not under consideration by another journal.

Code for the current analysis is publicly available in the accompanying GitHub repository. The underlying raw market-data archive is not currently deposited publicly; the manuscript states its availability status and the reproducibility code is provided.

Generative AI tools were used during the research and manuscript-preparation process. Claude (Anthropic) was used for implementation assistance, diagnostic analysis, and drafting/revision. ChatGPT (OpenAI) was used for manuscript revision, submission preparation, and implementation of the dependence-aware bootstrap analysis. I designed and directed the study, made the substantive research decisions, verified the reported results, and take full responsibility for the manuscript, code, citations, and conclusions. The manuscript contains a disclosure of AI assistance.

I declare no conflicts of interest and received no funding for this work.

Thank you for considering the manuscript.

Sincerely,

Jude Kriel Ramcharitar
Independent Researcher
j.k.ramcharitar@gmail.com
