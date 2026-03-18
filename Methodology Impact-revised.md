# 2026-03-04 Methodology

### Overview

The pipeline comprises two LLM-driven stages applied to a corpus of news articles on emerging pests and pathogens (EPPs): (I) **relevance filtering** to identify articles containing _cascading impacts_ attributable to a target EPP; and (II) **impact extraction and grouping** into a constrained set of **Unique Impact Groups (UIGs)**. Both stages will use **ensemble inference** to <span style="background:#fff88f">improve robustness and reduce single-model prompt sensitivity</span>. Every step will be also paired with performance indicators that can be computed from logs and a human-labeled reference set where applicable .

![[influx-impact_revised.svg]]
Let the full corpus contain $N$ articles $\{a_i\}_{i=1}^{N}$. After preprocessing, articles proceed through Step 3 to produce a relevant subset $R$ of size $N_R$,  then Step 4 operates on $R$.

## 1. Data preparation and logging

### 1.1 Cleaning and standardization

Articles will be standardized via de-duplication (exact and near-duplicate), URL removal, and chunking (if needed) to fit context windows. All transformations will be deterministic and logged.

**Duplicate rate**
$$\text{DupRate} = \frac{N_{\text{dups}}}{N}$$

## 2. Step 3 — Ensemble LLM relevance filtering (cascading impacts)

### 2.1 Task definition

For each target EPP $h$, each article $a_i$ is classified:

- **Relevant (1)** if it contains at least one _downstream/cascading impact_ attributable to $h$, or to response measures for $h$,
- **Not relevant (0)** otherwise (mentions, case counts, generic biology, or unrelated references).

The model output is structured JSON including: binary label, confidence, short rationale, and evidence spans.

### 2.2 Reference labels and annotation quality

A random subset $S \subset \{a_i\}$ will be human-labeled with a codebook defining cascading impacts . The goal of this step is to create a **reliable benchmark dataset** that represents the correct classification of articles. This dataset allows the researchers to **compare LLM predictions against human judgment**. The sampling is **stratified**. Stratification ensures the evaluation dataset is **representative of the entire corpus**, based on:
- different EPP types
- different regions
- different publication years
- different languages

To ensure the reference dataset is **high quality and consistent**, each article is **labeled independently by two annotators**. Then we calculate **Cohen's kappa** index to measure **inter-annotator agreement**.

- **Cohen's kappa**:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
### 2.3 Ensemble design for relevance filtering

Step 3 uses an ensemble to reduce false positives. For each article $a_i$, we ran $K$ ($K$ = 3) independent relevance classifiers (different models **and/or** different prompt variants/seeds), producing:

- predicted label $\hat{y}_{ik} \in \{0, 1\}$,
- predicted probability/confidence $p_{ik} \in [0, 1]$,
- evidence spans $e_{ik}$.

**Aggregation (two-tier).**

1. **Weighted Probability pooling:** compute an aggregated relevance probability:

$$\bar{p}_i = \sum_{k=1}^{K} w_k p_{ik}, \quad \sum_k w_k = 1$$

Weights $w_k$ can be equal across classifier or learned from validation performance on the human-labeled set $S$ (*see 2.5*).

2. **Decision rule (high precision):** classify relevant only if:

$$\hat{y}_i = \mathbb{1}(\bar{p}_i \geq \theta)$$

with $\theta$ as the <span style="background:#fdbfff">threshold</span> (e.g., $\theta = 0.7$–$0.85$), and optionally <span style="background:#fdbfff">require a minimum agreement:</span>

$$\text{Agree}_i = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}(\hat{y}_{ik} = 1) \geq \tau$$

(e.g., $\tau = 0.6$). This "probability threshold + minimum agreement" prevents single-model hallucinated relevance.

**Evidence consolidation.** We retained the union of top evidence spans from the models supporting relevance (or the highest-weight model if non-relevant), and logged whether evidence was present and attributable.
### 2.4 Performance indicators

**Classification accuracy**

- Confusion matrix: TP, FP, TN, FN
- Precision / Recall / F1:

$$\text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN},\quad F1=\frac{2PR}{P+R}$$

- Balanced accuracy: dealing with imbalanced dataset (irrelevance/relevance)

$$\text{BalAcc}=\frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)$$

**Calibration**: the models output probabilities that an article is relevant. Calibration will tell whether those probabilities are **trustworthy**.
- Brier score: The formula calculates the **average squared error between predicted probabilities and the true outcome (0 and 1)**.

$$\text{Brier}=\frac{1}{|S|}\sum_{i\in S}(\bar{p}_i-y_i)^2$$

- Expected calibration error (ECE) with bins $B_m$: **ECE specifically measures calibration**. It compares predicted confidence and actual accuracy

$$\text{ECE}=\sum_m\frac{|B_m|}{|S|}\left|\text{acc}(B_m)-\text{conf}(B_m)\right|$$

**Ensemble gain**

Let metric $M(\cdot)$ be Precision, Recall, or F1.

- Best single model performance:

$$M^*=\max_k M(k)$$

- Ensemble improvement:

$$\Delta M = M(\text{ensemble})-M^*$$

**Agreement and uncertainty**

- **Mean pairwise agreement** among classifiers:

$$\overline{A}=\frac{2}{K(K-1)}\sum_{k<\ell}\frac{1}{|S|}\sum_{i\in S}\mathbb{1}(\hat{y}_{ik}=\hat{y}_{i\ell})$$

- **Disagreement rate**:

$$\text{DisagreeRate}=\frac{1}{|S|}\sum_{i\in S}\mathbb{1}\left(\text{Agree}_i\in (0,1)\right)$$

**Evidence quality (audit-based)**

On an audited subset $S_e \subset S$, human reviewers judged whether evidence spans support the predicted decision.

- **Evidence support rate**

$$\text{EvidSupport}=\frac{\#\{i\in S_e:\text{evidence supports decision}\}}{|S_e|}$$

- **Attribution correctness** (impact linked to EPP):

$$\text{AttrCorrect}=\frac{\#\{i\in S_e:\text{impact attributable to } h\}}{|S_e|}$$

### 2.5 Threshold and weight selection (validation)
The goal is to **choose the best decision threshold and model weights** so that the relevance filter keeps **high precision while maintaining an acceptable recall**. The ensemble system is tested on the reference dataset with **different parameter values** to see which configuration performs best.

Weights $w_k$ and threshold $\theta$ (and optional agreement threshold $\tau$) are tuned on a validation split of $S$ to maximize **precision at a minimum recall**, reflecting the goal of ensuring downstream extraction focuses on truly impact-relevant articles. A typical objective:

$$\max_{\theta,\tau,w} \text{Precision}(\theta,\tau,w)\quad \text{s.t.}\quad \text{Recall}(\theta,\tau,w)\ge r_{\min}$$

where $r_{\min}$ is a predefined recall floor (e.g., 0.70–0.85 depending on tolerance for missed impacts).

**Performance indicators**

- $\text{Precision} @ \text{Recall} \geq r_{\min}$
- AUPRC (area under precision–recall curve) computed by sweeping $\theta$ 
  (optional, recommended for class imbalance)

## 3. Step 4 — Impact extraction and UIG discovery (ensemble + resampling)

Step 4 operates only on the relevant set $R$.

### 3.1 Resampling strategy

To discover robust impact categories while limiting per-batch taxonomy size, we used **bootstrap aggregating (bagging)** with $B$ bootstrap samples $S_b$ of size $n=1000$ drawn **with replacement** from $R$. Overlap arises naturally, enabling stability analysis and uncertainty estimation.

- **Pairwise overlap between samples**

$$\text{Overlap}(b_1,b_2)=\frac{|S_{b_1}\cap S_{b_2}|}{n}$$

- **Resample coverage of $R$**

$$\text{ResampleCoverage}=\frac{\left|\bigcup_{b=1}^B \text{unique}(S_b)\right|}{N_R}$$


### 3.2 Within-batch ensemble extraction: Propose → Align → Adjudicate

#### 3.2.1 Propose (free Unique Impact Group - UIG generation, ≤25 per run)

For each batch $S_b$, $K$ independent extraction runs output:

- ≤25 UIGs per run with name + definition + inclusion/exclusion notes,
- per-article assignments with evidence.

- **UIG count constraint:**

$$\text{UIGCount}_{bk}\le 25$$

- **Assignments per article:** how many impact categories each article receives on average

$$\text{AssignPerArticle}_{bk}=\frac{1}{n}\sum_{i\in S_b} m_{i,bk}$$

- **Evidence completeness:** whether each assigned impact is **supported by a quote from the article**.

$$\text{EvidenceRate}_{bk}=\frac{\#\text{assignments with evidence}}{\#\text{assignments}}$$

#### 3.2.2 Harmonize (canonicalize UIGs across runs)

UIGs were aligned using semantic similarity of definitions (and exemplar spans), producing canonical UIG clusters and mapping run-level UIGs to canonical UIGs. Each UIG contains:
- name
- definition
- example evidence spans

These texts are used to measure **semantic similarity**.
The process typically involves:
- Step 1: Convert UIG definitions into embeddings
- Step 2: Measure similarity
- Step 3: Cluster UIGs

**Indicators**

- Silhouette score (if embedding clustering): The **silhouette score** evaluates how good the clustering is.

$$\text{Silhouette}=\frac{1}{M}\sum_{j=1}^M \frac{b(j)-a(j)}{\max\{a(j),b(j)\}}$$

- Merge rate: The **merge rate** measures how many categories were merged during harmonization.

$$\text{MergeRate}_b=\frac{\sum_k \text{UIGCount}_{bk}-\text{CanonicalUIGCount}_b}{\sum_k \text{UIGCount}_{bk}}$$

#### 3.2.3 Adjudicate (agreement to finalize ≤25 UIG)

For each canonical UIG $u$, compute support score (measures how many extraction runs discovered the same UIG):

$$\text{Support}_{bu}=\frac{\#\{k:u\text{ appears in run }k\}}{K}$$

Retain if $\text{Support}_{bu}\ge\tau_u$ (e.g., 0.6, Meaning **at least 60% of runs must identify the category**), then if $>25$ remain, keep top 25 by:

$$\text{Score}_{bu}=\alpha\cdot \text{Support}_{bu}+(1-\alpha)\cdot \text{Coverage}_{bu}$$

$$\text{Coverage}_{bu}=\frac{\#\{i\in S_b: i\text{ assigned to }u\}}{n}$$

**Indicators**

- UIG consensus rate:

$$\text{ConsensusUIGRate}_b=\frac{\#\{u:\text{Support}_{bu}\ge\tau_u\}}{\text{CanonicalUIGCount}_b}$$

- Information loss from truncation to 25:

$$\text{Loss}_b=1-\frac{|A_b^{25}|}{|A_b|}$$

- Pairwise Jaccard similarity among run UIG sets: **the ration between intersection and union of 2 sets**

$$J_{k\ell}=\frac{|U_{bk}\cap U_{b\ell}|}{|U_{bk}\cup U_{b\ell}|}$$

summarized as $\overline{J}_b$.

### 3.3 Final per-article labeling (constrained judge)

A final "judge" pass assigned each article to UIGs from the finalized ≤25 list (batch-specific), forcing consistency.

**Indicators**

- Uncertain rate:

$$\text{UncertainRate}_b=\frac{\#\{i:\max_u p_{iu}<\gamma\}}{n}$$

- Human audit accuracy on a sampled set $A_b$: exact-match, partial-match, and 
  evidence sufficiency rates.

## 4. Cross-batch synthesis: global UIG taxonomy and stability

Batch UIGs were aligned into a global taxonomy. For each global UIG $g$, compute selection probability:

$$\pi_g=\frac{\#\{b:g\in U_b\}}{B}$$

Retain robust UIGs if $\pi_g\ge \pi^*$ (e.g., 0.5–0.7).

**Indicators**

- Global stability (pairwise Jaccard across batch UIG sets): This metric measures **how similar the UIG sets are between batches**.

$$\overline{J}=\text{mean}_{b_1<b_2}\frac{|U_{b_1}\cap U_{b_2}|}{|U_{b_1}\cup U_{b_2}|}$$

- Selection probability distribution: counts above 0.5 / 0.7 / 0.9. This indicator counts how many categories fall above thresholds:
- Global coverage: This metric measures **how many relevant articles are explained by the discovered categories**.

$$\text{GlobalCoverage}=\frac{\#\{i\in R:\exists g \text{ assigned}\}}{N_R}$$

- Bootstrap percentile CIs for key metrics computed across batches:

$$CI_{95\%}=\left[M^{(0.025)},\,M^{(0.975)}\right]$$
