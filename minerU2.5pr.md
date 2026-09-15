3 Data Engine

To address the data deficiencies identified above, we first examine the limitations of existing data pipelines. MinerU2.5 [25] built a data pipeline comprising cluster-based sampling, Iterative Model Inference Consistency (IMIC) hard sample mining, and model-based annotation refinement, but these components operate independently without joint optimization of coverage, informativeness, and accuracy: sampling is not informed by difficulty, annotation refinement applies a uniform strategy regardless of sample difficulty, and hard samples mined by IMIC still face unreliable automatic annotation. Similar limitations exist in PaddleOCR-VL-1.5’s Uncertainty-Aware Cluster Sampling (UACS) [8].

The Data Engine of MinerU2.5-Pro is co-designed around these three dimensions. DDAS expands data coverage through task-aware clustering and mitigates distribution shift (Section 3.1). CMCV performs difficulty stratification on the sampled data via multi-model cross-validation, identifying highly informative samples (Section 3.2). The Annotation Pipeline for Hard Case improves annotation accuracy through render-then-verify iterative correction, with residual samples beyond automatic correction routed to targeted expert annotation to guarantee final quality (Section 3.3). Together, these components form a coarse-to-fine quality progression, enabling simultaneous data scaling (under 10M → 65.5M) and annotation quality improvement. The overall pipeline is illustrated in Figure 2.

## 3.1 Diversity-and-Difficulty-Aware Data Sampling

Training data for document parsing exhibits a typical long-tail distribution problem: high-frequency categories (e.g. standard academic papers, single-column reports) dominate the data pool, while long-tail scenarios such as complex nested tables, dense formula layouts, and unconventional multi-column layouts are severely underrepresented. Existing approaches [25, 8] rely on single-model signals for difficulty estimation, which cannot distinguish model-specific weaknesses from universally hard samples.

We propose Diversity-and-Difficulty-Aware Sampling (DDAS), which jointly optimizes diversity and difficulty at both page and element granularity. Central to DDAS is Cross-Model Consistency Verification (CMCV, detailed in Section 3.2), which leverages prediction agreement among heterogeneous models to classify samples into Easy/Medium/Hard difficulty tiers. The overall pipeline is shown in Figure 3.

### Stage 1: Page-level sampling

Pages in the document pool are represented using 512-dimensional ViT-base features and grouped through K-Means clustering. An initial uniform sample from each cluster is evaluated by page-level CMCV (Section 3.2) to obtain difficulty labels. Based on the difficulty distribution within each cluster, sampling weights are adjusted: clusters dominated by Easy samples receive lower weight, clusters with diverse difficulty distributions receive higher weight, and clusters dominated by invalid content (non-target languages, blank pages, etc.) are filtered out. The adjusted weights are then used to expand sampling from the original document pool and obtain the complete page-level candidate set with CMCV difficulty annotations.

### Stage 2: Element-level sampling

From the page-level candidate set, individual elements (text, formula, and table blocks) are extracted using MinerU2.5 and PaddleOCR-VL layout detection models. Visual features for each element type are extracted and clustered independently, while element-level CMCV assigns difficulty labels. At this point, all four subtasks—layout, text, formula, and table—carry annotations along both the diversity (clustering) and difficulty (CMCV) dimensions.

### Final sampling

Balanced sampling is performed in the joint cluster-difficulty space across all four subtasks. Along the diversity dimension, large clusters are downsampled and small clusters are upsampled to correct long-tail shift. Along the difficulty dimension, Medium and Hard samples are upweighted to increase the informativeness of the training signal. The final output is an SFT training set covering all subtasks while balancing diversity and difficulty.

By coupling clustering with CMCV at both page and element granularity, DDAS allows sampling decisions to account simultaneously for data distribution and training value, maximizing training-signal density while controlling the overall data volume.

## 3.2 Cross-Model Consistency Verification

DDAS uses difficulty labels to determine sampling-weight allocation, while subsequent annotation refinement and expert annotation also require difficulty information to determine how annotation resources should be invested. However, ground truth is unavailable for massive unlabeled data. IMIC in MinerU2.5 [25] and UACS in PaddleOCR-VL-1.5 [8] use output consistency from multiple inferences of a single model as a proxy for difficulty. This approach captures only the epistemic uncertainty of one model and cannot distinguish model-specific blind spots from genuinely hard problems. Model-specific weaknesses can be addressed through cross-model consensus, whereas universally hard samples require further quality refinement or human intervention. This distinction is important when selecting an annotation strategy.

We propose Cross-Model Consistency Verification (CMCV), extending difficulty assessment from single-model introspection to multi-model cross-validation. The premise is that when multiple heterogeneous models generate consistent outputs for a sample, the result is highly likely to be correct; when all models diverge substantially, the sample is genuinely difficult and none of the models can parse it reliably.

Three heterogeneous document parsing models—MinerU2.5 [25], PaddleOCR-VL [6], and Qwen3-VL-30B [43]—are run independently on the candidate data generated by DDAS. Task-specific pairwise consistency metrics are computed: edit distance for text, TEDS for tables, and CDM for formulas. Samples are then assigned to three difficulty tiers according to their consistency patterns. Because MinerU2.5 is the target model being improved, the difficulty taxonomy is anchored on its performance relative to the external models:

- **Easy:** MinerU2.5’s output is highly consistent with at least one external model. Model consensus indicates that the parsing result is reliable, and the output of any model can be used directly as annotation.
- **Medium:** The two external models agree with one another, while MinerU2.5 differs significantly from both. The external consensus can therefore serve as a reliable pseudo-label.
- **Hard:** All three models show significant pairwise disagreement, so no reliable annotation can be obtained through model consensus.

The three categories have different roles in training. Easy data is abundant and reliably annotated, providing the foundation for basic capability development, but the model has largely learned these cases and their marginal training value is limited. Medium data has the greatest training value because it identifies MinerU2.5’s capability gaps relative to peer models, while successful parsing by the external models demonstrates that the samples are learnable and their consensus directly supplies reliable annotations without further correction. Hard data is important for capability breakthroughs, but its annotations are unreliable and must undergo Judge-and-Refine correction or expert annotation (Section 3.3) before being safely used.

CMCV therefore enables rapid difficulty assessment over massive unlabeled data without human annotation, making large-scale data expansion and iteration practical. Because Medium data is scarce but highly valuable, DDAS prioritizes its proportion during sampling. The optimal ratio among the three categories differs by subtask: formula and table recognition are more sensitive to Hard samples, whereas text recognition benefits more from Medium samples.

## 3.3 Annotation Pipeline for Hard Case

CMCV provides reliable automatic annotations for Easy and Medium samples. Hard samples, where all models fail to reach consensus, would introduce annotation noise that could degrade rather than improve model performance if they were used directly for training. Improving annotation quality for these critical samples without depending on large-scale human annotation is the central challenge in moving the Data Engine from filtering to refinement. To address this, a two-stage pipeline is designed: an automated Judge-and-Refine correction loop followed by targeted expert annotation for residual failures.

### Judge-and-Refine Annotation Pipeline

A natural method for improving Hard-sample annotations is to use additional test-time computation through an iterative judge-then-correct mechanism, allowing a model to inspect and refine its own parsing results. However, naive self-reflection has a systematic tendency to accept its own outputs: when asked to evaluate its output, the model often confirms that the result is correct and misses existing errors.

The underlying cause is the asymmetry of cross-modal mapping. Models are strong at generating structured sequences from document images but have difficulty inferring visual appearance from structured sequences. For complex structural mappings such as LaTeX formulas and HTML tables, a model cannot accurately determine how an output sequence will render visually in implicit space, which substantially limits its ability to detect structural errors.

To overcome this limitation, render-then-verify is incorporated into the iterative correction loop. LaTeX formulas are compiled and HTML tables are rendered into images. The original document image and the rendered image are then supplied to the model as paired inputs together with the judge-and-refine prompt. This provides two benefits. First, it completes the missing mapping from structured text to visual layout and reduces the cross-modal reasoning burden. Second, rendering amplifies errors: subtle structural defects in the text domain, such as missing alignment symbols or unclosed tags, become visible anomalies or layout collapse, making them easier to identify through visual comparison.

Based on this design, a visual-comparison-driven Judge-and-Refine iterative correction pipeline is constructed. Qwen3-VL-235B is used as the Judge-Refine model because of its strong multimodal reasoning capability and its independence from the CMCV model pool, which avoids systematic bias in error detection. Multi-round error localization and targeted correction are performed through direct visual comparison between the original document image and the rendered image. After this processing, a subset of extremely complex cases still remains beyond automatic correction, and these samples are sent to the expert annotation workflow.

### Targeted Expert Annotation

For Hard samples that cannot be corrected automatically, expert human annotation is introduced to guarantee final quality. The annotation budget is allocated according to two priority dimensions based on intermediate Judge-and-Refine outputs:

1. **Correction efficiency:** Samples for which the Judge stage has identified errors with high confidence but the Refine stage has failed to correct them receive the highest priority. Annotators only need to perform local corrections at the identified locations, maximizing annotation throughput.
2. **Marginal impact:** Within this pool, priority is further given to subtask categories where the current model is weakest, as determined by CMCV disagreement patterns, maximizing the marginal contribution of the limited annotation budget to overall performance.

Human annotation follows an AI pre-annotation and expert review-and-correction workflow. Gemini 3 Pro is used for pre-annotation because of its strong multimodal reasoning capability and its independence from the CMCV model pool, thereby avoiding data leakage. Automated QA tools are additionally used to maintain annotation consistency. Compared with MinerU2.5’s human annotation process [25], annotation targets move from random sampling to a precisely targeted subset identified through three-stage filtering, substantially improving annotation-resource utilization.

The Data Engine produces a stratified dataset: approximately 65.5M Easy and Medium samples, automatically annotated through CMCV, are used for Stage 1 pre-training; 192K expert-annotated Hard samples are used for Stage 2 fine-tuning and Stage 3 GRPO alignment.
"""

out = "/mnt/data/MinerU2.5Pro_Data_Engine.md"
pypandoc.convert_text(md, "md", format="md", outputfile=out, extra_args=["--standalone"])
print(out)
