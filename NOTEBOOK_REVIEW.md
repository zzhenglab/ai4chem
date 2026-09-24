# Notebook review

Reviewed source revision `8f9eb71` on 2026-09-21. Scope: all 26 source notebooks (lectures 01–25 plus lecture-0-solution), comprising 776 executable cells in six `.ipynb` files and twenty MyST Markdown files. Generated notebooks under `book/_build` were treated as copies, not additional lessons. The six generated `.ipynb` counterparts match their source cell contents.

This is a review, not a patch to the course. Findings below distinguish failures from improvements to experimental design. Locations in `.ipynb` files refer to raw JSON line numbers at the reviewed revision.

**Fix before teaching or relying on the results**

1. **Lecture 15: Bayesian optimization ignores its acquisition score.** At `book/lecture-15.md:1989`, `np.lexsort((-ei, cand_idx))` makes the candidate index the primary sort key. Thus the model-based branch of `suggest_experiments` picks the smallest unused row IDs, regardless of expected improvement. A local probe with indices `[1,3,7]` and EI `[.01,.1,.9]` returned `[1,3,7]`, whereas the intended ranking is `[7,3,1]`. Use `np.lexsort((cand_idx, -ei))`. Also preserve objective weights when resuming experiments: lines 2811–2817 replace the user's weights with equal weights.

2. **Lectures 21–22: image evaluation leaks related images across train and test.** The splits at `book/lecture-21.md:438` and `book/lecture-22.md:409` operate on individual tiles/rotations. The supplied 240 images represent only **10 original photographs: eight crystal and two no-crystal**, with 80 tiles and three rotations per tile. Using sorted filenames and the demonstrated 20%/seed-42 split logic, all ten test-photo origins also appeared in training. In the CNN split, 45/48 test images had another rotation of their tile in training; in the stratified CLIP split, 48/48 did. Exact CNN membership can vary with directory order, but the splitting unit is wrong in either case. Split by original photograph before augmentation. Collect more independent photographs, especially negatives, before interpreting the score as generalization to new experiments.

3. **Lectures 10, 13, 20, and 23 depend on missing notebook state.** Fresh-namespace probes fail immediately in lecture 10 (`book/lecture-10.ipynb:444`) and lecture 13 (`book/lecture-13.ipynb:697`) because `pd` is undefined, and lecture 20 (`book/lecture-20.ipynb:110`) because `client` is undefined. These notebooks also omit other required imports/configuration; adding only the first missing name is insufficient. Lecture 23 omits RDKit imports before `Chem`/`Draw` at `book/lecture-23.ipynb:475`, and sklearn imports before its classifier at line 2189. Add complete setup cells and verify each notebook from a restarted kernel. Missing API credentials should produce an explicit optional-demo status.

4. **Lecture 10: the solubility model is tested without training.** `book/lecture-10.ipynb:1405` calls `trainer.test(mpnn_sol, test_loader)`, but the source contains no `trainer.fit` call to train the newly constructed solubility model. Restoring imports alone does not reproduce the displayed trained-model results. Add training with train/validation loaders, select the checkpoint using validation data, and explicitly evaluate that checkpoint. Regenerate saved output after a clean run.

5. **Lecture 9: the test set selects the model checkpoint.** `book/lecture-09.md:1862` passes `test_graphs` as validation data to a wrapper that chooses its best checkpoint, then reports accuracy on those same graphs at line 1865. The optional grid-search example repeats this at line 1892. Reserve separate validation and test splits. Similar threshold/depth/regularization sweeps in lectures 5, 7, and 8 should use validation data for choosing settings, then score the final model once on the test set.

6. **Lecture 8: the log-solubility weight demo actually fits toxicity labels.** The classifier overwrites `Xtr/ytr` at `book/lecture-08.md:1035`. The later section explicitly promises a log-solubility regressor, but fits `reg_inspect` on those binary labels at line 1119. Keep task-specific names, or inspect the already fitted solubility pipeline. This currently runs without an exception while teaching the wrong interpretation.

7. **Lecture 25: several advertised agent paths cannot complete.** `book/lecture-25.ipynb:3259` calls `web_search(query, max_results=...)`, but its helper at line 2924 accepts only `query`. Both voice implementations call undefined `record_audio` (lines 3821 and 3997). The voice tool list advertises `generate_tecan_csv` at line 3702, but the dispatcher has no matching handler. The custom-code replies at lines 3900/4084 also use `custom_tool_output`; the required type is `custom_tool_call_output`, confirmed in the installed OpenAI SDK and [official OpenAI documentation](https://developers.openai.com/api/docs/guides/async-tool-calling). Use one registry for schemas and handlers, validate arguments, and test every dispatch path with fake model responses before a live demo.

8. **Lecture 24: extraction errors become false scientific negatives.** At `book/lecture-24.ipynb:5113`, every figure label defaults to `"N"`. Invalid JSON, absent keys, or invalid values leave those defaults in place, and line 5125 returns a normal-looking annotation row. A fake malformed model response reproduced all-negative output. Preserve `unknown`/`error` separately from absence; validate the full schema, keep the raw response, and retry or review failures. This matters more than prompt wording because it changes the resulting dataset silently.

**Correct the comparisons and scientific interpretation**

9. **Lecture 14 compares unequal experiment budgets.** The BO run has eight initial observations and thirty additional evaluations (`book/lecture-14.md:908`, line 927). The comparison at lines 983–1008 nevertheless labels BO results for budgets up to 140, reusing its 38-observation maximum for every larger budget. Grid and random methods get the full budget. Limit the plot to observed budgets or run every method to the maximum, counting initialization equally. Create one noise RNG per run: line 993 currently resets the generator for every baseline observation, repeating the same noise draw. Repeat runs across seeds and plot uncertainty.

10. **Lecture 5 compares regularization across different train/test splits.** The learning-curve loop overwrites split variables at `book/lecture-05.md:407`, leaving seed 4 active. The Lasso/Ridge comparisons around lines 511–573 reuse the seed-42 linear baseline metrics. On the supplied data, the displayed comparison gives linear R² 0.874103 versus Lasso 0.878852; comparing both on seed 42 gives Lasso 0.874090 instead. Preserve fixed splits and refit every comparator on them. Put scaling inside the CV pipeline when comparing penalized linear models.

11. **Lecture 20's optimization oracle takes the best of hidden conditions.** `book/lecture-20.ipynb:1567` discards concentration and solvent, then `score_yield` at line 1694 returns the maximum yield for a temperature/time pair. The supplied data contains 20 concentration/solvent combinations per pair. At 25 °C/12 h, yields range from 0.15 to 0.44; the oracle returns 0.44 as if this were one specified experiment. Fix the omitted factors or have the planner propose all four conditions. Otherwise describe the objective explicitly as a best-case envelope over hidden conditions. Also correct the follow-up question's role from `assistant` to `user` at line 1034.

12. **Lecture 18 evaluates samples already seen during representation training.** Chemistry pretraining uses all rows at `book/lecture-18.md:1129`, before the classifier split at line 1315. The spectra example does likewise at lines 1576/1615/1639 before the split at line 1686. This is a transductive evaluation: labels may be held out, but the inputs were seen. For unseen-sample claims, split first and train the encoder, scaler, and PCA using training data only. Alternatively, name and explain the transductive protocol. Compare against raw-feature and PCA baselines on the same split.

13. **Lecture 17's calibration holdout was already used for training.** The toy classifier trains on all positive/unlabeled data at `book/lecture-17.md:327`, then selects its supposed positive holdout at line 333. The MOF example repeats this at lines 557/569, as does the solution at lines 728/734. The reserved-training variables are never used to fit the classifier. Split positives first, exclude calibration examples from classifier/scaler fitting, and estimate the calibration constant on those held-out positives. Separate in-sample MOF ranking/AUC diagnostics from performance on unseen data. This becomes particularly consequential for the proposed random-forest replacement, which can memorize training positives.

14. **Lecture 16's interactive Thompson sampler does not sample the posterior it teaches.** At `book/lecture-16.md:2169`, the JavaScript demo forms ratios of sums of uniform draws, rather than Gamma draws. It even uses identical sampler loop counts for Beta(1,1) and Beta(2,1), so the first success cannot change the sampled posterior. Although the comments call it an approximation, it loses a central behavior of the method. Use a tested Beta sampler, or exact Gamma draws from sums of exponential variables for the integer parameters here. The later Python `rng.beta` example already implements the correct distribution.

**Additional concrete repairs**

| Source location | Finding and repair |
|---|---|
| `book/lecture-01.md:879` | The glucose solution never assigns its addition back to `M_glucose`. It prints 0.0 g/mol. Use `+=`; expected value is about 180.156. |
| `book/lecture-02.md:538` | The URL CSV option creates `df`, but the next section uses `df_csv`. All alternative loading paths should define the same variable. |
| `book/lecture-02.md:156` | A separate-cell `plt.savefig` can save a blank figure after the inline backend closes the plotted figure. Save in the plotting cell or retain `fig` and call `fig.savefig`. Reproduced with an all-white PNG. |
| `book/lecture-03.md:66` | Installing RDKit in the exception branch never imports `Chem`/`Draw` afterward. Move required imports after the install check. |
| `book/lecture-03.md:132` | The supposedly equivalent isopropanol SMILES are different C4H10O compounds: 2-butanol and 2-methoxypropane. Use `CC(O)C` and `CC(C)O`, and assert equal canonical SMILES. |
| `book/lecture-03.md:887` | The solution prints a ring container as a count and truncates aromatic bond orders 1.5 to 1. Use the ring count and preserve floating bond orders or named bond types. |
| `book/lecture-04.md:421` | The CIR widget calls undefined `cir_by_name`, although `cir_get` exists. Route the widget through the implemented helper. |
| `book/lecture-09.md:1493` | The molecular visualization prints `edge_attr`, but the returned variable is named `ea`. The broad exception handler masks the failure and skips drawing. Use `ea`. |
| `book/lecture-12.md:734` | DBSCAN silhouette scoring includes noise label -1 as a cluster. Score non-noise points after checking the cluster/sample counts, and report the excluded noise fraction. |
| `book/lecture-14.md:960` | Plot marker size uses the last coordinate of `U` (concentration), while the title says yield. Use observed `y`. |
| `book/lecture-15.md:366` | Hypervolume shading connects points diagonally instead of drawing the dominated staircase. A single point (0.9,0.8) should dominate a rectangle of area 0.72, not the displayed triangle of area 0.36. |
| `book/lecture-15.md:2658` | Upload handlers use the ipywidgets 7 dictionary interface (`value.values()`); local ipywidgets 8.1.7 supplies a tuple. Support the installed interface and test upload/resume. |
| `book/lecture-16.md:532` | Canvas and document key listeners both toggle pause for one bubbling key event. Keep one listener or stop propagation. The training loop at line 610 also updates state alongside the ordinary loop at line 500; give a single loop ownership of transitions. Similar patterns recur in the other games. |
| `book/lecture-18.md:1188` | The advertised no-PyTorch path uses undefined `Z_chem_small`. The spectra PCA fallback also never assigns `enc`, used at line 1774. Require PyTorch clearly or complete both fallbacks. |
| `book/lecture-19.md:365` | Missing PyTorch is caught earlier as optional, but this section uses it unconditionally. Require it for the neural-network section or skip that section explicitly. |
| `book/lecture-24.ipynb:1982` | Reusing double quotes inside the f-string fails on Python ≤3.11, including this workspace's Python 3.8. Use single quotes for dictionary keys or declare Python ≥3.12. |

The `multi_class` argument used at `book/lecture-18.md:1693` and `book/lecture-22.md:420` is incompatible with scikit-learn 1.8. Remove it for the binary classifier; use an explicit one-vs-rest wrapper when that behavior is intended for multiclass. This is a version-conditional issue: the local scikit-learn 1.3.2 still accepts it. [Official migration notes](https://scikit-learn.org/1.8/whats_new/v1.5.html), [1.8 API signature](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).

**Course-wide reproducibility**

- The root README's `pip install -r requirements.txt` cannot find the supplied file, which is `book/requirements.txt`. That file lists only Jupyter Book, matplotlib, and NumPy; many lessons also need pandas, scikit-learn, RDKit, PyTorch, and other packages. Provide a tested Python environment with separate optional groups for chemistry, deep learning, and API demos.
- Prefer bundled `_data` files, with a cached download fallback for Colab. Existing build logs record GitHub HTTP 429 failures for lectures 6/7 and a 600-second download timeout for lecture 22. These are historical failures, not a claim that GitHub is currently unavailable.
- Execution exclusions in `book/_config.yml:14` use names such as `lecture-6` and `lecture-11`, which do not match `lecture-06.md` and `lecture-11.md` in the installed execution engine. The six explicit `.ipynb` exclusions do match. Fix the patterns and separate a fast automated execution check from the static publishing build. A successful cached/excluded build is not evidence that every source notebook runs.
- Add a clean-kernel check for runnable examples. Keep deliberate student exercise blanks out of that check or supply a solution execution path. Preserve selected teaching outputs only after they have been regenerated from source.
- For optional API exercises, use environment/Colab secrets, cached sample responses, and fake tool results for testing. Bound agent iterations and separate recording/device interactions from basic notebook execution. For lecture 25's generated-code demo, prefer reviewed callable tools or an isolated runner: `exec(..., {})` still permits ordinary Python imports and file access.

**Coverage and the most useful demo improvement for each notebook**

| Notebook | Assessment / next improvement |
|---|---|
| 0 solution | No major independent failure confirmed. Make input paths self-contained and align the optional generated CSV filename with the exercise. |
| 01 Python | Fix the glucose answer; add simple expected-value checks to worked solutions. |
| 02 pandas/plotting | Fix CSV alternatives and figure saving; overlay individual measurements when showing distributions from only three replicates. |
| 03 RDKit | Fix setup and chemical-identity examples; verify names, molecular formulas, and canonical identities in demonstrations. Show atom indices before graph edits. |
| 04 identifiers | Fix the CIR handler and mismatched solution numbering; use one resolver interface with explicit input/output types. |
| 05 regression/classification | Fix comparison splits; scale penalized models and tune thresholds on validation data. |
| 06 cross-validation | The train-only CV/grid-search structure is sound. Standardize features before interpreting coefficient magnitudes; retain the pipeline inside CV. |
| 07 trees/forests | No unconditional failure found with installed dependencies. Use identical splits across comparisons and validation data for depth/leaf sweeps. |
| 08 neural networks | Fix wrong-target weight inspection. Hold learning rate and early stopping fixed when demonstrating architecture effects; worse training loss alone is not overfitting. |
| 09 graph networks | Fix checkpoint/test leakage; compare to the descriptor baseline on the identical split. |
| 10 property/reaction prediction | Restore setup and the omitted training step, then regenerate outputs. |
| 11 dimension reduction | Use original-space neighbors and quantitative neighborhood checks alongside attractive 2D plots. |
| 12 self-supervised learning | Fix DBSCAN noise scoring. Compare clustering in original feature space with clustering projected embeddings to expose projection artifacts. |
| 13 molecular generation | Restore setup; compare sparse-fingerprint reconstruction to an all-zero baseline using on-bit precision/recall and Tanimoto similarity. Report generation validity, uniqueness, and novelty. |
| 14 Bayesian optimization | Correct budget/noise comparisons; repeat matched-budget runs and show variability. |
| 15 multi-objective BO | Fix acquisition ranking, preserve weights across resume, and test the upload workflow. |
| 16 reinforcement learning | Correct the Thompson sampler and overlapping UI update/event paths; evaluate fixed policies against matched-budget baselines. |
| 17 PU learning | Correct holdout calibration; assess sensitivity to the label-selection assumptions and use explicit top-k ranking to avoid ties from clipped probabilities. |
| 18 contrastive learning | Correct or disclose transductive evaluation, finish fallbacks, and add raw/PCA baselines. |
| 19 transformers | Core masking/pooling implementation appears coherent. Treat epoch-monitored scores as validation and add a final untouched test set and baseline. |
| 20 LLMs | Restore setup, correct message role, and define an experimental oracle with all controlled factors. |
| 21 computer vision | Split original photos before augmentation; expand independent negative examples. |
| 22 vision-language | Use the same held-out photo groups for zero-shot and linear-probe comparisons; cache data/embeddings and batch encoding. |
| 23 agents | Restore imports and test every tool handler with fixed example inputs before invoking a model. |
| 24 literature mining | Keep failure/unknown separate from negative. Add hand-labeled examples with precision/recall and retain page-level evidence for extracted claims. |
| 25 self-driving labs | Repair tool dispatch and recording setup; add bounded execution and validate generated experiment tables before export. |

**Validation and limits**

Every source notebook received code/prose review, and all 776 executable cells were checked for parseability using IPython transformation under Python 3.8.10. The sole syntax failure found by that pass is the version-dependent lecture-24 f-string. Targeted execution covered all 45 lecture-1 cells, 37/38 lecture-3 cells with the bundled spreadsheet replacing the download and a remote image display skipped, fresh-namespace failures, the CSV/plot/widget issues, acquisition ranking, image split overlap, and the lecture-5 numerical comparison. Fake responses exercised selected agent and extraction failures. No paid API calls, package installations, full GPU training runs, or full-course builds were performed.

The old lecture-8 `ConvergenceWarning` NameError and lecture-9 `add_notes_from` error logs refer to code already corrected in current sources; they are not current findings. Passing a small probe or finding no major issue during review is not a claim that a full notebook has passed end-to-end execution.
