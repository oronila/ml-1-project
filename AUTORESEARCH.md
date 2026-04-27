# autoresearch

This file describes an experiment loop for improving the wildlife-strike damage
model. The goal is to let an LLM try model ideas, run them, compare scores, and
keep only changes that actually improve validation results.

## Strategy

1.  **Physics First:** Focus on the kinetic energy and physical realities of the strike (SPEED, AC_MASS, bird SIZE).
2.  **Zero Leakage:** Strictly exclude all text from `REMARKS` and `COMMENTS`.
3.  **Imbalanced Handling:** Use native algorithm weighting (sample weights) instead of oversampling.
4.  **Rigorous Evaluation:** Use stratified splits to ensure rare positive classes are accurately represented.

## Notes From Current Work

- **CONSTRAINT:** NEVER use words from `COMMENTS`, `REMARKS`, or any other text column. 
- The Physics-Focused model establishes a new, cleaner baseline for evaluating environmental factors.
- Native weighting helps the model focus on the 6% damaged class without the overfitting risks of manual duplication.

## Latest Results (Non-Text Champion)

| Model | ROC-AUC | PR-AUC | F1 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Final Physics Model** | TBD | TBD | TBD | Current |
| Readable No-Text | 0.90975 | 0.54448 | 0.52477 | Baseline |
