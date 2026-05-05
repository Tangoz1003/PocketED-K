# CKD Audit

This update rewrites the MC-MED `chronic_kidney_disease` flag using explicit diagnosis/history text,
while keeping the existing MIMIC-ED CKD flag unchanged.

## Why this was needed

- Original MC-MED CKD positives: 38/23,503 (0.16%)
- Updated MC-MED CKD positives: 1,769/23,503 (7.53%)
- MIMIC-ED CKD positives (unchanged): 3,474/13,612 (25.52%)

## Evidence of under-ascertainment in the original MC-MED flag

- MC-MED eGFR <60: 5,214/23,503 (22.2%)
- MC-MED eGFR <45: 3,013/23,503 (12.8%)
- MC-MED eGFR <30: 1,533/23,503 (6.5%)
- Original MC-MED CKD only flagged 38 patients total, which is not clinically plausible against the eGFR distribution.

## Reasonable interpretation

- Updated MC-MED CKD is diagnosis/history-based, not lab-defined CKD.
- A reasonable diagnosis-based CKD prevalence range for MC-MED is roughly 7% to 15%.
- The updated explicit-text estimate is 7.5%, which is conservative and avoids relabeling transient renal dysfunction as CKD.
- MIMIC-ED CKD at 25.5% is broadly consistent with its eGFR distribution:
  - eGFR <45: 3,348/13,612 (24.6%)
  - eGFR <60: 5,071/13,612 (37.3%)

## MC-MED CKD text pattern used

`\bckd\b|chronic kidney disease|chronic renal disease|chronic renal insuff|chronic renal failure|end stage renal|end-stage renal|esrd|dialysis dependent|hemodialysis|peritoneal dialysis`
