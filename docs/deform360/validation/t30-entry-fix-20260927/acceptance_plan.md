# T30 checkpoint / T31 acceptance preparation

2026-09-27. Preparation only; no inference or new training job submitted.

## Select checkpoint after the running allocation ends

Read Slurm terminal status and observer report; select the last fully written readable epoch checkpoint, never a partial file. Record epoch, iter, size, SHA256. A partial-budget checkpoint must be labeled intermediate; TIMEOUT or exploratory acceptance is not T30 full-budget PASS. Latest observed complete checkpoint is epoch22/iter13200, pending post-job confirmation.

## CPU integrity gate

Read checkpoint with map_location=cpu. Verify model and Adam tensors finite, expected optimizer mapping and steps consistent with metadata,12FP64Normalizer statistics finite. Record config hash, code commit and authorized max_seq diff. Keep checkpoint and tensor data server-only. Preserve Adam/counters for possible future resume.

## T31 execution gate (not run)

Confirm whether user wants intermediate exploratory review or completion of61epochs first. Do not choose checkpoint by test performance. Read actual continuous code and test entry before execution. Use canonical initial source113/local0 SH0; controller candidate7mm; ConfigA cameras/calibration/gravity unchanged. Roll local0..193/source113..306 densely. Audit every coarse template boundary: later state must originate from earlier online predictions, not T27 future coarse starts, reconstructed PLY or GT Gaussian injection. Corresponding controller is allowed; RGB/mask GT only for comparison. Record identity/count/finite, template-state provenance and full source/local mapping. Stop at first schema/nonfinite/geometry-path discrepancy; no automatic algorithm changes.

## Review deliverables

Checkpoint provenance; continuous prediction provenance; representative RGB/render/mask comparisons; train/test frame separation; finite/drift/coverage observations. T31 PASS means valid continuous state transfer tosource306, not prediction quality or full training completion. T32 metrics and further tuning require their own scope. Any test-informed decision to extend training is exploratory development, not untouched final test evaluation.

Await clarification on whether to finish the full61epoch budget or review a complete intermediate checkpoint. No job cancellation, training resume or T31 execution performed as part of this preparation.
