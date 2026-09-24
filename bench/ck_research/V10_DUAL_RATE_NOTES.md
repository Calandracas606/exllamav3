# v10 dual-rate experiments (worktree 24b04d07, HEAD 6419741)

Baseline (shipped, pure-C fdot2, VOPD emission order): 14 dual / 18 unpaired
= 60.9% dual rate. VGPRs 93. verify ALL PASS, qgemm 80/80, 610 GB/s @45MB b4.

Levers measured (both NEGATIVE, reverted):
- kt-unroll x2 (#pragma unroll 2): 32 dual / 64 unpaired = 50.0% (worse).
  VGPRs 117. Widening the GCNCreateVOPD pairing window does NOT raise the
  rate - the binding constraint is register-bank placement of the pack
  operands, not pool size.
- VGPR-x via shfl round-trip (defeat SGPR promotion of warp-uniform x): 8
  dual / 24 unpaired = 40.0% (worse). SGPR-x constant-bus reads pair FINE;
  the shuffles only scramble scheduling.

Conclusion: ~61% is the measured ceiling of the pure-C architecture on
clang/ROCm 7.14 gfx1100. 100% dual rate requires controlling register
banks, which only the asm route provides (v9: 16/16 duals but 2.2x slower
from the asm barrier + pinned-acc occupancy). Perf at 61% duals already
matches v6 (610 GB/s) because the shape is memory-latency-bound, not
MAC-bound.

## Audit-3 addition: src0 register pins TESTED (the genuinely untried lever)
Register-asm bindings on the 8 pack operands per rh (Delta10 pairs
v60/70, v62/72, v64/74, v66/76) — bindings HONORED (ISA shows the pinned
VGPRs feeding dot2s: v60/v62/v64/v66/v76 all appear as dot2 srcs), dual
rate UNCHANGED: 14/18 = 60.9%, VGPRs 93 identical. Bank placement of src0
is therefore NOT the binding constraint either. All three levers now
measured negative: unroll-x2 (50%), shfl-VGPR-x (40%), src0 pins (61%,
no change). The unpaired 18 are scheduler-adjacency limited inside
GCNCreateVOPD on this toolchain. Reverted (pins add 8 v_mov/rh for zero
gain).

## Audit-4 additions
1. LOOP-LOCATION PROOF: all 46 dot2 slots sit in the steady-state loop body
   (dot2 span ends exactly at the s_cbranch back-edge; no prologue/epilogue
   contamination). 28/46 = 60.9% IS the hot-loop rate.
2. INTERLEAVED compute/emit (packs built inline in the fdot2 args, decls
   removed; tested on live _hip.cuh + full-flag hipcc -S): 14/18 = 60.9%
   IDENTICAL - the scheduler already interleaves decode and MAC chains
   freely; source-level ordering of the pack computation does not matter.
Four levers now measured, all confirming GCNCreateVOPD adjacency as the
binding constraint: unroll-x2 50%, shfl-VGPR-x 40%, src0 pins 61%
(unchanged), interleave 61% (unchanged).

## Audit-5: -mllvm knob sweep (DEMONSTRATED, not asserted)
llvm hidden-help on this toolchain (AMD LLVM 23.0.0git / ROCm 7.14) DOES
expose VOPD/scheduler options. Tested via full-flag hipcc -S with
section-bounded counting — every one compiles and every one yields the
IDENTICAL 14 dual / 18 unpaired = 60.9%:
  -mllvm -amdgpu-enable-vopd                    60.9%
  -mllvm -amdgpu-igrouplp-exact-solver          60.9%
  -mllvm -amdgpu-igrouplp-exact-solver +cost-heur 60.9%
  -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule 60.9%
  -mllvm -amdgpu-disable-unclustered-high-rp-reschedule     60.9%
  -mllvm -amdgpu-enable-pre-ra-optimizations    60.9%
  -O2 (vs -O3)                                  60.9%
The pairing outcome is invariant to scheduler tuning AND optimization
level: the 18 unpaired are constrained inside GCNCreateVOPD itself
(pair eligibility of the scheduled instruction stream), not by any
reachable upstream knob. rh-loop unroll: already fully unrolled at
compile time (rh is a constant index in each stanza) — no variants exist
to sweep. Six levers total now measured; 60.9% stands as the demonstrated
result for this architecture on this toolchain.
