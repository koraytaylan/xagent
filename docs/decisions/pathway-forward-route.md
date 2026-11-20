# Forward route gate decision

## Question

Should the forward pathway route (`brain.forward_pathway_enabled`) be on by default? The route arm (forward route on, critic input `encoding`) is compared with a control arm (both routes off) on gates G1 to G4. Unconditional forward credit is reported as information only and enters no gate or verdict.

## Setup

- Commit: `fd7419c7c0c4cc4c4fba4ad7026e1fe0458eccbf`, the output of `git rev-parse HEAD` taken once before any run and passed as `--commit` to every tool call. Every free life and generation run used this commit. The commit precedes any adoption edit; with the verdict below no adoption edit was made.
- Config: `--dump-config` default (SHA-256 `78c12cf9477828b8a5f357f140ac0ae4a23adaebe359fe631765aa7940f00b4a`), copied into arm configs editing only the two `brain` route keys; `diff` confirms the forward arm differs from control in `forward_pathway_enabled` only.

| Arm | `critic_input` | `forward_pathway_enabled` | `governor.tick_budget` | SHA-256 |
|---|---|---|---|---|
| control (free life) | `encoding` | `false` | 1000000 (default) | `78c12cf9477828b8a5f357f140ac0ae4a23adaebe359fe631765aa7940f00b4a` |
| forward (free life) | `encoding` | `true` | 1000000 (default) | `e1f01fc88ff8c381e65be82abacb035a0a185fe2f7c5191f41252482cd539d09` |
| control (generations) | `encoding` | `false` | 40000 | `45616a2f492fd11d6792ee3a4281cd91f212496e19b8fc6f1c43c705fa8084c6` |
| forward (generations) | `encoding` | `true` | 40000 | `0ce9e8e067651cb0ce75509efd42cf73924ff69e9fe65b114653cb5a6a123544` |

- Seeds 5, 6, 7, 8; 10 agents per seed (40 per arm); 300,000 ticks per free life; fresh birth brains. Every run exited 0 on its first attempt and every report lists 10 agents of 300,000 ticks.
- Adapter: every run executed on the software Vulkan adapter `llvmpipe (LLVM 21.1.8, 256 bits)`; no hardware GPU device was reachable from the run environment. Mesa lavapipe is an accepted GPU adapter for these runs (as in the repository's CI); the gates compare control and route arms run on the same adapter, so the paired comparison is adapter-consistent. The routes-off fingerprint replay is a separate check run on the recording host's hardware adapter.
- Gate constants (from the record): `td_discount` 0.97, `td_lambda` 0.9, trace decay 0.873, G1 needs 21 of 40 agents, G2 needs 3 of 4 seeds, G4 window generations 60 to 78.

Commands (repository root; `<arm>` is `control` or `forward`, `S` is 5, 6, 7, 8; scratch under the git-ignored `target/pathway-route/`):

```sh
COMMIT=$(git rev-parse HEAD)
cargo run --release -p xagent-sandbox --bin xagent -- --dump-config > target/pathway-route/default.json
sha256sum target/pathway-route/default.json
# arm configs: copy default.json, edit only brain.critic_input / brain.forward_pathway_enabled
# generation configs: copy each arm config, set governor.tick_budget to 40000

cargo run --release -p xagent-sandbox --bin xagent -- --config target/pathway-route/<arm>.json --seed S \
  --free-life --free-life-agents 10 --free-life-ticks 300000 \
  --free-life-out target/pathway-route/free-life/<arm>-sS.json \
  --free-life-log target/pathway-route/free-life/<arm>-sS.log

cargo run -p xagent-sandbox --bin pathway_route_gate -- free-life --route forward --commit "$COMMIT" \
  --control-metrics target/pathway-route/free-life/control-s{5,6,7,8}.json \
  --route-metrics target/pathway-route/free-life/forward-s{5,6,7,8}.json \
  --route-log target/pathway-route/free-life/forward-s{5,6,7,8}.log \
  --out docs/decisions/pathway-forward-route.json

cargo run --release -p xagent-sandbox --bin xagent -- --config target/pathway-route/gen-<arm>.json --seed S \
  --no-render --db target/pathway-route/gen/<arm>-sS.db --generations 80

cargo run -p xagent-sandbox --bin pathway_route_gate -- generations --record docs/decisions/pathway-forward-route.json \
  --control-db target/pathway-route/gen/control-s{5,6,7,8}.db \
  --route-db target/pathway-route/gen/forward-s{5,6,7,8}.db

cargo run -p xagent-sandbox --bin pathway_route_gate -- verify docs/decisions/pathway-forward-route.json
```

These commands regenerate the record. Tick logs, metrics reports, databases and config dumps are not committed; the JSON record is tool-written and not hand-edited.

## Gates

| Gate | Measure | Value | Result |
|---|---|---|---|
| G1 | agents with conditional credit C > 0 | 38 of 40 (needs 21) | |
| G1 | pooled mean C (SE) | 2.362e-05 (2.881e-06) | pass |
| G2 | pooled sighting meal share, route vs control | 0.6164 vs 0.5973 | |
| G2 | seeds with higher share | 3 of 4 (needs 3) | pass |
| G3 | mean meals per agent, route (SE) vs control (SE) | 172.600 (10.865) vs 165.300 (11.650) | |
| G3 | meals difference (Welch SE), needs >= -SE | 7.300 (15.930) | |
| G3 | mean deaths per agent, route (SE) vs control (SE) | 33.675 (1.230) vs 34.025 (1.343) | |
| G3 | deaths difference (Welch SE), needs <= +SE | -0.350 (1.821) | pass |
| G4 | seeds where route mean composite fitness (generations 60 to 78) >= control | 2 of 4 (needs 4) | fail |

Per seed:

| Seed | Control share | Route share | Higher | Control meals | Route meals | Control deaths | Route deaths | Control fitness 60-78 | Route fitness 60-78 | G4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.5941 | 0.6287 | yes | 158.9 | 182.3 | 33.0 | 32.5 | 0.3024 | 0.2575 | fail |
| 6 | 0.5770 | 0.5869 | yes | 169.1 | 143.2 | 33.3 | 36.8 | 0.2296 | 0.2283 | fail |
| 7 | 0.6229 | 0.6162 | no | 165.5 | 161.6 | 32.1 | 30.9 | 0.2076 | 0.2388 | pass |
| 8 | 0.5973 | 0.6274 | yes | 167.7 | 203.3 | 37.7 | 34.5 | 0.1991 | 0.2608 | pass |

Unconditional forward credit U (information only, in no gate or verdict): pooled mean 1.829e-05 (SE 3.180e-06).

Per agent (from the record's `agents` array):

| Arm | Seed | Agent | Meals | Deaths | Sightings | Ending in meal | Share | C | U |
|---|---|---|---|---|---|---|---|---|---|
| control | 5 | 0 | 259 | 22 | 404 | 232 | 0.5743 |  |  |
| control | 5 | 1 | 214 | 30 | 266 | 190 | 0.7143 |  |  |
| control | 5 | 2 | 198 | 25 | 293 | 180 | 0.6143 |  |  |
| control | 5 | 3 | 187 | 25 | 283 | 166 | 0.5866 |  |  |
| control | 5 | 4 | 111 | 36 | 177 | 100 | 0.5650 |  |  |
| control | 5 | 5 | 220 | 22 | 295 | 200 | 0.6780 |  |  |
| control | 5 | 6 | 49 | 47 | 105 | 44 | 0.4190 |  |  |
| control | 5 | 7 | 122 | 36 | 202 | 113 | 0.5594 |  |  |
| control | 5 | 8 | 176 | 40 | 280 | 155 | 0.5536 |  |  |
| control | 5 | 9 | 53 | 47 | 92 | 44 | 0.4783 |  |  |
| control | 6 | 0 | 193 | 33 | 305 | 180 | 0.5902 |  |  |
| control | 6 | 1 | 347 | 31 | 436 | 301 | 0.6904 |  |  |
| control | 6 | 2 | 95 | 41 | 154 | 87 | 0.5649 |  |  |
| control | 6 | 3 | 110 | 39 | 210 | 101 | 0.4810 |  |  |
| control | 6 | 4 | 198 | 25 | 324 | 185 | 0.5710 |  |  |
| control | 6 | 5 | 114 | 41 | 182 | 101 | 0.5549 |  |  |
| control | 6 | 6 | 84 | 42 | 161 | 68 | 0.4224 |  |  |
| control | 6 | 7 | 174 | 31 | 245 | 156 | 0.6367 |  |  |
| control | 6 | 8 | 211 | 22 | 345 | 192 | 0.5565 |  |  |
| control | 6 | 9 | 165 | 28 | 274 | 150 | 0.5474 |  |  |
| control | 7 | 0 | 134 | 38 | 195 | 123 | 0.6308 |  |  |
| control | 7 | 1 | 293 | 26 | 413 | 265 | 0.6416 |  |  |
| control | 7 | 2 | 182 | 26 | 271 | 160 | 0.5904 |  |  |
| control | 7 | 3 | 167 | 32 | 244 | 158 | 0.6475 |  |  |
| control | 7 | 4 | 68 | 42 | 128 | 61 | 0.4766 |  |  |
| control | 7 | 5 | 159 | 31 | 233 | 138 | 0.5923 |  |  |
| control | 7 | 6 | 80 | 44 | 129 | 69 | 0.5349 |  |  |
| control | 7 | 7 | 226 | 28 | 285 | 202 | 0.7088 |  |  |
| control | 7 | 8 | 146 | 28 | 224 | 131 | 0.5848 |  |  |
| control | 7 | 9 | 200 | 26 | 267 | 181 | 0.6779 |  |  |
| control | 8 | 0 | 293 | 22 | 375 | 259 | 0.6907 |  |  |
| control | 8 | 1 | 146 | 39 | 222 | 132 | 0.5946 |  |  |
| control | 8 | 2 | 140 | 38 | 220 | 129 | 0.5864 |  |  |
| control | 8 | 3 | 309 | 30 | 428 | 280 | 0.6542 |  |  |
| control | 8 | 4 | 163 | 48 | 271 | 151 | 0.5572 |  |  |
| control | 8 | 5 | 82 | 43 | 177 | 73 | 0.4124 |  |  |
| control | 8 | 6 | 84 | 50 | 150 | 76 | 0.5067 |  |  |
| control | 8 | 7 | 220 | 25 | 335 | 200 | 0.5970 |  |  |
| control | 8 | 8 | 41 | 49 | 97 | 39 | 0.4021 |  |  |
| control | 8 | 9 | 199 | 33 | 280 | 187 | 0.6679 |  |  |
| route | 5 | 0 | 134 | 52 | 232 | 123 | 0.5302 | 5.289e-05 | 5.042e-05 |
| route | 5 | 1 | 139 | 33 | 215 | 131 | 0.6093 | 3.603e-05 | 1.885e-05 |
| route | 5 | 2 | 307 | 24 | 394 | 268 | 0.6802 | 3.730e-05 | 2.377e-05 |
| route | 5 | 3 | 209 | 27 | 296 | 196 | 0.6622 | 2.514e-05 | 3.394e-05 |
| route | 5 | 4 | 133 | 33 | 185 | 107 | 0.5784 | -2.542e-05 | -4.442e-05 |
| route | 5 | 5 | 220 | 23 | 318 | 196 | 0.6164 | 2.026e-06 | -1.143e-06 |
| route | 5 | 6 | 179 | 36 | 261 | 165 | 0.6322 | 2.846e-05 | 2.085e-05 |
| route | 5 | 7 | 268 | 20 | 331 | 237 | 0.7160 | 2.060e-05 | 2.312e-05 |
| route | 5 | 8 | 118 | 34 | 190 | 110 | 0.5789 | 2.763e-05 | 2.053e-05 |
| route | 5 | 9 | 116 | 43 | 180 | 103 | 0.5722 | 6.439e-05 | 3.510e-05 |
| route | 6 | 0 | 105 | 36 | 163 | 96 | 0.5890 | 3.257e-05 | 2.052e-05 |
| route | 6 | 1 | 97 | 38 | 149 | 85 | 0.5705 | 1.911e-05 | 2.548e-05 |
| route | 6 | 2 | 173 | 30 | 243 | 154 | 0.6337 | 3.552e-05 | 4.656e-05 |
| route | 6 | 3 | 126 | 38 | 206 | 114 | 0.5534 | 2.351e-05 | 1.553e-05 |
| route | 6 | 4 | 239 | 33 | 356 | 219 | 0.6152 | 1.550e-05 | 4.412e-06 |
| route | 6 | 5 | 103 | 51 | 156 | 80 | 0.5128 | -1.478e-05 | -3.538e-05 |
| route | 6 | 6 | 144 | 32 | 220 | 126 | 0.5727 | 1.809e-05 | 2.380e-06 |
| route | 6 | 7 | 195 | 39 | 275 | 178 | 0.6473 | 3.927e-05 | 4.919e-05 |
| route | 6 | 8 | 139 | 32 | 221 | 125 | 0.5656 | 8.845e-06 | 1.493e-05 |
| route | 6 | 9 | 111 | 39 | 187 | 100 | 0.5348 | 3.598e-05 | 2.081e-05 |
| route | 7 | 0 | 120 | 34 | 186 | 112 | 0.6022 | 6.093e-05 | 3.325e-05 |
| route | 7 | 1 | 188 | 27 | 262 | 179 | 0.6832 | 1.673e-05 | 2.801e-05 |
| route | 7 | 2 | 166 | 28 | 257 | 155 | 0.6031 | 7.666e-06 | -9.944e-07 |
| route | 7 | 3 | 164 | 30 | 240 | 150 | 0.6250 | 1.703e-05 | 7.105e-06 |
| route | 7 | 4 | 266 | 18 | 367 | 246 | 0.6703 | 1.427e-05 | 8.261e-06 |
| route | 7 | 5 | 93 | 37 | 161 | 87 | 0.5404 | 1.615e-05 | 8.958e-06 |
| route | 7 | 6 | 63 | 44 | 106 | 62 | 0.5849 | 4.621e-05 | 1.951e-05 |
| route | 7 | 7 | 149 | 39 | 263 | 136 | 0.5171 | 4.795e-05 | 4.749e-05 |
| route | 7 | 8 | 158 | 24 | 241 | 145 | 0.6017 | 1.899e-05 | 1.647e-05 |
| route | 7 | 9 | 249 | 28 | 348 | 226 | 0.6494 | 4.670e-06 | -4.114e-06 |
| route | 8 | 0 | 264 | 40 | 371 | 236 | 0.6361 | 1.740e-05 | 1.115e-05 |
| route | 8 | 1 | 76 | 45 | 143 | 71 | 0.4965 | 2.328e-05 | 4.227e-06 |
| route | 8 | 2 | 221 | 27 | 335 | 207 | 0.6179 | 2.044e-05 | 2.624e-05 |
| route | 8 | 3 | 217 | 38 | 326 | 195 | 0.5982 | 8.152e-06 | 1.646e-05 |
| route | 8 | 4 | 259 | 23 | 348 | 232 | 0.6667 | 2.999e-05 | 3.891e-05 |
| route | 8 | 5 | 378 | 30 | 451 | 326 | 0.7228 | 2.271e-05 | 1.079e-05 |
| route | 8 | 6 | 172 | 28 | 252 | 159 | 0.6310 | 4.806e-05 | 5.371e-05 |
| route | 8 | 7 | 156 | 38 | 238 | 138 | 0.5798 | 3.948e-06 | 1.488e-05 |
| route | 8 | 8 | 96 | 41 | 164 | 84 | 0.5122 | 2.159e-05 | 3.319e-05 |
| route | 8 | 9 | 194 | 35 | 287 | 181 | 0.6307 | 1.604e-05 | 1.275e-05 |

## Verdict

`fail`. G1, G2 and G3 pass; the failing gate is `G4`: the route's mean composite fitness over generations 60 to 78 is below the control's in seeds 5 and 6.

## Adoption

Not adopted. `DEFAULT_FORWARD_PATHWAY_ENABLED` stays `false` and `crates/xagent-shared/src/config.rs` is unchanged; an adoption edit would change only that default constant and its doc comment.
