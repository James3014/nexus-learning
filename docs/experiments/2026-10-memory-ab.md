# Memory A/B 實驗紀錄（2026-10）

狀態：結論已定，`adoption = DEFER`。本文件是 2026-10-08 至 2026-10-12 的實驗紀錄與後續方向，
不是任何採用授權。

## 結論（一句話）

把過去修 bug 的經驗蒸餾成 lesson，在重試時注入 prompt，**在我們測過的所有模型與題組上都沒有可量測的幫助**；
最好的條件下（M5 Qwen3.6-35B、校正過難度的 v5 題組、三次重跑）給 lesson 反而少答對 3 題，差距在隨機範圍內。
系統每一輪都正確判定 `DEFER`，沒有發生錯誤採用。

## 實驗設計

- Runner：Nexus-new `scripts/eval/run_memory_ab_v1.py`。同一組題目跑兩個 arm：
  `memory_off`（不檢索 lesson）與 `memory_on`（先跑 off arm 反思出 lesson，再在 on arm 檢索注入）。
- 題目：每題一個壞掉的模組加一個會失敗的 repro；`-a` / `-b` 兩個變體共用同一個陷阱、換不同情境，
  讓 `-a` 學到的 lesson 有機會幫上 `-b`。
- 判定：`nexus_learning.adoption.build_adoption_from_scorecard`，需要至少 3 組「off 失敗、on 成功」配對且淨 uplift > 0（#67/#68）。
- 狀態隔離：每輪獨立 state root，避免汙染 repo 的 `.nexus/`。

## 結果

| 輪次 | 模型 | 題組 | off 成功 | on 成功 | 配對 uplift | 退步 | 判定 |
|---|---|---|---|---|---|---|---|
| 1–4 | qwen2.5-coder 7B | v2（12 題） | 4 | 4 | 0 | 0 | DEFER |
| 5 | qwen2.5-coder 7B | v2 | 4 | 4 | 1 | 1 | DEFER |
| 6 | 7B 規劃 + 14B 修補 | v3（24 題） | 14 | 13 | 0 | 1 | DEFER |
| 7 | 7B + 14B，加 fix recipe 與語意檢索 | v3 | 14 | 14 | 1 | 1 | DEFER |
| 8 | M5 Qwen3.6-35B-A3B | v3 | 20 | 22 | 2 | 0 | DEFER |
| 8 重跑 | 同上 | v3 | 22 | 22 | 0 | 0 | DEFER |
| 9（3 次） | M5 Qwen3.6-35B-A3B | v5（校正至約 50%） | 36/72 | 33/72 | — | — | DEFER |

第 9 組逐題：5 題給 lesson 後變好、7 題變差、12 題不變；符號檢定 p = 0.77。
on arm 有 13 次成功歸因到 lesson，但歸因的題目整體並沒有比較會答對。
變差最明顯的是 `clock_rate_limit-b`、`remainder_allocation-b`：off 3/3，on 1/3。

注意：第 8 輪的 +2 在重跑時消失（off arm 自己也答對了），M5 模型的單次結果有約 ±3 題的起伏。
本機 qwen2.5-coder 在溫度 0 下 off arm 完全可重現；M5 模型不行，所以必須多次重跑。

## 過程中修掉的斷點（基礎設施現在是可用的）

1. Nexus-new closure bridge 從未提供 episode qualification，所有 episode 都是 UNQUALIFIED、所有 lesson 都是 simulated。
2. 失敗的嘗試是 PARKED 且沒有 verifier receipt，失敗經驗反思不出 lesson（#63/#64）。
3. 反思 judge 需要 `format: json` 才穩定。
4. 記憶軌跡在 `_finalize_run` 才掛上，重試 prompt 從來看不到 lesson。
5. 真實修補走一般重試迴圈而不是語意重試，lesson 注入掛錯地方。
6. 關鍵字檢索打不中同類題，lesson 太抽象（#93/#94、Nexus-new #1785）。
7. Patch 片段與 fix recipe 的 prompt-injection 防護（#94）、模型 HTTP 只允許 localhost（Nexus-new #1807）。

## 與外部研究對照

我們的觀察和 2025–2026 的研究一致：

- **具體軌跡會造成負遷移，抽象洞見才可遷移。** Memory Transfer Learning（arXiv 2604.14004）在 6 個 coding benchmark 上發現
  high-level insights 可泛化、low-level traces 常造成 negative transfer；跨領域平均只 +3.7%，主要來自驗證流程這類 meta 知識。
  我們第 7 輪起加入的「具體 fix recipe」正好是低層細節，與第 9 組的退步模式相符。
- **負遷移特別傷害本來就難的題。** A Study of Experience Reuse in LLM Agents（arXiv 2604.27003）指出 cross-task retrieval
  disproportionately harms hard cases，且要同時量 forward transfer 與 forgetting。
- **經驗跟隨與錯誤傳播。** How Memory Management Impacts LLM Agents（ACL 2026，arXiv 2505.16067）：agent 會模仿相似的舊紀錄；
  全部加入記憶會持續變差，只有嚴格 evaluator 篩選加入、並配合刪除，才有約 +10% 的長期改善。
- **選對才有用。** SWE-ContextBench（arXiv 2602.08316）：正確摘要並選中的經驗能提升解題率，未篩選或選錯的 context 效益有限甚至為負。
  它的相關題對是從真實 issue/PR 依賴關係找出來的，不是人工合成。
- **檢索時過濾前置條件。** RPMS（arXiv 2603.17831）在單次嘗試內先檢查每筆記憶的前置條件是否仍成立，再放進 prompt。
- **另一條路：用成功軌跡訓練。** SWE-Gym（ICML 2025）用 rejection sampling fine-tuning：只保留通過驗證的軌跡做微調，
  32B 模型 491 條軌跡就讓 SWE-Bench Verified +13.6 個百分點；7B 自我改進到 SWE-Bench Lite 約 10%。

## 後續方向（依建議順序）

1. **lesson 改回抽象、程序性的洞見，關掉具體 fix recipe 注入。** 研究與第 9 組結果都指向具體修法會帶偏模型。
   保留 recipe 欄位供人看，但不注入重試 prompt。
2. **嚴格篩選加入與刪除。** 只把「被歸因且最終成功」的 lesson 留在可檢索集合；被歸因卻失敗的 lesson 降權或淘汰
   （nexus-learning 已有 `lesson_disposition` 的 reinforce / contradict / retire，可接上）。
3. **檢索閘門。** 只在 off-path 第一次失敗後、且語意相似度高於門檻時才注入；注入前檢查 `avoid_when` 與目前題目的前置條件。
4. **改用真實的相關題對評估。** 依 SWE-ContextBench 的做法，從 Nexus 自己的 issue/PR 歷史挑出真正相關的修補對，
   取代合成題；報告時分開列「本來就會」與「本來不會」的題目，並同時量退步。
5. **長期：用通過驗證的修補軌跡做 rejection-sampling 微調。** 這是另一種「從經驗學習」，需要訓練資源，屬於新題目，需要 owner 決定。

每一項都必須用本實驗的方法驗收：校正難度到約 50%、至少 3 次重跑、看配對 uplift 與退步，而不是看單輪總數。

## 資料位置

- 每輪 artifacts：Nexus-new `artifacts/runtime/memory_ab_v1/`（未追蹤，本機）。
- 題組：Nexus-new `scripts/eval/memory_ab_tasks_v1`、`_v2`、`_v3`（已合併）；`_v4`、`_v5`（本機，未提交）。
- 相關 PR：nexus-learning #55 #57 #60 #64 #66 #68 #94；nexus-runtime #89 #91；Nexus-new #1637 #1638 #1641 #1672 #1804 #1805 #1807。

## 參考資料

- Memory Transfer Learning: How Memories are Transferred Across Domains in Coding Agents — https://arxiv.org/abs/2604.14004
- When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents — https://arxiv.org/abs/2604.27003
- How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior — https://arxiv.org/abs/2505.16067
- SWE Context Bench: A Benchmark for Context Learning in Coding — https://arxiv.org/abs/2602.08316
- RPMS: Rule-Augmented Memory Synergy — https://arxiv.org/abs/2603.17831
- Training Software Engineering Agents and Verifiers with SWE-Gym — https://arxiv.org/abs/2412.21139
- SWE-MeM: Learning Adaptive Memory Management for Long-Horizon Coding Agents — https://arxiv.org/abs/2606.28434
