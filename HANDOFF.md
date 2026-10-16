# HANDOFF

更新：2026-10-16 · by Claude · HEAD `4117a42`+#96 本地 commit
<!-- 規則見 AGENTS.md「Handoff」。上限 150 行；過期內容移到 docs/archive/HANDOFF-YYYY-MM.md。 -->

## ▶ 下一步（只放一件，必須帶 #Issue）
#87 修補 lesson_id content address 未涵蓋 evidence_origin/retrieval_eligible/confidence 的缺口（需 lesson_id v2 遷移）。依據：2026-10-11 invariant audit 開出的四個缺口中，#84/#85/#86 已由 PR #90 處理，#87 是唯一明確排除在 PR #90 之外、尚無人處理的一項。注意：涉及 lesson_id 格式變更，屬 persistent schema / migration，動手前先確認是否需升級授權。

## 進行中
- PR #90（Refs #84、#85、#86）producer-only invariants 改在 consumer 端強制：mergeable=clean，test 與 Nexus Core issue run/completion 最新一輪皆綠（Core issue 綁 #85）。等 owner 審查合併；合併後跑 `nexus-certify issue-check` 再逐一關閉 #84/#85/#86（PR 未用 closing keyword）。

- #96 程式端已完成（本地分支 fix/lessons-abstract-only-96，未推送）：fix_recipe 獨立欄位、檢索 row 無 recipe、新增 `injection_text`、prompt 改要求程序層級 lesson。量測（校正題組、≥3 次重跑、配對 uplift 與退步）尚未做，屬實驗環節。

## 待確認（沒有 Issue 號）
- 無

## 卡住 / 注意
- Memory A/B 2026-10 已結案（#95，`docs/experiments/2026-10-memory-ab.md`）：九組設定都 adoption=DEFER，最終 M5 Qwen3.6-35B ×3 在校正題組上 off 36/72 vs on 33/72（p=0.77）。下一個實驗方向依文件排序，第一項是 #96（只注入抽象 lesson、不注入 fix recipe）；任何新方向都要用同一方法（校正難度、≥3 次重跑、看配對 uplift 與退步）驗收。
- #78 [P2] 重複的 flock、`_text`、hash 格式與 status helper，尚未開工；hash 格式（`sha256:` 前綴 vs 純 hex）不一致可能影響未來比對，與 #87 一起動時要留意。
- #41 為 `[HYBRID_CANARY]` 研究探針（no implementation），不是工程待辦；各 issue 上的 `NEXUS-HYBRID-REPLICATION-*` 留言屬 RESEARCH_OBSERVATION_ONLY，不構成授權。
- Learning 不擁有 route/planner/workforce 決策；adoption 相關修補不得擴權（見 AGENTS.md）。
