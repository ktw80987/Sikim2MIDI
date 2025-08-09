# import os, json, mido, pandas as pd

# # ────────────────────────────────────────────────────────────────
# # 1.  Δ(semitone)·시간 조건 ← 실측 통계 기반
# # ────────────────────────────────────────────────────────────────
# # • 요성(VB):  A → A+2 → A     (Δt ≤ 0.15s)
# # • 퇴성(DW):  A → A−2~3       (0.05s ≤ Δt ≤ 0.25s)
# # • 전성(SP):  3음 이상 연속 하강, 단계당 ≥2, 전체 ≥6 (0.2s ≤ 총 Δt ≤ 0.6s)

# def detect_sikim_events_note_based(midi_path: str):
#     try:
#         midi = mido.MidiFile(midi_path)
#     except Exception:
#         return {"VB": 0, "DW": 0, "SP": 0, "error": True}

#     events  = {"VB": 0, "DW": 0, "SP": 0, "error": False}
#     notes   = []                         # [(time_sec, pitch), …]

#     for tr in midi.tracks:
#         t = 0
#         for msg in tr:
#             t += mido.tick2second(msg.time, midi.ticks_per_beat, 500000)
#             if msg.type == "note_on" and msg.velocity > 0:
#                 notes.append((t, msg.note))

#     # ───── ① 요성 (VB)  ─────
#     for i in range(len(notes) - 2):
#         t1, p1 = notes[i]
#         t2, p2 = notes[i + 1]
#         t3, p3 = notes[i + 2]
#         if (t3 - t1) <= 0.15 and (p2 - p1) == 2 and p3 == p1:
#             events["VB"] += 1

#     # ───── ② 퇴성 (DW)  ─────
#     for i in range(len(notes) - 1):
#         t1, p1 = notes[i]
#         t2, p2 = notes[i + 1]
#         if 0.05 <= (t2 - t1) <= 0.25 and 2 <= (p1 - p2) <= 3:
#             events["DW"] += 1

#     # ───── ③ 전성 (SP)  ─────
#     for i in range(len(notes) - 2):
#         t_start, p_start = notes[i]
#         t_mid,   p_mid   = notes[i + 1]
#         t_end,   p_end   = notes[i + 2]
#         if (
#             0.2 <= (t_end - t_start) <= 0.6 and
#             (p_start - p_mid) >= 2 and
#             (p_mid   - p_end) >= 2 and
#             (p_start - p_end) >= 6
#         ):
#             events["SP"] += 1

#     return events

# # ────────────────────────────────────────────────────────────────
# # 2.  JSONL → 경로 치환 → 감지 실행
# # ────────────────────────────────────────────────────────────────
# jsonl_path      = "/home/wjg980807/ko-text2midi/eval/test_generated_general.jsonl"
# ROOT_DIR        = "/home/wjg980807/ko-text2midi"
# # prefix_old, prefix_new = "eval/generated/20250622_0748", "eval/generated/generated_general"

# results = []
# with open(jsonl_path, encoding="utf-8") as f:
#     for line in f:
#         try:
#             obj       = json.loads(line)
#             # rel_path  = obj["midi_path"].replace(prefix_old, prefix_new)
#             rel_path=obj["midi_path"]
#             midi_path = os.path.join(ROOT_DIR, rel_path)

#             res = detect_sikim_events_note_based(midi_path)
#             res["filename"] = os.path.basename(midi_path)
#             results.append(res)

#         except (json.JSONDecodeError, KeyError) as e:
#             print("⚠️ JSON/KEY Error:", e)

# df = pd.DataFrame(results).to_csv("sikim_detection_results.csv", index=False)

# # ────────────────────────────────────────────────────────────────
# # 3.  통계 요약 & SSR 계산
# # ────────────────────────────────────────────────────────────────
# df = pd.read_csv("sikim_detection_results.csv")

# total_vb = df["VB"].sum()
# total_dw = df["DW"].sum()
# total_sp = df["SP"].sum()

# valid_df           = df[df["error"] == False]
# valid_df["has_sikim"] = (valid_df[["VB", "DW", "SP"]].sum(axis=1) > 0).astype(int)

# total_valid        = len(valid_df)
# total_with_sikim   = valid_df["has_sikim"].sum()
# SSR                = total_with_sikim / total_valid if total_valid else 0

# print("📊 시김새 탐지 총합")
# print(f"  ✔ VB: {total_vb}, DW: {total_dw}, SP: {total_sp}")
# print(f"  ✔ 시김새 포함 파일: {total_with_sikim} / {total_valid}")
# print(f"  ✔ SSR = {SSR:.4f}  ({SSR*100:.2f}%)")

import os, json, mido, argparse, pandas as pd

# ────────────────────────────────────────────────────────────────
# 1.  커맨드라인 인자 파싱
# ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="ERR-based Sikim evaluator")
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="평가 대상 JSONL 파일 경로"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/ceph_data/wjg980807/k_t2m",
        help="MIDI 경로 기준이 되는 프로젝트 루트"
    )
    return parser.parse_args()

# ────────────────────────────────────────────────────────────────
# 2.  시김새 감지 함수 (동일)
# ────────────────────────────────────────────────────────────────
def detect_sikim_events_note_based(midi_path: str):
    try:
        midi = mido.MidiFile(midi_path)
    except Exception:
        return {"VB": 0, "DW": 0, "SP": 0, "error": True}

    events = {"VB": 0, "DW": 0, "SP": 0, "error": False}
    notes  = []

    for tr in midi.tracks:
        t = 0
        for msg in tr:
            t += mido.tick2second(msg.time, midi.ticks_per_beat, 500000)
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append((t, msg.note))

    # 요성 (VB)
    for i in range(len(notes) - 2):
        t1, p1 = notes[i]
        t2, p2 = notes[i + 1]
        t3, p3 = notes[i + 2]
        if (t3 - t1) <= 0.15 and (p2 - p1) == 2 and p3 == p1:
            events["VB"] += 1

    # 퇴성 (DW)
    for i in range(len(notes) - 1):
        t1, p1 = notes[i]
        t2, p2 = notes[i + 1]
        if 0.05 <= (t2 - t1) <= 0.25 and 2 <= (p1 - p2) <= 3:
            events["DW"] += 1

    # 전성 (SP)
    for i in range(len(notes) - 2):
        t0, p0 = notes[i]
        t1, p1 = notes[i + 1]
        t2, p2 = notes[i + 2]
        if 0.2 <= (t2 - t0) <= 0.6 and (p0 - p1) >= 2 and (p1 - p2) >= 2 and (p0 - p2) >= 6:
            events["SP"] += 1

    events.update({
        "has_VB": int(events["VB"] > 0),
        "has_DW": int(events["DW"] > 0),
        "has_SP": int(events["SP"] > 0),
    })
    return events

# ────────────────────────────────────────────────────────────────
# 3.  메인 로직
# ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    ROOT_DIR  = args.root_dir
    jsonl_path = args.jsonl_path

    results = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            midi_path = os.path.join(ROOT_DIR, obj["midi_path"])

            res = detect_sikim_events_note_based(midi_path)
            res["filename"] = os.path.basename(midi_path)
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("sikim_detection_results.csv", index=False)

    valid_df = df[df["error"] == False]
    total_files = len(valid_df)
    print(f'total_Files:{total_files}')

    # 이벤트 총합
    event_totals = {k: valid_df[k].sum() for k in ["VB", "DW", "SP"]}

    # 파일 단위 ERR
    ERR = {
        "VB": valid_df["has_VB"].sum() / total_files,
        "DW": valid_df["has_DW"].sum() / total_files,
        "SP": valid_df["has_SP"].sum() / total_files,
        "Any": valid_df[["has_VB", "has_DW", "has_SP"]].max(axis=1).mean(),
    }
    print(f'VB num:{valid_df["has_VB"].sum()}')
    print(f'DW num:{valid_df["has_DW"].sum()}')
    print(f'SP num:{valid_df["has_SP"].sum()}')

    #결과 출력
    print("📊 이벤트 총합")
    for k, v in event_totals.items():
        print(f"  • {k}: {v}")
        print(k,v)

    print("\n📊 ERR (파일 단위 비율)")
    for k, v in ERR.items():
        print(f"  • {k}: {v:.4f}")
    
if __name__ == "__main__":
    main()
