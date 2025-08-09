#!/usr/bin/env python
"""
prepare_test_captions_and_generate.py
-------------------------------------
* 입력 JSONL에서 9:1(test)로 분할한 캡션으로 MIDI를 생성
* 생성물은  ▶ eval/generated/<timestamp>/xxx_gen.mid
* 결과 JSONL은 ▶ eval/test_generated_<timestamp>.jsonl
"""

import argparse, jsonlines, pathlib, random, pickle, sys, datetime
from pathlib import Path
from tqdm import tqdm
import torch, hashlib
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# 프로젝트 루트 경로 & 모델
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _model_transformers import Transformer

# ──────────────────────────────────────────────────────────────────────────────
def load_assets(model_ckpt, vocab_pkl, device):
    with open(vocab_pkl, "rb") as f:
        remi_tok = pickle.load(f)
    model = Transformer(len(remi_tok), 768, 8, 512, 18, 1024,
                        False, 8, device=device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device).eval()
    sent_tok = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base",
                                             use_fast=False)
    return model, remi_tok, sent_tok

def generate_midi(cap, out_path, model, remi_tok, sent_tok, device):
    inp = sent_tok(cap, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        ids = model.generate(inp["input_ids"],
                             inp["attention_mask"], max_len=500)[0].tolist()
    remi_tok.decode(ids).dump_midi(out_path)

# ──────────────────────────────────────────────────────────────────────────────
def main():
    cli = argparse.ArgumentParser("Generate MIDI for test captions")
    cli.add_argument("--captions_jsonl", required=True)
    cli.add_argument("--tokenizer_vocab", required=True)
    cli.add_argument("--model_ckpt",      required=True)
    cli.add_argument("--generated_dir",   default=None,
                     help="(옵션) 생성 MIDI 저장 루트")
    cli.add_argument("--output_jsonl",    default=None,
                     help="(옵션) 결과 JSONL 경로")
    cli.add_argument("--device", default="cpu")
    args = cli.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # ─── 기본 경로 자동 설정
    if args.generated_dir is None:
        args.generated_dir = f"{ROOT}/eval/generated/{ts}"
    if args.output_jsonl is None:
        args.output_jsonl  = f"{ROOT}/eval/test_generated_{ts}.jsonl"
    Path(args.generated_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    model, remi_tok, sent_tok = load_assets(args.model_ckpt,
                                            args.tokenizer_vocab,
                                            args.device)

    # ─── JSONL 읽기 & 9:1 split(seed 42)
    with jsonlines.open(args.captions_jsonl) as r:
        raw = list(r)
    random.seed(42)
    random.shuffle(raw)
    raw = raw[int(len(raw) * 0.9):]          # 10 % test

    # ─── 메타정보 정리
    lines, gen_paths = [], []
    for e in raw:
        ref = e.get("reference_path") or e.get("location")
        assert ref, "reference_path/location 필드 필요"
        cid = Path(ref).stem
        out_mid = Path(args.generated_dir) / f"{cid}_gen.mid"
        dup = 2
        while out_mid.exists():
            out_mid = Path(args.generated_dir) / f"{cid}_gen{dup}.mid"
            dup += 1
        lines.append({
            "id": cid,
            "caption": e["caption"],
            "reference_path": ref,
            "midi_path": str(out_mid)
        })
        gen_paths.append(out_mid)

    # ─── MIDI 생성
    for obj, out_mid in tqdm(zip(lines, gen_paths),
                             total=len(lines), desc="generate"):
        if out_mid.is_file():
            continue
        generate_midi(obj["caption"], out_mid,
                      model, remi_tok, sent_tok, args.device)

    # ─── 결과 JSONL 저장
    with jsonlines.open(args.output_jsonl, "w") as w:
        w.write_all(lines)

    print(f"[✓] Generated {len(lines)} test MIDIs → {args.generated_dir}")
    print(f"[✓] JSONL saved            → {args.output_jsonl}")

if __name__ == "__main__":
    main()
