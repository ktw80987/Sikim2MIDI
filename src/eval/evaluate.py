#!/usr/bin/env python
"""
Text2MIDI 평가 도구 - COSIATEC 압축률 포함
==================================================
• COSIATEC 기반 압축률 (CR) - 음악 패턴 발견 및 압축
• CLAP 기반 의미적 유사도 (optional)
• Tempo Bin 지표 (TB / TBT) – Text2MIDI 논문 기준
• Key 지표  (CK / CKD)  – Text2MIDI 논문 기준

COSIATEC 구현
--------------
David Meredith의 COSIATEC 알고리즘을 기반으로 한 
기하학적 패턴 발견 및 압축률 계산
"""

import argparse
import os
import pickle
import warnings
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

# Audio / MIDI
import pretty_midi
from music21 import converter, tempo as m21tempo, meter, key as m21key

# Torch + CLAP
import torch
from torch.nn.functional import cosine_similarity

# ────────────────────────────────────────────────────────────
# COSIATEC 구현 클래스들
# ────────────────────────────────────────────────────────────

class Point:
    """2D 점 (시작시간, 피치)"""
    def __init__(self, onset: float, pitch: int):
        self.onset = onset
        self.pitch = pitch
    
    def __eq__(self, other):
        return abs(self.onset - other.onset) < 0.01 and self.pitch == other.pitch
    
    def __hash__(self):
        return hash((round(self.onset, 2), self.pitch))
    
    def __repr__(self):
        return f"Point({self.onset:.2f}, {self.pitch})"

class Vector:
    """변위 벡터 (시간차, 피치차)"""
    def __init__(self, dt: float, dp: int):
        self.dt = dt
        self.dp = dp
    
    def __eq__(self, other):
        return abs(self.dt - other.dt) < 0.01 and self.dp == other.dp
    
    def __hash__(self):
        return hash((round(self.dt, 2), self.dp))
    
    def __repr__(self):
        return f"Vector({self.dt:.2f}, {self.dp})"

class TEC:
    """Translational Equivalence Class"""
    def __init__(self, pattern: List[Point], vectors: List[Vector]):
        self.pattern = pattern
        self.vectors = vectors
        self.covered_set = self._compute_covered_set()
    
    def _compute_covered_set(self) -> Set[Point]:
        """TEC가 커버하는 모든 점들 계산"""
        covered = set()
        for vector in self.vectors:
            for point in self.pattern:
                new_point = Point(
                    point.onset + vector.dt,
                    point.pitch + vector.dp
                )
                covered.add(new_point)
        return covered
    
    def compression_ratio(self) -> float:
        """압축률 계산: |covered_set| / (|pattern| + |vectors|)"""
        if len(self.pattern) + len(self.vectors) == 0:
            return 0.0
        return len(self.covered_set) / (len(self.pattern) + len(self.vectors))
    
    def compactness(self) -> float:
        """압축성: |pattern| / |covered_set|"""
        if len(self.covered_set) == 0:
            return 0.0
        return len(self.pattern) / len(self.covered_set)

class SIA:
    """Structure Induction Algorithm"""
    
    @staticmethod
    def compute_vectors(point_set: List[Point]) -> List[Vector]:
        """모든 점 쌍 간의 벡터 계산"""
        vectors = []
        for i, p1 in enumerate(point_set):
            for j, p2 in enumerate(point_set):
                if i != j:
                    vector = Vector(p2.onset - p1.onset, p2.pitch - p1.pitch)
                    vectors.append(vector)
        return vectors

class SIATEC:
    """SIATEC 알고리즘 - 최적화된 버전"""
    
    @staticmethod
    def find_tecs(point_set: List[Point], min_pattern_size: int = 2, max_pattern_size: int = 5) -> List[TEC]:
        """주어진 점 집합에서 TEC 찾기 (크기 제한)"""
        tecs = []
        
        # 점 집합이 너무 크면 샘플링
        if len(point_set) > 50:
            # 시간순으로 정렬하고 대표적인 점들만 선택
            point_set.sort(key=lambda p: p.onset)
            step = len(point_set) // 50
            point_set = point_set[::step]
        
        # 패턴 크기 제한
        max_size = min(max_pattern_size, len(point_set))
        
        # 작은 패턴부터 시작
        for pattern_size in range(min_pattern_size, max_size + 1):
            # 조합 수 제한
            combinations = list(itertools.combinations(point_set, pattern_size))
            if len(combinations) > 100:  # 너무 많으면 샘플링
                import random
                combinations = random.sample(combinations, 100)
            
            for pattern_points in combinations:
                pattern = list(pattern_points)
                
                # 이 패턴을 다른 곳으로 변위시키는 벡터들 찾기
                vectors = SIATEC._find_translating_vectors(pattern, point_set)
                
                if len(vectors) >= 2:  # 최소 2개 이상의 발생
                    tec = TEC(pattern, vectors)
                    tecs.append(tec)
                    
                    # TEC 수 제한
                    if len(tecs) > 20:
                        return tecs
        
        return tecs
    
    @staticmethod
    def _find_translating_vectors(pattern: List[Point], point_set: List[Point]) -> List[Vector]:
        """패턴을 다른 위치로 변위시키는 벡터들 찾기"""
        if not pattern:
            return []
        
        # 첫 번째 패턴 점을 기준점으로 사용
        base_point = pattern[0]
        vectors = []
        
        for point in point_set:
            # 이 점이 base_point의 변위된 위치라고 가정
            vector = Vector(point.onset - base_point.onset, point.pitch - base_point.pitch)
            
            # 이 벡터로 전체 패턴을 변위시켰을 때 모든 점이 point_set에 있는지 확인
            if SIATEC._pattern_exists_at_vector(pattern, point_set, vector):
                vectors.append(vector)
        
        return vectors
    
    @staticmethod
    def _pattern_exists_at_vector(pattern: List[Point], point_set: List[Point], vector: Vector) -> bool:
        """벡터만큼 변위된 패턴이 point_set에 존재하는지 확인"""
        point_set_tuples = {(round(p.onset, 2), p.pitch) for p in point_set}
        
        for point in pattern:
            translated_point = (
                round(point.onset + vector.dt, 2),
                point.pitch + vector.dp
            )
            if translated_point not in point_set_tuples:
                return False
        return True

class COSIATEC:
    """COSIATEC 알고리즘 - 최적화된 버전"""
    
    @staticmethod
    def compress(point_set: List[Point], min_compression_ratio: float = 1.2, max_iterations: int = 10) -> List[TEC]:
        """COSIATEC 압축 알고리즘 (최적화)"""
        if len(point_set) < 2:
            return []
            
        remaining_points = set(point_set)
        selected_tecs = []
        iteration = 0
        
        while remaining_points and iteration < max_iterations:
            # 남은 점들에서 SIATEC 실행
            remaining_list = list(remaining_points)
            if len(remaining_list) < 2:
                break
                
            tecs = SIATEC.find_tecs(remaining_list, min_pattern_size=2, max_pattern_size=4)
            
            if not tecs:
                break
            
            # 최고의 TEC 선택 (압축률 기준)
            best_tec = max(tecs, key=lambda t: t.compression_ratio())
            
            # 최소 압축률 체크
            if best_tec.compression_ratio() < min_compression_ratio:
                break
            
            selected_tecs.append(best_tec)
            
            # 선택된 TEC의 커버 영역 제거
            remaining_points -= best_tec.covered_set
            iteration += 1
        
        return selected_tecs

# ────────────────────────────────────────────────────────────
# Music21 to Point Set 변환
# ────────────────────────────────────────────────────────────

def midi_to_point_set(midi_path: str) -> List[Point]:
    """MIDI 파일을 점 집합으로 변환"""
    try:
        score = converter.parse(midi_path)
        points = []
        
        for note in score.flatten().notes:
            if hasattr(note, 'pitch'):  # 단일 음표
                point = Point(float(note.offset), note.pitch.midi)
                points.append(point)
            elif hasattr(note, 'pitches'):  # 화음
                for pitch in note.pitches:
                    point = Point(float(note.offset), pitch.midi)
                    points.append(point)
        
        return points
    except Exception as e:
        warnings.warn(f"MIDI to point set conversion failed for {midi_path}: {e}")
        return []

def cosiatec_compression_ratio(midi_path: str) -> float:
    """COSIATEC을 사용한 압축률 계산 (최적화)"""
    try:
        point_set = midi_to_point_set(midi_path)
        
        if len(point_set) < 2:
            return 1.0
        
        # 점 집합이 너무 크면 더 적극적으로 샘플링
        if len(point_set) > 100:
            # 시간 기준으로 정렬하고 균등하게 샘플링
            point_set.sort(key=lambda p: p.onset)
            step = len(point_set) // 50
            point_set = point_set[::max(1, step)]
        
        # COSIATEC 압축 실행 (타임아웃 없이 빠른 버전)
        tecs = COSIATEC.compress(point_set, min_compression_ratio=1.1, max_iterations=5)
        
        if not tecs:
            return 1.0
        
        # 전체 압축률 계산
        total_original_size = len(point_set)
        total_compressed_size = sum(len(tec.pattern) + len(tec.vectors) for tec in tecs)
        
        if total_compressed_size == 0:
            return 1.0
        
        compression_ratio = total_original_size / total_compressed_size
        return max(1.0, compression_ratio)  # 최소값 1.0
        
    except Exception as e:
        warnings.warn(f"COSIATEC compression failed for {midi_path}: {e}")
        return 1.0

# ────────────────────────────────────────────────────────────
# 기존 Text2MIDI 평가 코드 (수정)
# ────────────────────────────────────────────────────────────

# Tempo‑bin constants
BIN_BORDERS = [40, 60, 70, 90, 110, 140, 160, 210]  # Text2MIDI paper

def bpm_to_bin(bpm: float) -> int:
    for i, border in enumerate(BIN_BORDERS):
        if bpm < border:
            return i
    return len(BIN_BORDERS)

# Key helper for relative major/minor
MAJOR_TO_REL_MINOR = 9   # +9 semitones
MINOR_TO_REL_MAJOR = 3   # +3 semitones

def is_relative(gen_key: Tuple[int, str], ref_key: Tuple[int, str]) -> bool:
    pc_ref, mode_ref = ref_key
    pc_gen, mode_gen = gen_key
    if mode_ref == "major" and mode_gen == "minor":
        return pc_gen == (pc_ref + MAJOR_TO_REL_MINOR) % 12
    if mode_ref == "minor" and mode_gen == "major":
        return pc_gen == (pc_ref + MINOR_TO_REL_MAJOR) % 12
    return False

def midi_to_audio(midi_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Quick FluidSynth render via pretty_midi; fallback sine."""
    try:
        return pretty_midi.PrettyMIDI(midi_path).fluidsynth(fs=sample_rate)
    except Exception:
        warnings.warn(f"FluidSynth unavailable → sine fallback for {midi_path}")
        pm = pretty_midi.PrettyMIDI(midi_path)
        return pm.synthesize(sample_rate=sample_rate)

# def compute_clap_scores(caps: List[str], audio_paths: List[str], ckpt: str,
#                         device: str, batch: int) -> List[float]:
#     from laion_clap import CLAP
#     model = CLAP(version="music_audioset", use_cuda=device.startswith("cuda"))
#     model.load_ckpt(ckpt)
#     sims: List[float] = []
#     for start in tqdm(range(0, len(caps), batch), desc="CLAP"):
#         sub_caps = caps[start:start+batch]
#         wavs = [midi_to_audio(p) for p in audio_paths[start:start+batch]]
#         L = max(len(w) for w in wavs)
#         wavs = [np.pad(w, (0, L-len(w))) for w in wavs]
#         with torch.no_grad():
#             t_emb = model.get_text_embedding(sub_caps)
#             a_emb = model.get_audio_embedding_from_data(wavs, use_tensor=False)
#             sims.extend(cosine_similarity(t_emb, a_emb).cpu().numpy())
#     return sims

def m21_parse(path: str):
    try:
        return converter.parse(path)
    except Exception:
        warnings.warn(f"music21 parse failed for {path}")
        raise

def estimate_tempo_m21(path: str) -> float:
    score = m21_parse(path)
    mm_marks = score.recurse().getElementsByClass(m21tempo.MetronomeMark)
    if not mm_marks:
        return 120.0
    bpms = [mm.getQuarterBPM() for mm in mm_marks if mm.getQuarterBPM()]
    return float(np.mean(bpms)) if bpms else 120.0

def estimate_key_m21(path: str) -> Tuple[int, str]:
    score = m21_parse(path)
    k = score.analyze("key")
    return k.tonic.pitchClass, ("major" if k.mode == "major" else "minor")

def load_jsonl(path: str):
    import jsonlines
    ids, caps, gen, ref = [], [], [], []
    with jsonlines.open(path) as r:
        for o in r:
            ids.append(o["id"])
            caps.append(o["caption"])
            gen.append(o["midi_path"])
            ref.append(o.get("reference_path"))
    return ids, caps, gen, ref

class DummyTokenizer:
    def encode(self, midi_path: str):
        pm = pretty_midi.PrettyMIDI(midi_path)
        return [f"{n.pitch}_{n.velocity}_{int(n.end-n.start)}" for inst in pm.instruments for n in inst.notes]

def load_tokenizer(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
        print("tokenizer load성공")
    except Exception as e:
        warnings.warn("Tokenizer load failed → DummyTokenizer used\n"+str(e))
        return DummyTokenizer()

def compression_ratio_tokens(tokens: List[str]) -> float:
    return len(set(tokens))/len(tokens) if tokens else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions_file", required=True)
    p.add_argument("--tokenizer_vocab", required=True, help="Path to tokenizer vocabulary")
    p.add_argument("--clap_ckpt", help="Path to CLAP checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_csv", default="evaluation_results_cosiatec.csv")
    p.add_argument("--use_cosiatec", action="store_true", 
                   help="Use COSIATEC for compression ratio (default: token-based)")
    a = p.parse_args()

    ids, caps, gen_p, ref_p = load_jsonl(a.captions_file)

    # CLAP scores (optional)
    clap = [None]*len(ids)
    # if a.clap_ckpt:
    #     clap = compute_clap_scores(caps, gen_p, a.clap_ckpt, a.device, a.batch_size)

    # Compression Ratio
    print("Computing Compression Ratios...")
    if a.use_cosiatec:
        print("Using COSIATEC algorithm...")
        cr = [cosiatec_compression_ratio(pth) for pth in tqdm(gen_p, desc="COSIATEC CR")]
    else:
        print("Using token-based compression ratio...")
        tokenizer = load_tokenizer(a.tokenizer_vocab)
        cr = [compression_ratio_tokens(tokenizer.encode(pth)) for pth in tqdm(gen_p, desc="Token CR")]

    # Tempo and Key metrics
    tb=tbt=ck=ckd=[None]*len(ids)
    tb=list(tb);tbt=list(tbt);ck=list(ck);ckd=list(ckd)
    
    if any(ref_p):
        for i,(g,r) in enumerate(tqdm(zip(gen_p, ref_p), total=len(ids), desc="TB/CK")):
            if not r or not os.path.exists(r):
                continue
            
            try:
                g_tempo,r_tempo=estimate_tempo_m21(g),estimate_tempo_m21(r)
                g_bin,r_bin=bpm_to_bin(g_tempo),bpm_to_bin(r_tempo)
                tb[i]=int(g_bin==r_bin)
                tbt[i]=int(abs(g_bin-r_bin)<=1)
                
                g_key,r_key=estimate_key_m21(g),estimate_key_m21(r)
                ck[i]=int(g_key==r_key)
                ckd[i]=int(g_key==r_key or is_relative(g_key,r_key))
            except Exception as e:
                warnings.warn(f"Evaluation failed for {g}: {e}")
                continue

    # Results
    df=pd.DataFrame({
        "id":ids,
        "CLAP":clap,
        "CR":cr,
        "TB":tb,
        "TBT":tbt,
        "CK":ck,
        "CKD":ckd
    })
    df.to_csv(a.output_csv,index=False)

    print("\n===== Aggregate Metrics =====")
    print("CLAP mean :", df["CLAP"].mean(skipna=True) if df["CLAP"].notna().any() else "– (skipped)")
    print("CR   mean :", df["CR"].mean())
    if any(ref_p):
        print("TB   mean :", df["TB"].mean())
        print("TBT  mean :", df["TBT"].mean())
        print("CK   mean :", df["CK"].mean())
        print("CKD  mean :", df["CKD"].mean())
    else:
        print("TB/CK metrics skipped (no reference_path provided)")

if __name__ == "__main__":
    main()