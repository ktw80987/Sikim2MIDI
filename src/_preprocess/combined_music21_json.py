import pandas as pd
import numpy as np
import re

def preprocess_music_data(
    commu_meta_path, 
    gugak_midi_features_path,
    gugak_labels_path
):
    """
    음악 데이터 전처리 함수
    
    Parameters:
    -----------
    commu_meta_path : str
        commu_meta.csv 파일 경로
    gugak_midi_features_path : str
        국악MIDI 특징 추출 CSV 파일 경로
    gugak_labels_path : str
        국악 라벨 CSV 파일 경로 (all_labels.csv)
        
    Returns:
    --------
    dict
        전처리된 데이터셋을 포함하는 사전
    """
    # CSV 파일 읽기
    commu_meta = pd.read_csv(commu_meta_path)
    gugak_midi_features = pd.read_csv(gugak_midi_features_path)
    gugak_labels = pd.read_csv(gugak_labels_path)
    
    # 'Unnamed: 0' 열이 있으면 이름을 '' (빈 문자열)로 변경
    if 'Unnamed: 0' in commu_meta.columns:
        commu_meta = commu_meta.rename(columns={'Unnamed: 0': ''})
    
    # music_src_nm 기반 매핑 사전 생성
    music_src_to_info = create_music_src_mapping(gugak_labels)
    
    # Step 1: commu_meta에서 chord_progressions 열 제거
    if 'chord_progressions' in commu_meta.columns:
        commu_meta = commu_meta.drop('chord_progressions', axis=1)
    
    # Step 2: 국악 MIDI 특징에서 chord_progressions 열 제거
    if 'chord_progressions' in gugak_midi_features.columns:
        gugak_midi_features = gugak_midi_features.drop('chord_progressions', axis=1)
    
    # 국악 MIDI 특징 데이터 전처리 - commu_meta 형식에 맞게
    processed_gugak = preprocess_gugak_features(
        gugak_midi_features,
        music_src_to_info
    )
    
    # 결과 반환
    return {
        'commu_meta': commu_meta,
        'processed_gugak': processed_gugak
    }

def create_music_src_mapping(gugak_labels):
    """
    music_src_nm을 기준으로 instrument_name, gukak_style, split_data를 매핑하는 사전 생성
    """
    mapping = {}
    for _, row in gugak_labels.iterrows():
        music_src = row['music_src_nm']
        instrument = row['instrument_name'].lower().replace(' ', '_') if not pd.isna(row['instrument_name']) else None
        style = row['gukak_style'] if not pd.isna(row['gukak_style']) else None
        split = row['split_data'] if not pd.isna(row['split_data']) else 'train'  # 기본값은 'train'
        
        if music_src not in mapping:
            mapping[music_src] = {
                'instrument': instrument,
                'gukak_style': style,
                'split_data': split
            }
    
    return mapping

def create_track_role_mapping(instrument_names):
    """
    Image 2 기반 track_role 매핑 사전 생성
    """
    # 이미지의 매핑 정보를 기반으로 한 사전
    role_mapping = {
        '대금': 'main_melody',
        '피리': 'main_melody',
        '해금': 'main_melody',
      
        '단소': 'sub_melody',
        '소금': 'sub_melody',
        '가야금': 'sub_melody',

        '태평소': 'riff',
        '양금': 'riff',

        '훈': 'pad',
        '편종': 'pad',
        '편경': 'pad',
        
        '아쟁': 'bass',
        '거문고': 'bass',

        '장구': 'accompaniment'
    }
    
    # 입력된 악기 이름에 대한 매핑 결과 반환
    result = {}
    for inst in instrument_names:
        if inst in role_mapping:
            result[inst] = role_mapping[inst]
        else:
            # 기본값으로 main_melody 설정 (또는 다른 적절한 기본값)
            result[inst] = 'main_melody'
    
    return result

def preprocess_gugak_features(
    data, 
    music_src_to_info
):
    """
    국악 MIDI 특징 데이터를 commu_meta.csv 형식에 맞게 전처리
    """
    # 결과를 저장할 새 데이터프레임 생성
    result = pd.DataFrame()
    
    # 인덱스 설정
    result[''] = range(len(data))
    
    # key_signature를 audio_key 형식으로 변환
    result['audio_key'] = data['key_signature'].apply(convert_key_signature)
    
    # pitch_range 매핑
    result['pitch_range'] = [
        map_pitch_range(min_pitch, max_pitch)
        for min_pitch, max_pitch in zip(data['pitch_min'], data['pitch_max'])
    ]
    
    # 기본 열 복사
    result['num_measures'] = data['num_measures']
    result['bpm'] = data['bpm'].round().astype(int)  # BPM을 정수로 변환
    result['time_signature'] = data['time_signature']
    result['min_velocity'] = data['min_velocity']
    result['max_velocity'] = data['max_velocity']
    
    # music_src_nm 열이 있는지 확인하고 없으면 id를 사용
    music_src_column = 'music_src_nm' if 'music_src_nm' in data.columns else 'id'
    
    # 악기, 장르, 트랙 역할, split_data 매핑
    instruments = []
    genres = []
    split_datas = []  # split_data를 저장할 리스트 추가
    
    for _, row in data.iterrows():
        music_src = row[music_src_column]
        info = music_src_to_info.get(music_src, {})
        
        instrument = info.get('instrument', 'default_instrument')
        genre = info.get('gukak_style', None)
        split_data = info.get('split_data', 'train')  # all_labels.csv의 split_data 값 사용
        
        # 악기가 없으면 instruments 열에서 추출
        if instrument is None and 'instruments' in row:
            instrument = process_instruments(row['instruments'])
        
        instruments.append(instrument)
        genres.append(genre)
        split_datas.append(split_data)  # split_data 추가
    
    result['inst'] = instruments
    result['genre'] = genres
    
    # track_role 매핑
    unique_instruments = set(filter(None, instruments))
    track_role_mapping = create_track_role_mapping(unique_instruments)
    result['track_role'] = [track_role_mapping.get(inst, 'main_melody') if inst else 'main_melody' for inst in instruments]
    
    # sample_rhythm 처리
    result['sample_rhythm'] = data['sample_rhythm'].apply(process_sample_rhythm)
    
    # split_data 설정 - all_labels.csv의 정보 사용
    result['split_data'] = split_datas
    
    # id 설정
    result['id'] = data['id']
    
    # 컬럼 순서 재배열
    column_order = [
        '', 'audio_key', 'pitch_range', 'num_measures', 'bpm', 'genre',
        'track_role', 'inst', 'sample_rhythm', 'time_signature', 
        'min_velocity', 'max_velocity', 'split_data', 'id'
    ]
    
    # 요청한 컬럼 순서로 재배열
    result = result[column_order]
    
    return result

def convert_key_signature(key_signature):
    """
    key_signature를 audio_key 형식으로 변환
    예: "C minor" -> "cminor"
    """
    if pd.isna(key_signature):
        return ''
    
    # 소문자로 변환하고 공백 제거
    audio_key = key_signature.lower().replace(' ', '')
    
    return audio_key

import re
from fractions import Fraction
import pandas as pd

# ─────────────────────────────────────────────────────────
# ① 문자열 → Fraction 목록 파싱
# ─────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
sample_rhythm → {standard | triplet | sextuplet | duodecuplet | irregular_x} 분류
"""
import re
import pandas as pd
from fractions import Fraction

# 2의 거듭제곱(규칙 박자) 분모
POW2 = {1, 2, 4, 8, 16, 32}

# ────────────────────────────── 헬퍼 ────────────────────────────── #
def _extract_fractions(txt: str) -> list[Fraction]:
    """
    문자열 안에서
      • Fraction(a, b)
      • 실수(예: 3.75)
    를 찾아 Fraction 객체 목록으로 반환
    """
    if pd.isna(txt):
        return []

    # ① Fraction(·) 패턴
    fracs = [
        Fraction(int(a), int(b))
        for a, b in re.findall(r"Fraction\((\d+),\s*(\d+)\)", str(txt))
    ]

    # ② 소수 → Fraction (분모 최대 48으로 제한)
    fracs += [
        Fraction(float(x)).limit_denominator(48)
        for x in re.findall(r"(?<!\d\.)(\d*\.\d+)", str(txt))
    ]
    return fracs

# ────────────────────────────── 메인 ────────────────────────────── #
def process_sample_rhythm(rhythm_str: str) -> str:
    """sample_rhythm 한 항목을 리듬 유형으로 변환"""
    fracs = _extract_fractions(rhythm_str)

    # Fraction이 하나도 없거나, 모두 2ⁿ 분모라면
    if not fracs:
        return "standard"

    # 등장한 모든 분모
    denoms = {f.denominator for f in fracs}

    # 규칙 분모(2ⁿ) 제거 → 특수 분모 집합
    irregular = {d for d in denoms if d not in POW2}
    if not irregular:
        return "standard"

    # 가장 작은 특수 분모 기준으로 분류
    base = min(irregular)
    if base == 3:
        return "triplet"
    elif base == 6:
        return "sextuplet"
    elif base == 12:
        return "duodecuplet"   # 12-tuplet. 필요하면 명칭 변경!
    else:
        return f"irregular_{base}"

# ────────────────────────────── 사용 예시 ────────────────────────────── #
# df["rhythm_type"] = df["sample_rhythm"].apply(process_sample_rhythm)




def process_instruments(instruments_str):
    """
    instruments 문자열에서 악기 이름 추출
    """
    if pd.isna(instruments_str) or instruments_str == '[None]':
        return 'default_instrument'
    
    try:
        # Python 리스트 형식 처리
        if instruments_str.startswith('[') and instruments_str.endswith(']'):
            # 대괄호 사이의 내용 추출
            content = instruments_str[1:-1].strip()
            
            # 따옴표로 감싸진 문자열(악기 이름) 확인
            if content.startswith("'") and content.endswith("'"):
                instrument = content[1:-1]
                return instrument.lower().replace(' ', '_')
        
        return 'default_instrument'
    except Exception as e:
        print(f"악기 처리 중 오류 발생: {instruments_str}, {e}")
        return 'default_instrument'



def map_pitch_range(min_pitch, max_pitch):
    """
    pitch_min과 pitch_max 값을 기반으로 pitch_range 매핑
    Image 1에 있는 MIDI 노트 범위 기준
    """
    # 평균 피치 계산
    avg_pitch = (min_pitch + max_pitch) / 2
    
    # Image 1에 있는 MIDI 노트 범위 기준
    if avg_pitch < 12:  # C-2 (0) ~ B0 (11)
        return 'very_low'
    elif avg_pitch < 24:  # C1 (12) ~ B1 (23)
        return 'low'
    elif avg_pitch < 36:  # C2 (24) ~ B2 (35)
        return 'mid_low'
    elif avg_pitch < 48:  # C3 (36) ~ B3 (47)
        return 'mid'
    elif avg_pitch < 60:  # C4 (48) ~ B4 (59)
        return 'mid_high'
    elif avg_pitch < 72:  # C5 (60) ~ B5 (71)
        return 'high'
    else:  # C6 (72) 이상
        return 'very_high'

def save_to_csv(data, output_path):
    """
    처리된 데이터를 CSV로 저장
    """
    data.to_csv(output_path, index=False)
    print(f"데이터가 {output_path}에 저장되었습니다.")

# 사용 예시:
# ===================================================
processed = preprocess_music_data(
    commu_meta_path='ktm/ComMU/commu_meta.csv',
    gugak_midi_features_path='extracted_features_all.csv',
    gugak_labels_path='ktm/aihub/all_labels.csv'  # all_labels.csv 사용
)

# 처리된 데이터 저장
save_to_csv(processed['commu_meta'], 'processed_commu.csv')
save_to_csv(processed['processed_gugak'], 'processed_gugak.csv')
#-------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np

# def clean_default_inst(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     1) inst 컬럼 값이 'default_instrument' 인 행 제거
#     2) 인덱스 reset → 맨 앞 정수 컬럼(이름 '0')이 있으면 0부터 다시 채움
#     """
#     # ① default_instrument 행 제거
#     df_clean = df[df["inst"] != "default_instrument"].copy()

#     # ② 인덱스 리셋
#     df_clean.reset_index(drop=True, inplace=True)

#     # ③ '0' 컬럼이 실제로 존재하면 새로 채움
#     if "0" in df_clean.columns:
#         df_clean["0"] = np.arange(len(df_clean))

#     return df_clean


# # ─── 사용 예시 ───────────────────────────────────────────
# df = pd.read_csv("/home/wjg980807/ai_music/dataset/processed_gugak.csv")
# df = clean_default_inst(df)
# df.to_csv("/home/wjg980807/ai_music/dataset/processed_gugak.csv",index=False)
#-------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np

# file =r"/home/wjg980807/ai_music/dataset/processed_gugak.csv"
# df = pd.read_csv(file)

# # 0컬럼을 다시 0~N-1로 덮어쓰기
# df['index'] = np.arange(len(df))
# df.to_csv(file, index=False)
