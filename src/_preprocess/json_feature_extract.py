from pathlib import Path
import json

# ─── 1) 매핑 테이블 정의 ───────────────────────────────────

# 악기 코드 → 악기명
instrument_map = {
    'SP01':'가야금','SP02':'거문고','SP03':'비파',
    'SP04':'철현금','SP05':'금','SP06':'슬',
    'SR01':'해금','SR02':'아쟁',
    'WN01':'대금','WN02':'소금','WN03':'단소',
    'WN04':'통소','WN05':'훈','WN06':'지','WN07':'소',
    'WR01':'피리','WR02':'태평소','WR03':'생황',
    'WR04':'나발','WR05':'나각',
    'PN01':'장구','PN02':'꽹과리','PN03':'북','PN04':'징',
    'PN05':'바라','PN06':'목탁','PN07':'충','PN08':'소고',
    'PN09':'정주','PN10':'축','PN11':'어',
    'PT01':'양금','PT02':'편종','PT03':'편경',
    'VF01':'여성','VM02':'남성','VH03':'혼성'
}

def get_mode_category(mode_cd: str) -> str:
    """Mode 코드(MFxx/MGxx) → 평조계열 / 계면조계열"""
    if mode_cd.startswith('MF'):
        return '평조계열'
    if mode_cd.startswith('MG'):
        return '계면조계열'
    return ''

def get_beat_category(beat_cd: str) -> str:
    """Beat 코드(DQxx/QNxx → 균등, ENxx → 불균등, NB → 혼합 및 기타)"""
    if beat_cd.startswith(('DQ','QN')):
        return '균등'
    if beat_cd.startswith('EN'):
        return '불균등'
    if beat_cd.startswith('NB'):
        return '혼합 및 기타'
    return ''

def get_sigimsa(mode_cd: str) -> str:
    """시김새 코드(VBxx → 요성, DWxx → 퇴성, SPxx → 전성)"""
    if mode_cd.startswith('VB'):
        return '요성'
    if mode_cd.startswith('DW'):
        return '퇴성'
    if mode_cd.startswith('SP'):
        return '전성'
    return ''

# ─── 2) 데이터 추출 함수 ───────────────────────────────────
def extract_to_row(data: dict, category_folder_name: str, split: str) -> list:
    """
    반환: [instrument_name, mode_category, beat_category,
           gukak_style, music_src_nm, data_split]
    """
    instrument_cd = data['music_type_info']['instrument_cd']
    mode_cd       = data['annotation_data_info']['mode_cd']
    gukak_beat_cd = data['annotation_data_info']['gukak_beat_cd']
    music_src_nm  = data['music_source_info']['music_src_nm']

    gukak_style   = category_folder_name.split('_')[-1]

    instrument_name = instrument_map.get(instrument_cd, '')
    mode_category   = get_mode_category(mode_cd)
    beat_category   = get_beat_category(gukak_beat_cd)

    return [
        instrument_name,
        mode_category,
        beat_category,
        gukak_style,
        music_src_nm,
        split                    # ★ train / val 구분 추가
    ]

# ─── 3) 폴더 전체 처리 ───────────────────────────────────
def process_label_folder(label_dir: Path, split: str) -> list:
    """split: 'train' 또는 'val'"""
    rows = []
    for json_path in label_dir.rglob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            category = json_path.parent.name
            rows.append(extract_to_row(data, category, split))
        except KeyError as e:
            print(f"[경고] {json_path} 필드 누락: {e}")
        except json.JSONDecodeError as e:
            print(f"[경고] {json_path} JSON 파싱 실패: {e}")
    return rows

# ─── 4) 스크립트 실행부 ───────────────────────────────────
if __name__ == "__main__":
    train_label_dir = Path("train/label")
    val_label_dir   = Path("val/label")

    train_rows = process_label_folder(train_label_dir, split='train')
    val_rows   = process_label_folder(val_label_dir,   split='val')

    header = [
        'instrument_name',
        'mode_category', 'beat_category',
        'gukak_style',
        'music_src_nm',
        'split_data'         # ★ 헤더 추가
    ]

    # train + val 통합
    all_rows = train_rows + val_rows
    print(f"▶ 통합 레코드 수: {len(all_rows)}개")

    # CSV 저장
    import csv
    with open('all_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    print("✅ CSV 파일로 저장 완료: all_labels.csv")