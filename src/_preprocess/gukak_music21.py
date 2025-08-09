import os
import pandas as pd
from music21 import converter, chord, note, stream, tempo, meter, instrument, analysis

# 기본 디렉토리 설정
base_directory = '/home/wjg980807/ai_music/dataset/ktm/aihub'

# 모든 소스 디렉토리 경로 가져오기
def get_all_directories(base_dir):
    directories = []
    
    # train/src 디렉토리 내 모든 하위 디렉토리 찾기
    train_src_path = os.path.join(base_dir, 'train', 'src')
    if os.path.exists(train_src_path):
        for dir_name in os.listdir(train_src_path):
            dir_path = os.path.join(train_src_path, dir_name)
            if os.path.isdir(dir_path):
                directories.append(dir_path)
    
    # val/src 디렉토리 내 모든 하위 디렉토리 찾기
    val_src_path = os.path.join(base_dir, 'val', 'src')
    if os.path.exists(val_src_path):
        for dir_name in os.listdir(val_src_path):
            dir_path = os.path.join(val_src_path, dir_name)
            if os.path.isdir(dir_path):
                directories.append(dir_path)
    
    return directories

# 현재 작업 디렉토리 및 MIDI 파일 리스트 수집
def get_midi_files(dir_path):
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    if not os.path.isdir(dir_path):
        print(f"Error: '{dir_path}' 폴더를 찾을 수 없습니다.")
        return []
    files = [f for f in os.listdir(dir_path) if f.lower().endswith('.mid')]
    # print(f"Found {len(files)} MIDI files in '{dir_path}': {files}")
    return [os.path.join(dir_path, f) for f in files]

# MIDI 파일에서 특성 추출 함수 정의
def extract_features(midi_path):
    print(f"Processing: {midi_path}")
    try:
        s = converter.parse(midi_path)

        # 조성 (Key Signature) 분석
        try:
            key_obj = s.analyze('key')
            key_signature = f"{key_obj.tonic.name} {key_obj.mode}"
        except Exception:
            key_signature = None

        # 화음 진행 (Chordify)
        chords = s.chordify().recurse().getElementsByClass(chord.Chord)
        chord_progressions = ['.'.join(p.name for p in c.pitches) for c in chords]

        # 음높이 범위: Note와 Chord 모두 처리
        pitch_values = []
        for element in s.recurse():
            if isinstance(element, note.Note):
                pitch_values.append(element.pitch.midi)
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    pitch_values.append(p.midi)
        pitch_min, pitch_max = (min(pitch_values), max(pitch_values)) if pitch_values else (None, None)

        # 마디 수
        measures = s.recurse().getElementsByClass(stream.Measure)
        num_measures = len(measures)

        # 템포 (BPM)
        bpm = None
        mm = s.flat.getElementsByClass(tempo.MetronomeMark)
        if mm:
            bpm = mm[0].getQuarterBPM()

        # 박자표
        ts = s.flat.getElementsByClass(meter.TimeSignature)
        time_signature = ts[0].ratioString if ts else None

        # 리듬 샘플 (노트 길이 리스트)
        sample_rhythm = []
        for element in s.recurse():
            if isinstance(element, (note.Note, chord.Chord)):
                sample_rhythm.append(element.duration.quarterLength)

        # 벨로시티 최소/최대: Note와 Chord 모두 처리
        velocities = []
        for element in s.recurse():
            if isinstance(element, note.Note) and element.volume and element.volume.velocity is not None:
                velocities.append(element.volume.velocity)
            elif isinstance(element, chord.Chord) and element.volume and element.volume.velocity is not None:
                velocities.append(element.volume.velocity)
        min_velocity, max_velocity = (min(velocities), max(velocities)) if velocities else (None, None)

        # 악기 정보
        insts = []
        parts = instrument.partitionByInstrument(s)
        if parts:
            for p in parts:
                if p.getInstrument() and p.getInstrument().instrumentName:
                    insts.append(p.getInstrument().instrumentName)
        else:
            insts = list({n.getInstrument().instrumentName for n in s.recurse().notes 
                         if hasattr(n, 'getInstrument') and n.getInstrument() 
                         and n.getInstrument().instrumentName})

        # music_src_nm 설정: '디렉토리명/파일명'
        # 예: 'TS_유사국악_R_창작국악/BP_CR1_02925'
        dir_name = os.path.basename(os.path.dirname(midi_path))
        file_name = os.path.splitext(os.path.basename(midi_path))[0]
        # music_src_nm = f"{dir_name}/{file_name}"

        return {
            'id': file_name,  # 파일명만 ID로 설정
            'key_signature': key_signature,
            'chord_progressions': chord_progressions,
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            'num_measures': num_measures,
            'bpm': bpm,
            'time_signature': time_signature,
            'sample_rhythm': sample_rhythm,
            'min_velocity': min_velocity,
            'max_velocity': max_velocity,
            'instruments': insts,
            # 'music_src_nm': music_src_nm
        }
    except Exception as e:
        print(f"Error extracting features from {midi_path}: {e}")
        return None

def main():
    # 모든 디렉토리 가져오기
    all_directories = get_all_directories(base_directory)
    print(f"총 {len(all_directories)}개의 디렉토리를 찾았습니다:")
    for dir_path in all_directories:
        print(f"  - {dir_path}")
    
    if not all_directories:
        print("처리할 디렉토리가 없습니다. 기본 디렉토리 경로를 확인해주세요.")
        return

    all_rows = []
    
    for directory in all_directories:
        print(f"\n처리 중인 디렉토리: {directory}")
        midi_files = get_midi_files(directory)
        if not midi_files:
            print(f"  이 디렉토리에는 MIDI 파일이 없습니다.")
            continue
        
        print(f"  {len(midi_files)}개의 MIDI 파일을 처리합니다...")
        for midi_path in midi_files:
            try:
                features = extract_features(midi_path)
                if features:
                    all_rows.append(features)
            except Exception as e:
                print(f"  파일 처리 중 오류 발생: {midi_path}: {e}")
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = 'extracted_features_all.csv'
        df.to_csv(out_path, index=False)
        print(f"\n특성 추출 완료. [{out_path} 저장됨]")
        print(f"추출된 파일 수: {len(df)}")
    else:
        print("\n추출된 데이터가 없습니다. MIDI 파일 내용을 확인하세요.")

if __name__ == '__main__':
    main()