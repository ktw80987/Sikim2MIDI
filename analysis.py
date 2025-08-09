import pretty_midi
import os

def analyze_midi_file(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)

    # 악기 추출
    instruments = [inst for inst in midi.instruments if not inst.is_drum]
    instrument_names = [pretty_midi.program_to_instrument_name(inst.program) for inst in instruments]
    main_instruments = list(set(instrument_names))[:3] if instrument_names else ['없음']

    # 피치 분석
    pitches = [note.pitch for inst in instruments for note in inst.notes]
    if pitches:
        avg_pitch = sum(pitches) / len(pitches)
        pitch_range = (min(pitches), max(pitches))

        if avg_pitch < 50:
            pitch_desc = '저음역대'
        elif avg_pitch < 70:
            pitch_desc = '중음역대'
        else:
            pitch_desc = '고음역대'
    else:
        avg_pitch, pitch_range, pitch_desc = 0, (0, 0), '없음'

    # 템포 추출
    _, tempi = midi.get_tempo_changes()
    tempo = round(tempi[0], 2) if len(tempi) > 0 else '알 수 없음'

    # 박자 추출
    ts = midi.time_signature_changes
    time_sig = f'{ts[0].numerator}/{ts[0].denominator}' if ts else '알 수 없음'

    # 길이 (초)
    duration_sec = round(midi.get_end_time(), 2)

    # 🎚 벨로시티
    velocities = [note.velocity for inst in instruments for note in inst.notes]
    velocity_avg = round(sum(velocities)/len(velocities), 2) if velocities else '없음'

    # ✅ 요약 출력
    print(f' MIDI 분석 결과 ({os.path.basename(midi_path)})')
    print(f"- 주요 악기 : {', '.join(main_instruments)}")
    print(f'- 음역대 : 평균 {pitch_desc} ({pitch_range[0]} ~ {pitch_range[1]})')
    print(f'- 평균 피치 : {round(avg_pitch, 2)}')
    print(f'- 평균 벨로시티 : {velocity_avg}')
    print(f'- 템포 (BPM) : {tempo}')
    print(f'- 박자 : {time_sig}')
    print(f'- 길이 (초) : {duration_sec}')
    print('-' * 50)

# 사용 예시
midi_path = './output2.mid'
analyze_midi_file(midi_path)