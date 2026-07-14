"""
בדיקת המודל על כל קבצי האודיו בתיקיית audio/
מריץ: python test_audio_files.py
"""
import json, os, glob
import numpy as np
import librosa
from tensorflow import keras

# --- טעינת מודל ---
model = keras.models.load_model('models/sos_model.keras')
with open('models/norm_stats.json') as f:
    stats = json.load(f)
MEAN, STD = stats['mean'], stats['std']

SR        = 22050
DURATION  = 2.0
THRESHOLD = 0.35
SUPPORTED = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')

def predict_file(path):
    """מחזיר רשימת תוצאות לכל 2 שניות בקובץ"""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    chunk     = int(SR * DURATION)
    results   = []

    for start in range(0, len(audio) - chunk + 1, chunk):
        seg = audio[start:start + chunk]
        mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=128)
        mel = librosa.power_to_db(mel, ref=1.0)   # ref קבוע — עקבי עם realtime
        mel = (mel - MEAN) / (STD + 1e-8)
        mel = mel[np.newaxis, ..., np.newaxis]
        probs = model.predict(mel, verbose=0)[0]
        results.append({
            'start': start // SR,
            'crying': float(probs[0]),
            'background': float(probs[1]),
        })

    if not results:
        # קובץ קצר מ-2 שניות — padding
        seg = np.pad(audio, (0, chunk - len(audio)))
        mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=128)
        mel = librosa.power_to_db(mel, ref=1.0)   # ref קבוע — עקבי עם realtime
        mel = (mel - MEAN) / (STD + 1e-8)
        mel = mel[np.newaxis, ..., np.newaxis]
        probs = model.predict(mel, verbose=0)[0]
        results.append({'start': 0, 'crying': float(probs[0]), 'background': float(probs[1])})

    return results

# --- הרצה על כל הקבצים ---
audio_files = sorted([
    f for f in glob.glob('audio/*')
    if f.lower().endswith(SUPPORTED)
])

print(f'נמצאו {len(audio_files)} קבצים\n')
print('=' * 60)

for path in audio_files:
    filename = os.path.basename(path)
    try:
        results = predict_file(path)
        max_cry = max(r['crying'] for r in results)
        avg_cry = np.mean([r['crying'] for r in results])
        detected = max_cry >= THRESHOLD

        status = '🚨 בכי' if detected else '🔇 רקע'
        print(f'\n{status} | {filename}')
        print(f'       max crying={max_cry*100:.1f}%  avg={avg_cry*100:.1f}%  ({len(results)} segments)')

        # פירוט לפי segment
        for r in results:
            bar      = '█' * int(r['crying'] * 20)
            alert    = ' ← 🚨' if r['crying'] >= THRESHOLD else ''
            print(f'       {r["start"]:>3}s: crying={r["crying"]*100:5.1f}% [{bar:<20}]{alert}')

    except Exception as e:
        print(f'\n❌ שגיאה ב-{filename}: {e}')

print('\n' + '=' * 60)
print(f'threshold = {THRESHOLD}')
