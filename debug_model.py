"""
סקריפט דיבאג — מריץ את המודל על קבצי הבכי ומראה מה הוא מחזיר בפועל.
הרץ מאותו טרמינל שאתה מריץ בו את השרת:
  python debug_model.py
"""
import json
import os
import sys
import numpy as np
import librosa
from tensorflow import keras

print(f"Python: {sys.executable}")
print(f"Working dir: {os.getcwd()}")

# --- טעינת מודל ---
model = keras.models.load_model("models/sos_model.keras")
with open("models/norm_stats.json") as f:
    stats = json.load(f)
MEAN, STD = stats["mean"], stats["std"]
print(f"Norm stats: mean={MEAN:.2f}, std={STD:.2f}")

SR = 22050
DURATION = 2.0
CATEGORIES = ["crying", "background"]

def predict_segment(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel_raw_mean = mel.mean()
    mel = (mel - MEAN) / (STD + 1e-8)
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
    mel_in = mel[np.newaxis, ..., np.newaxis]
    probs = model.predict(mel_in, verbose=0)[0]
    probs = np.nan_to_num(probs, nan=0.5)
    probs = probs / (probs.sum() + 1e-8)
    return probs, mel_raw_mean

print("\n" + "="*60)
print("בדיקת קבצי בכי:")
print("="*60)

cry_files = [f for f in os.listdir("audio") if f.endswith((".mp3", ".wav")) and "baby" in f.lower() or "cry" in f.lower()]
cry_files = [f for f in os.listdir("audio") if f.endswith((".mp3", ".wav"))]

for fname in sorted(cry_files):
    path = os.path.join("audio", fname)
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        chunk = int(SR * DURATION)
        segments = []
        for start in range(0, len(audio) - chunk + 1, chunk):
            seg = audio[start:start+chunk]
            rms = float(np.sqrt(np.mean(seg**2)))
            probs, mel_mean = predict_segment(seg)
            segments.append((start//SR, probs, rms, mel_mean))

        if not segments:
            seg = np.pad(audio, (0, chunk - len(audio)))
            rms = float(np.sqrt(np.mean(seg**2)))
            probs, mel_mean = predict_segment(seg)
            segments.append((0, probs, rms, mel_mean))

        max_cry = max(p[0] for _, p, _, _ in segments)
        print(f"\n{'='*50}")
        print(f"קובץ: {fname}")
        print(f"  max crying = {max_cry*100:.1f}%")
        for t, probs, rms, mel_mean in segments:
            status = "🚨 בכי" if probs[0] >= 0.35 else "🔇 רקע"
            print(f"  {t:>3}s | crying={probs[0]*100:5.1f}% bg={probs[1]*100:5.1f}% | rms={rms:.4f} mel_mean={mel_mean:.1f} | {status}")

    except Exception as e:
        print(f"  ❌ שגיאה: {e}")

print("\n" + "="*60)
print("סיכום:")
print("  אם כל הקבצים מראים bg>95% — המודל לא עובד כמו שצריך")
print("  אם קבצי בכי מזוהים — הבעיה היא רק ב-realtime (מיקרופון/באפר)")
print("="*60)
