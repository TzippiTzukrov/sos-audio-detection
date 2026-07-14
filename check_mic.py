"""
בדיקת מכשירי אודיו זמינים
הרץ: python check_mic.py
"""
import sounddevice as sd
import numpy as np

print("=== כל מכשירי האודיו ===")
devices = sd.query_devices()
for i, d in enumerate(devices):
    tag = ""
    if d['max_input_channels'] > 0:
        tag += " [INPUT]"
    if d['max_output_channels'] > 0:
        tag += " [OUTPUT]"
    print(f"  [{i:2d}] {d['name']}{tag}")

print()
default_in = sd.default.device[0]
default_out = sd.default.device[1]
print(f"ברירת מחדל INPUT:  [{default_in}] {devices[default_in]['name']}")
print(f"ברירת מחדל OUTPUT: [{default_out}] {devices[default_out]['name']}")

print()
print("=== בדיקת קריאה קצרה (0.5 שניות) ===")
try:
    audio = sd.rec(int(22050 * 0.5), samplerate=22050, channels=1, dtype="float32", blocking=True).flatten()
    print(f"  הצלחה! RMS={np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  min={audio.min():.4f}, max={audio.max():.4f}")
    has_inf = not np.all(np.isfinite(audio))
    print(f"  יש inf/NaN: {has_inf}")
except Exception as e:
    print(f"  שגיאה: {e}")
