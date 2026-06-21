import shutil, os, csv

src_base = r'C:\Users\This User\Downloads\UrbanSound8K'
dst_base = r'C:\Users\This User\Desktop\SOS-Audio-Detection\data\raw'
mapping = {
    '2': 'scream',      # children_playing
    '6': 'explosion',   # gun_shot
    '0': 'background',  # air_conditioner
    '1': 'background',  # car_horn
    '3': 'background',  # dog_bark
    '4': 'background',  # drilling
    '5': 'background',  # engine_idling
    '7': 'background',  # jackhammer
    '9': 'background',  # street_music
}

copied = {cat: 0 for cat in set(mapping.values())}
skipped = 0

csv_path = os.path.join(src_base, 'metadata', 'UrbanSound8K.csv')
rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
print(f'סה"כ שורות ב-CSV: {len(rows)}')

for row in rows:
    cid = row['classID']
    if cid not in mapping:
        skipped += 1
        continue
    cat = mapping[cid]
    src = os.path.join(src_base, 'audio', 'fold' + row['fold'], row['slice_file_name'])
    dst = os.path.join(dst_base, cat, 'us8k_' + row['slice_file_name'])
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        copied[cat] += 1

print('הועתקו (חדשים):', copied)
print(f'דולגו (siren — כבר ממופה מ-ESC50): {skipped}')

print('\nסה"כ בכל קטגוריה:')
for cat in ['scream', 'crying', 'explosion', 'background']:
    count = len(os.listdir(os.path.join(dst_base, cat)))
    print(f'  {cat}: {count}')
