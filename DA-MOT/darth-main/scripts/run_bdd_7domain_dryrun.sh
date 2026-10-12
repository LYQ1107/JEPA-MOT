#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
bash scripts/check_and_clean_darth_7domain.sh
bash scripts/prepare_bdd_7domain_cocovid.sh
python tools/generate_bdd_7domain_configs.py
bash scripts/init_bdd_7domain_jobs.sh
python - <<'PY'
import json, pathlib
root=pathlib.Path('data/bdd_da/annotations/box_track_20')
for d in ['clear','dawn_dusk','night','partly_cloudy','overcast','rainy','snowy']:
    print('DOMAIN', d)
    for sp in ['train','val','all']:
        p=root/f'box_track_{d}_{sp}_cocofmt.json'
        if not p.exists(): print(' ', sp, 'MISSING'); continue
        data=json.load(open(p)); print(' ', sp, 'videos', len(data.get('videos',[])), 'images', len(data.get('images',[])), 'annotations', len(data.get('annotations',[])))
PY
