"""Sequential, resumable paired-seed study with immutable input manifests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/research_v2'


def main():
    RESULTS.mkdir(exist_ok=True)
    manifest = ROOT / 'data/research_v2/manifest.json'
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    protocol = {
        'manifest_sha256': manifest_hash, 'seeds': [17, 43, 101],
        'variants': ['sipakmed_only', 'combined'], 'epochs_ceiling': 150,
        'patience': 30, 'workers': 2, 'imgsz': 640,
        'primary_endpoint': 'Abnormal-cell detection recall and false detections per field at a threshold selected only on development data',
        'limitation': 'Retrospective reanalysis; historical data already examined. Dense fields held out of new training and checkpoint selection. No clinical claim.',
    }
    protocol_path = RESULTS / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; create a new version')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    status_path = RESULTS / 'training_status.json'
    status = {'pid': os.getpid(), 'started': time.time(), 'protocol': protocol, 'runs': []}
    for seed in protocol['seeds']:
        for variant in protocol['variants']:
            name = f'v2_{variant}_seed{seed}'
            summary = ROOT / 'results' / f'{name}_training_summary.json'
            checkpoint = RESULTS / 'runs' / name / 'weights/best.pt'
            if summary.exists() and checkpoint.exists():
                status['runs'].append({'name': name, 'status': 'complete_existing'})
                continue
            if (RESULTS / 'runs' / name).exists():
                raise RuntimeError(f'Incomplete run {name}; inspect before resuming or restarting')
            command = [sys.executable, '-u', str(ROOT / 'training_pipeline.py'), '--data',
                       str(ROOT / 'data/research_v2' / variant / 'data.yaml'),
                       '--project', str(RESULTS / 'runs'), '--name', name,
                       '--seed', str(seed), '--epochs', '150', '--patience', '30', '--workers', '2']
            entry = {'name': name, 'status': 'running', 'started': time.time(), 'command': command}
            status['runs'].append(entry)
            status_path.write_text(json.dumps(status, indent=2))
            print(f'Starting {name}', flush=True)
            with (RESULTS / f'{name}.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                entry['pid'] = process.pid
                status_path.write_text(json.dumps(status, indent=2))
                return_code = process.wait()
            entry.update(status='complete' if return_code == 0 and checkpoint.exists() and summary.exists() else 'failed',
                         returncode=return_code, ended=time.time())
            status_path.write_text(json.dumps(status, indent=2))
            print(f'{name}: {entry["status"]}', flush=True)
            if entry['status'] != 'complete':
                raise RuntimeError(f'Inspect log for {name}; not proceeding with incomplete evidence')
    status['status'] = 'training_complete_evaluation_pending'
    status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
