"""CPU-only observer events -> auditable historical W&B run.

No torch/model imports. Default is validation only; --emit creates one offline
run. Each output directory is exclusive, preventing accidental duplicate import.
For Stage-2 use exactly one event file (one Slurm attempt) per invocation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite ' + label)
    return value


def convert(paths, stage, first, last):
    records, sources = {}, []
    for path in paths:
        # Snapshot boundary: do not chase an actively growing Stage-2 event log.
        limit = path.stat().st_size
        consumed, h, pending, caches = 0, hashlib.sha256(), {}, {}
        with path.open('rb') as stream:
            while consumed < limit:
                raw = stream.readline(limit - consumed)
                if not raw.endswith(b'\n'):
                    break  # in-progress final JSON line is excluded
                consumed += len(raw)
                h.update(raw)
                event = json.loads(raw)
                it = event.get('iteration')
                if event['event'] == 'cache_provenance':
                    caches[event['cache_key']] = event
                if event['event'] == 'iteration_start':
                    pending[it] = event
                if event['event'] != 'optimizer_step_complete':
                    continue
                if stage == 2 and it > last:
                    continue  # explicit snapshot cutoff, never pretend run is complete
                if it in records:
                    raise ValueError('Duplicate iteration; separate attempts: ' + str(it))
                start = pending.pop(it, None)
                if start is None:
                    raise ValueError('Missing iteration_start: ' + str(it))
                losses = event['losses']
                row = {'optimization/iter': it, 'optimization/epoch': event['epoch'],
                       'train/loss_total': finite(losses['loss'], 'loss'),
                       'runtime/source_attempt': path.parent.name,
                       'runtime/source_elapsed_seconds': finite(event['seconds'], 'elapsed')}
                for target, prefix in [('momentum', 'decode.loss_mse_momentum'),
                                       ('l2', 'decode.loss_l2_render'),
                                       ('ssim', 'decode.loss_ssim_render')]:
                    matched = [v for k, v in losses.items() if k.startswith(prefix)]
                    if len(matched) != 1:
                        raise ValueError('Expected one Config A loss: ' + prefix)
                    row['train/loss_' + target] = finite(matched[0], prefix)
                lr = event.get('lr', start.get('lr'))
                if not isinstance(lr, list) or len(lr) != 1:
                    raise ValueError('Expected single optimizer LR')
                row['optimization/lr'] = finite(lr[0], 'lr')
                row['optimization/grad_norm_preclip'] = finite(
                    event.get('preclip_norm', event.get('grad_norm')), 'preclip')
                if 'postclip_norm' in event:
                    row['optimization/grad_norm_postclip'] = finite(event['postclip_norm'], 'postclip')
                for key, value in event.get('memory', {}).items():
                    row['runtime/gpu_' + key + '_bytes'] = finite(value, key)
                if stage == 1:
                    if event['epoch'] != (it - 1) // 50 + 1:
                        raise ValueError('Stage-1 epoch mapping mismatch')
                    row['runtime/rollout'] = event['rollout']
                else:
                    if event['epoch'] != (it - 1) // 600 + 1:
                        raise ValueError('Stage-2 epoch mapping mismatch')
                    for key in ('requested_rollout', 'effective_rollout', 'cache_key',
                                'window_start', 'source_start', 'target_source'):
                        row['stage2/' + key] = event[key]
                    row['stage2/dense_target_source_start'] = event['source_start'] + 1
                    row['stage2/dense_target_source_end'] = event['target_source']
                    row['stage2/dense_target_local_end'] = event['target_source'] - 113
                    cache = caches[event['cache_key']]
                    row['stage2/cache_sha256'] = cache['sha256']
                    row['stage2/window_source_end_inclusive'] = cache['source_window'][1]
                    row['stage2/window_local_end_inclusive'] = cache['source_window'][1] - 113
                    if event['target_source'] >= 268:
                        raise ValueError('Training target crosses split')
                records[it] = row
        sources.append({'path': str(path), 'snapshot_bytes': consumed,
                        'snapshot_sha256': h.hexdigest()})
    steps = sorted(records)
    if not steps or steps != list(range(first, last + 1)):
        raise ValueError('Missing/unexpected steps: expected %d..%d, got %s..%s (%d)' %
                         (first, last, steps[:1], steps[-1:], len(steps)))
    rows = [records[it] for it in steps]
    ranges = {key: {'min': min(r[key] for r in rows), 'max': max(r[key] for r in rows)}
              for key in rows[0] if key.startswith('train/')}
    return rows, {'step_count': len(rows), 'first_step': first, 'last_step': last,
                  'missing_steps': [], 'duplicate_steps': [], 'loss_ranges': ranges,
                  'sources': sources}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, nargs='+', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--stage', type=int, choices=[1, 2], required=True)
    p.add_argument('--first-step', type=int, required=True)
    p.add_argument('--last-step', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--emit', action='store_true')
    p.add_argument('--name', required=True)
    p.add_argument('--entity', required=True)
    args = p.parse_args()
    if args.stage == 2 and len(args.events) != 1:
        p.error('Stage-2 attempts must use separate runs')
    if args.stage == 1 and (args.first_step, args.last_step) != (1, 2300):
        p.error('T26 historical run requires all 2300 steps')
    rows, validation = convert(args.events, args.stage, args.first_step, args.last_step)
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    checkpoints = report.get('checkpoints', [])
    metadata = {'source': 'historical_import_from_events', 'stage': args.stage,
                'seed': report['seed'], 'training_git_commit': report['head'],
                'training_config_sha256': report['config_sha256'],
                'source_report': str(args.report),
                'source_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
                'event_sources': validation['sources'],
                'checkpoint_metadata': checkpoints,
                'scope': 'Recorded optimizer steps only; no reconstructed telemetry',
                'checkpoint_files_uploaded': False,
                'elapsed_time_semantics': 'seconds since each source process start',
                'loss_semantics': 'Observer aggregates with original weights; no reweighting'}
    if args.stage == 1:
        if len(checkpoints) != 46 or checkpoints[-1]['iter'] != 2300:
            raise ValueError('Expected 46 Stage-1 checkpoint metadata records')
        metadata['final_checkpoint_sha256'] = checkpoints[-1]['sha256']
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    run_id = 'historical-' + fingerprint[:20]
    # Exclusive reservation persists even after failure: investigate rather than duplicate.
    args.output.mkdir(parents=True, exist_ok=False)
    validation.update(run_id=run_id, metadata=metadata, wandb_status='not_created')
    (args.output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    with (args.output / 'metrics.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    if args.emit:
        import wandb
        run = wandb.init(project='SoMA-D360', entity=args.entity, name=args.name, id=run_id,
                         group='stage1_baseline' if args.stage == 1 else 'stage2_pinkcloth_seed5',
                         tags=['deform360', 'pink-cloth', 'no-tactile', 'historical'],
                         job_type='historical-import', mode='offline', dir=str(args.output),
                         config=metadata, settings=wandb.Settings(
                             x_disable_stats=True, x_disable_meta=True,
                             disable_git=True, disable_code=True, save_code=False))
        run.define_metric('optimization/iter')
        run.define_metric('*', step_metric='optimization/iter')
        for row in rows:
            run.log(row, step=row['optimization/iter'])
        run.summary.update({'validated_steps': len(rows), 'first_optimizer_step': args.first_step,
                            'last_optimizer_step': args.last_step,
                            'source': 'historical_import_from_events'})
        run_dir = str(Path(run.dir).parent)
        run.finish()
        validation.update(wandb_status='offline_written_not_uploaded', wandb_directory=run_dir)
        (args.output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({k: v for k, v in validation.items() if k != 'metadata'}, indent=2))


if __name__ == '__main__':
    main()
