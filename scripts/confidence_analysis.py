#!/usr/bin/env python3
"""
Step 8: Confidence vs Correctness analysis script

Usage examples:
  # Analyze a single confidence JSON file and write report
  python scripts/confidence_analysis.py --input metrics/baseline-bilstm-crf_confidence.json --output reports/baseline_conf_report.json --examples 50

  # Analyze all *_confidence.json files in metrics/
  python scripts/confidence_analysis.py --input metrics/ --output reports/all_conf_reports.json --examples 50

Behavior:
- Streams large JSON array files (the extraction output) without loading entire file into memory.
- Computes mean BiLSTM/CRF confidence for correct vs incorrect tokens.
- Computes mean confidence grouped by entity type.
- Collects low-confidence-correct examples and high-confidence-incorrect examples (configurable thresholds).
- Produces a JSON report and optional CSVs for examples. Attempts to produce simple plots if matplotlib/seaborn are available.
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from typing import Generator, Dict, Any

try:
    import math
    from pathlib import Path
except Exception:
    pass


def parse_json_array(path: str) -> Generator[Dict[str, Any], None, None]:
    """Yield objects from a JSON array in file without loading entire array.
    Assumes top-level is a JSON array: [ { ... }, { ... }, ... ]
    Works by scanning characters and extracting balanced {...} objects.
    """
    with open(path, 'r', encoding='utf8') as f:
        buf = ''
        in_obj = False
        brace_count = 0
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            for ch in chunk:
                if not in_obj:
                    if ch == '{':
                        in_obj = True
                        brace_count = 1
                        buf = ch
                    else:
                        # skip until object starts
                        continue
                else:
                    buf += ch
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # complete object
                            try:
                                yield json.loads(buf)
                            except Exception as e:
                                # fallback: small cleanup and try again
                                try:
                                    cleaned = buf.strip().rstrip(',')
                                    yield json.loads(cleaned)
                                except Exception:
                                    raise
                            in_obj = False
                            buf = ''
        # If any leftover (unlikely), try to parse
        if buf:
            s = buf.strip()
            if s:
                try:
                    yield json.loads(s)
                except Exception:
                    pass


def analyze_file(path: str, low_thr: float = 0.5, high_thr: float = 0.8, max_examples: int = 100, export_all: bool = False, export_dir: str = None):
    """Analyze one confidence JSON array file and return a report dict."""
    totals = {
        'tokens': 0,
        'correct': 0,
        'incorrect': 0,
    }

    # accumulators for bilstm and crf confidences
    acc = {
        'bilstm': {'correct_sum': 0.0, 'incorrect_sum': 0.0, 'correct_count': 0, 'incorrect_count': 0},
        'crf': {'correct_sum': 0.0, 'incorrect_sum': 0.0, 'correct_count': 0, 'incorrect_count': 0},
    }

    # per-entity type stats
    entity_stats = defaultdict(lambda: {'bilstm_sum_correct': 0.0, 'bilstm_sum_incorrect': 0.0, 'crf_sum_correct': 0.0, 'crf_sum_incorrect': 0.0, 'correct_count': 0, 'incorrect_count': 0})

    low_conf_correct = []
    high_conf_incorrect = []

    # If export_all is set, open CSVs and stream matching rows to disk as they are found
    low_writer = None
    high_writer = None
    low_exported = 0
    high_exported = 0
    if export_all:
        import csv as _csv
        out_dir = export_dir or os.path.dirname(path) or '.'
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        low_csv_path = os.path.join(out_dir, base + '_low_conf_correct.csv')
        high_csv_path = os.path.join(out_dir, base + '_high_conf_incorrect.csv')
        low_f = open(low_csv_path, 'w', newline='', encoding='utf8')
        high_f = open(high_csv_path, 'w', newline='', encoding='utf8')
        keys = ['doc_id', 'token_position', 'token_text', 'gold_label', 'predicted_label', 'bilstm_confidence', 'crf_confidence']
        low_writer = _csv.DictWriter(low_f, fieldnames=keys)
        high_writer = _csv.DictWriter(high_f, fieldnames=keys)
        low_writer.writeheader()
        high_writer.writeheader()

    # boundary errors (same entity type but boundary positions differ)
    boundary = {'count': 0, 'bilstm_sum': 0.0, 'crf_sum': 0.0}

    # label-level confusion counters (gold -> pred) with average confidence accumulation
    confusion = defaultdict(lambda: {'count': 0, 'bilstm_sum': 0.0, 'crf_sum': 0.0})

    # For streaming—use generator
    for obj in parse_json_array(path):
        totals['tokens'] += 1
        correct = obj.get('correct', False)
        bilstm_conf = obj.get('bilstm_confidence', None)
        crf_conf = obj.get('crf_confidence', None)
        gold = obj.get('gold_label')
        pred = obj.get('predicted_label')
        gold_ent = obj.get('gold_entity_type', 'O')
        pred_ent = obj.get('predicted_entity_type', 'O')
        gold_boundary = obj.get('gold_boundary_position')
        pred_boundary = obj.get('predicted_boundary_position')
        if correct:
            totals['correct'] += 1
            if bilstm_conf is not None:
                acc['bilstm']['correct_sum'] += float(bilstm_conf)
                acc['bilstm']['correct_count'] += 1
            if crf_conf is not None:
                acc['crf']['correct_sum'] += float(crf_conf)
                acc['crf']['correct_count'] += 1
            entity_stats[gold_ent]['bilstm_sum_correct'] += float(bilstm_conf) if bilstm_conf is not None else 0.0
            if crf_conf is not None:
                entity_stats[gold_ent]['crf_sum_correct'] += float(crf_conf)
            entity_stats[gold_ent]['correct_count'] += 1

            # low-confidence correct
            if bilstm_conf is not None and float(bilstm_conf) < low_thr:
                row = {
                    'doc_id': obj.get('doc_id'),
                    'token_position': obj.get('token_position'),
                    'token_text': obj.get('token_text'),
                    'gold_label': gold,
                    'predicted_label': pred,
                    'bilstm_confidence': bilstm_conf,
                    'crf_confidence': crf_conf,
                }
                if export_all:
                    low_writer.writerow({k: row.get(k) for k in ['doc_id', 'token_position', 'token_text', 'gold_label', 'predicted_label', 'bilstm_confidence', 'crf_confidence']})
                    low_exported += 1
                else:
                    if len(low_conf_correct) < max_examples:
                        low_conf_correct.append(row)
        else:
            totals['incorrect'] += 1
            if bilstm_conf is not None:
                acc['bilstm']['incorrect_sum'] += float(bilstm_conf)
                acc['bilstm']['incorrect_count'] += 1
            if crf_conf is not None:
                acc['crf']['incorrect_sum'] += float(crf_conf)
                acc['crf']['incorrect_count'] += 1
            entity_stats[gold_ent]['bilstm_sum_incorrect'] += float(bilstm_conf) if bilstm_conf is not None else 0.0
            if crf_conf is not None:
                entity_stats[gold_ent]['crf_sum_incorrect'] += float(crf_conf)
            entity_stats[gold_ent]['incorrect_count'] += 1

            # high-confidence incorrect
            if bilstm_conf is not None and float(bilstm_conf) > high_thr:
                row = {
                    'doc_id': obj.get('doc_id'),
                    'token_position': obj.get('token_position'),
                    'token_text': obj.get('token_text'),
                    'gold_label': gold,
                    'predicted_label': pred,
                    'bilstm_confidence': bilstm_conf,
                    'crf_confidence': crf_conf,
                }
                if export_all:
                    high_writer.writerow({k: row.get(k) for k in ['doc_id', 'token_position', 'token_text', 'gold_label', 'predicted_label', 'bilstm_confidence', 'crf_confidence']})
                    high_exported += 1
                else:
                    if len(high_conf_incorrect) < max_examples:
                        high_conf_incorrect.append(row)

        # boundary error detection: same entity type (not 'O') but different boundary positions
        if gold_ent != 'O' and pred_ent != 'O' and gold_ent == pred_ent and gold != pred:
            # e.g., gold B-PER but predicted I-PER or wrong boundary tags
            if gold_boundary and pred_boundary and gold_boundary != pred_boundary:
                boundary['count'] += 1
                if bilstm_conf is not None:
                    boundary['bilstm_sum'] += float(bilstm_conf)
                if crf_conf is not None:
                    boundary['crf_sum'] += float(crf_conf)

        # confusion
        key = f"{gold} -> {pred}"
        confusion[key]['count'] += 1
        if bilstm_conf is not None:
            confusion[key]['bilstm_sum'] += float(bilstm_conf)
        if crf_conf is not None:
            confusion[key]['crf_sum'] += float(crf_conf)

    # Compose report
    report = {
        'file': path,
        'totals': totals,
        'bilstm': {},
        'crf': {},
        'entity_stats': {},
        'low_confidence_correct_examples': low_conf_correct,
        'high_confidence_incorrect_examples': high_conf_incorrect,
        'boundary_errors': boundary,
        'confusion_summary': {},
    }

    # bilstm aggregates
    b = acc['bilstm']
    report['bilstm']['mean_confidence_correct'] = (b['correct_sum'] / b['correct_count']) if b['correct_count'] > 0 else None
    report['bilstm']['mean_confidence_incorrect'] = (b['incorrect_sum'] / b['incorrect_count']) if b['incorrect_count'] > 0 else None
    report['bilstm']['correct_count'] = b['correct_count']
    report['bilstm']['incorrect_count'] = b['incorrect_count']

    # crf aggregates
    c = acc['crf']
    report['crf']['mean_confidence_correct'] = (c['correct_sum'] / c['correct_count']) if c['correct_count'] > 0 else None
    report['crf']['mean_confidence_incorrect'] = (c['incorrect_sum'] / c['incorrect_count']) if c['incorrect_count'] > 0 else None
    report['crf']['correct_count'] = c['correct_count']
    report['crf']['incorrect_count'] = c['incorrect_count']

    # entity-level
    for ent, s in entity_stats.items():
        ent_entry = {
            'correct_count': s['correct_count'],
            'incorrect_count': s['incorrect_count'],
            'mean_bilstm_conf_correct': (s['bilstm_sum_correct'] / s['correct_count']) if s['correct_count'] > 0 else None,
            'mean_bilstm_conf_incorrect': (s['bilstm_sum_incorrect'] / s['incorrect_count']) if s['incorrect_count'] > 0 else None,
            'mean_crf_conf_correct': (s['crf_sum_correct'] / s['correct_count']) if s['correct_count'] > 0 and s['crf_sum_correct'] > 0 else None,
            'mean_crf_conf_incorrect': (s['crf_sum_incorrect'] / s['incorrect_count']) if s['incorrect_count'] > 0 and s['crf_sum_incorrect'] > 0 else None,
        }
        report['entity_stats'][ent] = ent_entry

    # confusion summary: keep top N frequent confusions
    confusion_list = []
    for k, v in confusion.items():
        confusion_list.append({'pair': k, 'count': v['count'], 'mean_bilstm_conf': (v['bilstm_sum'] / v['count']) if v['count'] > 0 else None, 'mean_crf_conf': (v['crf_sum'] / v['count']) if v['count'] > 0 else None})
    confusion_list.sort(key=lambda x: x['count'], reverse=True)
    report['confusion_summary'] = confusion_list[:200]

    # If we exported to CSVs, close files and report counts
    if export_all:
        try:
            low_f.close()
            high_f.close()
        except Exception:
            pass
        report['low_confidence_correct_exported'] = low_exported
        report['high_confidence_incorrect_exported'] = high_exported

    return report


def analyze_path(input_path: str, **kwargs):
    results = []
    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if fname.endswith('_confidence.json') or fname.endswith('_confidence.jsonl') or fname.endswith('.json'):
                path = os.path.join(input_path, fname)
                print(f"Analyzing {path}...")
                results.append(analyze_file(path, **kwargs))
    else:
        results.append(analyze_file(input_path, **kwargs))
    return results


def try_plot(report, out_prefix):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        print("matplotlib/seaborn not available — skipping plots")
        return

    # If example lists exist, create simple histograms
    all_bilstm_scores_correct = []
    all_bilstm_scores_incorrect = []

    # try to re-scan the original file to build distributions if needed
    path = report.get('file')
    if not path:
        return

    for obj in parse_json_array(path):
        bilstm_conf = obj.get('bilstm_confidence', None)
        if bilstm_conf is None:
            continue
        if obj.get('correct', False):
            all_bilstm_scores_correct.append(float(bilstm_conf))
        else:
            all_bilstm_scores_incorrect.append(float(bilstm_conf))

    plt.figure(figsize=(8, 4))
    sns.histplot(all_bilstm_scores_correct, color='g', label='correct', stat='density', kde=True, bins=50)
    sns.histplot(all_bilstm_scores_incorrect, color='r', label='incorrect', stat='density', kde=True, bins=50)
    plt.legend()
    plt.title(os.path.basename(path) + ' – BiLSTM confidence (correct vs incorrect)')
    plt.xlabel('BiLSTM confidence')
    plt.tight_layout()
    out_file = out_prefix + '_bilstm_conf_hist.png'
    plt.savefig(out_file)
    print(f"Saved histogram to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Path to a confidence JSON file or directory containing confidence JSON files')
    p.add_argument('--output', '-o', default='reports/confidence_analysis.json', help='Path to write JSON report (if input is directory, this will contain list of reports)')
    p.add_argument('--low-threshold', type=float, default=0.5, help='Threshold to consider a correct prediction as low-confidence')
    p.add_argument('--high-threshold', type=float, default=0.8, help='Threshold to consider an incorrect prediction as high-confidence')
    p.add_argument('--examples', type=int, default=50, help='Max number of examples to collect for low/high cases')
    p.add_argument('--plot', action='store_true', help='Attempt to generate plots (requires matplotlib and seaborn)')
    p.add_argument('--export-all', action='store_true', help='Stream every matching example to CSV (avoid storing examples in memory)')
    p.add_argument('--export-dir', default=None, help='Directory where streamed CSVs will be written (default: same directory as input file)')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    paths = [args.input]
    results = []
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            if fname.endswith('_confidence.json') or fname.endswith('.json'):
                full = os.path.join(args.input, fname)
                print(f"Processing {full}")
                results.append(analyze_file(full, low_thr=args.low_threshold, high_thr=args.high_threshold, max_examples=args.examples, export_all=args.export_all, export_dir=args.export_dir))
    else:
        results = analyze_path(args.input, low_thr=args.low_threshold, high_thr=args.high_threshold, max_examples=args.examples, export_all=args.export_all, export_dir=args.export_dir)

    # If multiple reports, write as list
    out_data = results[0] if len(results) == 1 else results
    with open(args.output, 'w', encoding='utf8') as out_f:
        json.dump(out_data, out_f, indent=2, ensure_ascii=False)
    print(f"Wrote analysis report to {args.output}")

    # Save example CSVs for each report (only when not using streaming export)
    for rep in results:
        base = os.path.splitext(os.path.basename(rep['file']))[0]
        out_dir = os.path.dirname(args.output) or '.'
        if not args.export_all:
            low_csv = os.path.join(out_dir, base + '_low_conf_correct.csv')
            high_csv = os.path.join(out_dir, base + '_high_conf_incorrect.csv')
            # write CSVs
            import csv
            if rep['low_confidence_correct_examples']:
                keys = ['doc_id', 'token_position', 'token_text', 'gold_label', 'predicted_label', 'bilstm_confidence', 'crf_confidence']
                with open(low_csv, 'w', newline='', encoding='utf8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for r in rep['low_confidence_correct_examples']:
                        writer.writerow({k: r.get(k) for k in keys})
                print(f"Wrote low-confidence-correct examples to {low_csv}")
            if rep['high_confidence_incorrect_examples']:
                keys = ['doc_id', 'token_position', 'token_text', 'gold_label', 'predicted_label', 'bilstm_confidence', 'crf_confidence']
                with open(high_csv, 'w', newline='', encoding='utf8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for r in rep['high_confidence_incorrect_examples']:
                        writer.writerow({k: r.get(k) for k in keys})
                print(f"Wrote high-confidence-incorrect examples to {high_csv}")

        # If streaming export was used, report where CSVs were written
        if args.export_all:
            export_dir = args.export_dir or os.path.dirname(rep['file']) or '.'
            print(f"Streamed exported examples for {base} into {export_dir} (files: {base}_low_conf_correct.csv, {base}_high_conf_incorrect.csv)")

        if args.plot:
            out_prefix = os.path.join(out_dir, base)
            try_plot(rep, out_prefix)

    print("Done.")


if __name__ == '__main__':
    main()
