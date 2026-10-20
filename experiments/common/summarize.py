"""把 results/*.json (pair_eval.py 输出) 汇总成 markdown 表格: python experiments/common/summarize.py results/*.json"""
import json
import sys


def main():
    rows = []
    for path in sorted(sys.argv[1:]):
        with open(path) as f:
            r = json.load(f)
        for name, d in r['datasets'].items():
            zs1 = d.get('zero_shot_ref1', {}).get('top1')
            zs5 = d.get('zero_shot_ref5', {}).get('top1')
            rows.append([r['model'], name, d['n_images'], d['n_classes'], f"{d['pr_auc']:.4f}", f"{d['roc_auc']:.4f}",
                         f"{d['best_f1']:.4f}", f"{d['best_threshold']:.3f}",
                         f'{zs1:.3f}' if zs1 is not None else '-', f'{zs5:.3f}' if zs5 is not None else '-'])
    head = ['model', 'dataset', 'images', 'classes', 'PR-AUC', 'ROC-AUC', 'best F1', 'threshold', 'zs top1 (1 ref)',
            'zs top1 (5 ref)']
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for row in rows:
        print('| ' + ' | '.join(map(str, row)) + ' |')


if __name__ == '__main__':
    main()
