#!/usr/bin/env python
# scripts/summarize_label_distribution.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.utils import load_config, get_project_root
from src.data import MultiGraderDataset

def summarize_split(df, split, cfg, simplified, hierarchical, label_cols):
    df_split = df[df['split'] == split]
    paths    = df_split[cfg['training']['datamodule']['video_col']].tolist()
    labels   = df_split[label_cols].values.tolist()

    ds = MultiGraderDataset(
        video_paths     = paths,
        labels          = labels,
        transform       = None,
        hierarchical    = hierarchical,
        simplified_base = simplified
    )

    base_ctr = Counter()
    sub_ctr  = Counter() if hierarchical else None

    for item in ds:
        if hierarchical:
            b = item['base_label'].item()
            s = item['subclass_label'].item()
            base_ctr[b] += 1
            sub_ctr[s]  += 1
        else:
            lab = item[1][0].item()
            base_ctr[lab] += 1

    idx_to_base = ds.idx_to_class if not hierarchical else ds.idx_to_base

    print(f"\n--- {split.upper()} (simplified={simplified}, hierarchical={hierarchical}) ---")
    for idx, cnt in sorted(base_ctr.items()):
        print(f"  {idx_to_base[idx]:>2}: {cnt}")
    if hierarchical:
        print("  subclass distribution:")
        for idx, cnt in sorted(sub_ctr.items()):
            print(f"    {ds.idx_to_subclass[idx]:>3}: {cnt}")

    names  = [idx_to_base[i] for i in sorted(base_ctr)]
    counts = [base_ctr[i]        for i in sorted(base_ctr)]
    fig = plt.figure(figsize=(4,3))
    plt.bar(names, counts)
    plt.title(f"{split} – base classes")
    plt.ylabel("count")
    plt.xlabel("class")
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Summarize class distributions for train/val/test splits")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--simplified", action="store_true",
                        help="Use simplified_base=True")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical=True (ignored if simplified)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = pd.read_csv(cfg['paths']['data_csv'])
    label_cols = cfg['training']['datamodule']['label_cols']

    # build output folder under <project_root>/<work_dir>/data_summary
    work_dir = cfg['output_configuration']['work_dir']
    out_dir  = os.path.join(get_project_root(), work_dir, "data_summary")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train","val","test"]:
        fig = summarize_split(
            df,
            split,
            cfg,
            simplified   = args.simplified,
            hierarchical = (args.hierarchical and not args.simplified),
            label_cols   = label_cols
        )
        fig_path = os.path.join(out_dir,
                                f"{split}_dist_simp{args.simplified}"
                                f"_hier{args.hierarchical}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {split} summary to {fig_path}")

if __name__ == "__main__":
    main()
