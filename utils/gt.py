import pandas as pd

df = pd.read_csv("Data-Set/padel-data-labels/labels/2022_BCN_FinalM_1_pose.csv")
filtrat = df[(df["seconds"] >= 15) & (df["seconds"] <= 120)]

gt = filtrat[["frame", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]]
gt.to_csv("Data-Set/padel-data-labels/GT_Retallat_15_120.csv", index=False)

print(f"GT generat: {len(gt)} files")
print(gt.head(3))