# after_edit

这一版是我按仓库评估重新整理后的结果文件夹。

数据集：

- UCI `Predict Students Dropout and Academic Success`
- 总样本：4424
- 划分：train / val / test = 3097 / 663 / 664

当前比较好的结果：

- `transformer_large`
  - test acc: 0.7831
  - macro-F1: 0.7131
  - best val macro-F1: 0.7142

这里放的是本轮最能说明结果的图：

- `transformer_large_curves.png`
- `transformer_large_confusion.png`
- `transformer_scale_comparison.png`
- `mlp_baseline_curves.png`

这版训练是实际跑出来的，不是手工拼的示意图。
