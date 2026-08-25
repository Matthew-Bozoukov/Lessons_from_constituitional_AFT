# Resolution sweep — `min_cluster_size` chosen by measurement

A floor on FEATURE STRINGS, not rollouts. `exportable` counts the groups that clear `min_group_records` and therefore reach the property list. `collapsing` means at least one seed's UMAP fit degenerated (the exported run retries past that, but a resolution that needs the retry is not one to report from). `bifurcating` means the seeds disagree about how many groups exist; `unstable` means they agree on the count and disagree about membership. Report from a `stable` row only.

## clusters_reasoning

| min_cluster_size | groups (3 seeds) | exportable | noise | pairwise ARI | collapsed | |
|--:|---|---|---|---|--:|---|
| 15 | [120, 130, 129] | [117, 127, 126] | ['24%', '27%', '25%'] | [0.6049, 0.6392, 0.6256] | 0 | stable |
| 25 | [74, 77, 74] | [73, 76, 73] | ['26%', '28%', '27%'] | [0.6748, 0.7038, 0.6509] | 0 | stable |
| 40 | [46, 45, 47] | [46, 45, 47] | ['29%', '28%', '27%'] | [0.7231, 0.73, 0.6795] | 0 | stable |
| 60 | [31, 27, 29] | [31, 27, 29] | ['28%', '23%', '25%'] | [0.7027, 0.7528, 0.7562] | 0 | stable |
| 90 | [18, 17, 20] | [18, 17, 20] | ['32%', '31%', '29%'] | [0.7839, 0.8156, 0.7997] | 0 | stable |
| 130 | [7, 10, 8] | [7, 10, 8] | ['27%', '28%', '24%'] | [0.7184, 0.8036, 0.7769] | 0 | stable |

## clusters_response

| min_cluster_size | groups (3 seeds) | exportable | noise | pairwise ARI | collapsed | |
|--:|---|---|---|---|--:|---|
| 15 | [118, 120, 121] | [117, 119, 120] | ['26%', '25%', '26%'] | [0.6499, 0.6428, 0.6223] | 0 | stable |
| 25 | [73, 65, 70] | [72, 64, 69] | ['25%', '25%', '25%'] | [0.6512, 0.6793, 0.6325] | 0 | stable |
| 40 | [41, 40, 3] | [41, 40, 3] | ['32%', '27%', '0%'] | [0.5996, 0.0365, 0.0328] | 0 | bifurcating |
| 60 | [23, 6, 2] | [23, 6, 2] | ['31%', '4%', '1%'] | [0.0446, 0.0397, 0.9256] | 0 | bifurcating |
| 90 | [16, 2, 2] | [16, 2, 2] | ['35%', '2%', '1%'] | [0.0431, 0.0471, 0.9639] | 0 | bifurcating |
| 130 | [6, 2, 2] | [6, 2, 2] | ['18%', '3%', '2%'] | [0.1891, 0.1708, 0.9238] | 0 | bifurcating |
