# Randomly False Patches

Currently, researchers only utilze the patches containing objects (True Patch, TP) to train models on large-scale aerial images. However, the patches with no object (False Patches, FP) are equally valuable for models. FPs include more scenes than TPs, which will benefit model to suppress false positives. but in most cases, the number of FP is far beyond the number of TP. Directly training TPs and FPs will cost unendurable time. Thus, to fully utilze FPs, we add part FPs in dataset and shuffle it in each epoch.

# Results

**note**: The argument `filter_empty` in split config should be setted to `false` to keep FP information in splitting phase.

To be continue!!!