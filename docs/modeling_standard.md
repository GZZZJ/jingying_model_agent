# Modeling Standard

Training commands must read project config and train config, require an explicit
feature list, and write all outputs into the active run workspace.

`jm train` supports two modes:

1. Real training when local feather data, feature list, and train config are
   available.
2. Scaffold output when required data is missing. Scaffold outputs are only a
   trace of intent and must not be used as model evidence.

The standard LightGBM path writes:

- `modeling/<experiment>/model.pkl`
- `modeling/<experiment>/metrics_train_valid.json`
- `modeling/<experiment>/feature_importance.csv`
- `modeling/<experiment>/feature_drop_detail.csv`
- `modeling/<experiment>/actual_feature_list.txt`
- `modeling/<experiment>/preprocessing.json`
- `modeling/<experiment>/run_config.json`
- optional scored feather and score-column summary

The current reusable implementation was hardened with the real Fujie GCard
`main_lgbm` case run, but is not GCard-specific. It handles numeric coercion, configured missing sentinels,
low-availability and constant-feature drops, median fill from train, LightGBM
training, feature importance, and all-split scoring.

Project-specific choices such as train/valid split values, label column,
historical score columns, and sample source paths belong in project config or
the model request. Do not hard-code them into generic modules unless they are
truly common across projects.
