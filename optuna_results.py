"""
Codi per analitzar l'estudi realitzat amb optuna (grafics)
"""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances
)

# Carrega l’estudi existent
storage = "sqlite:///optuna_study.db"
study_name = "yolo_tuning"
study = optuna.load_study(study_name=study_name, storage=storage)

# 1. Resultat òptim
best_trial = study.best_trial
print(f"Millor mAP50: {best_trial.value:.4f}")
print("Millors hiperparàmetres:")
for key, val in best_trial.params.items():
    print(f"  • {key}: {val}")

# 2. DataFrame de tots els experiments
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)

# Opcional: desa’l a CSV
df.to_csv("optuna_trials.csv", index=False)

# 3. Visualitzacions per entendre l’espai de cerca
fig1 = plot_optimization_history(study)
fig1.show()

fig2 = plot_parallel_coordinate(study)
fig2.show()

fig3 = plot_param_importances(study)
fig3.show()
