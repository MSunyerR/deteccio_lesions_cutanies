"""
Codi utilizat per fer una cerca de paràmetres amb optuna
"""

import optuna
import os
from ultralytics import YOLO


def objective(trial):
    lr0 = trial.suggest_loguniform('lr0', 1e-5, 1e-2)
    lrf = trial.suggest_uniform('lrf', 0.01, 1.0)
    momentum = trial.suggest_uniform('momentum', 0.7, 0.98)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    box = trial.suggest_uniform('box', 0.2, 0.8)
    cls = trial.suggest_uniform('cls', 0.2, 1.0)

    model = YOLO("yolo12n.pt")

    model.train(data='dataset/data.yaml',
                epochs=50,
                imgsz=1024,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                box=box,
                cls=cls,
                single_cls=True,
                batch=24,
                device=0)

    metrics = model.val()
    return metrics.results_dict['metrics/mAP50(B)']

if __name__ == "__main__":

    # Base de dades SQLite per guardar l'estudi
    storage = "sqlite:///optuna_study.db"
    study_name = "yolo_tuning"

    # Crea o carrega l'estudi
    if os.path.exists("optuna_study.db"):
        study = optuna.load_study(study_name=study_name, storage=storage)
        print("Estudi carregat des de fitxer.")
    else:
        study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage)
        print("Nou estudi creat.")

    # Llança el tuning
    study.optimize(objective, n_trials=50)
