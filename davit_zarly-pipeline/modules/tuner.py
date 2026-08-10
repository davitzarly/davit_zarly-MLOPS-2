"""NEU-DET Tuner module (v2 - search space diperluas).

Perubahan dari v1:
1. Search space diperluas: learning_rate, dropout, filters_1, filters_2,
   dense_units (baru), l2_factor implisit lewat model builder.
2. Jumlah trial tetap 5 (resource terbatas).
3. Tuner dapat berjalan untuk arsitektur CNN saja (MobileNetV2 tidak di-tune
   lewat Tuner karena terlalu lama; fine-tuning config-nya di-hardcode di
   custom_config).
"""
import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

try:
    from . import trainer as trainer_module
except ImportError:  # pragma: no cover
    import trainer as trainer_module  # type: ignore


class NEUDETHyperModel(kt.HyperModel):
    """Keras-Tuner HyperModel untuk CNN NEU-DET."""

    def __init__(self, tf_transform_output: tft.TFTransformOutput):
        self.tf_transform_output = tf_transform_output

    def build(self, hp):
        hyperparameters = {
            "learning_rate": hp.Float(
                "learning_rate", min_value=1e-4, max_value=5e-3, sampling="log",
            ),
            "dropout": hp.Float(
                "dropout", min_value=0.3, max_value=0.5, step=0.1,
            ),
            "filters_1": hp.Choice("filters_1", [32, 64]),
            "filters_2": hp.Choice("filters_2", [64, 128]),
            "dense_units": hp.Choice("dense_units", [128, 256]),
        }
        return trainer_module._build_cnn_model(hyperparameters)


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """TFX Tuner entry-point."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    tuner = kt.RandomSearch(
        NEUDETHyperModel(tf_transform_output),
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=5,
        seed=42,
        directory=fn_args.working_dir,
        project_name="neu_det_tuner_v2",
    )

    train_dataset = trainer_module._input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=trainer_module.BATCH_SIZE,
    )
    eval_dataset = trainer_module._input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=trainer_module.BATCH_SIZE,
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 5,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=3,
                    restore_best_weights=True,
                ),
            ],
        },
    )
