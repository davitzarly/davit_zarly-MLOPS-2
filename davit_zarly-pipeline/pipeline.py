"""NEU-DET TFX Pipeline Definition.

This module wires every TFX component together into a single pipeline that
can be orchestrated by Apache Beam (via ``local_runner.py``).

Components used (per submission Kriteria 1):
    1. CsvExampleGen        - ingest the CSV manifest.
    2. StatisticsGen        - compute dataset statistics.
    3. SchemaGen            - infer the data schema.
    4. ExampleValidator     - validate new examples against the schema.
    5. Transform            - feature engineering (one-hot label, image path).
    6. Tuner (bonus)        - hyper-parameter search.
    7. Trainer              - train the CNN classifier.
    8. Resolver             - find the latest blessed model for comparison.
    9. Evaluator            - evaluate the new model against the blessed one.
   10. Pusher               - push the validated model to the serving directory.
"""
from __future__ import annotations

import os
from typing import List

import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)
from tfx.orchestration import pipeline as pipeline_pkg
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel, standard_artifacts

# Bonus: Tuner component (only available on TFX >= 1.0).
try:
    from tfx.components import Tuner
    _TUNER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TUNER_AVAILABLE = False

import configs


def _create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    eval_data_path: str,
    modules_root: str,
    serving_model_dir: str,
    metadata_path: str,
    beam_pipeline_args: List[str],
    enable_tuner: bool = True,
) -> pipeline_pkg.Pipeline:
    """Build the NEU-DET TFX pipeline."""

    # ------------------------------------------------------------------ #
    # 1. Ingest - CsvExampleGen with an 80/20 train/eval split.
    # ------------------------------------------------------------------ #
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_path, output_config=output)

    # ------------------------------------------------------------------ #
    # 2. StatisticsGen
    # ------------------------------------------------------------------ #
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # ------------------------------------------------------------------ #
    # 3. SchemaGen
    # ------------------------------------------------------------------ #
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    # ------------------------------------------------------------------ #
    # 4. ExampleValidator
    # ------------------------------------------------------------------ #
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # ------------------------------------------------------------------ #
    # 5. Transform
    # ------------------------------------------------------------------ #
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join(modules_root, "transform.py"),
    )

    # ------------------------------------------------------------------ #
    # 6. Tuner (bonus) - skip when not requested or unavailable.
    # ------------------------------------------------------------------ #
    tuner = None
    if enable_tuner and _TUNER_AVAILABLE:
        tuner = Tuner(
            module_file=os.path.join(modules_root, "tuner.py"),
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            train_args=trainer_pb2.TrainArgs(num_steps=50),
            eval_args=trainer_pb2.EvalArgs(num_steps=15),
        )

    # ------------------------------------------------------------------ #
    # 7. Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        module_file=os.path.join(modules_root, "trainer.py"),
        custom_executor_spec=GenericExecutor,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=200),
        eval_args=trainer_pb2.EvalArgs(num_steps=50),
        hyperparameters=(tuner.outputs["best_hparams"] if tuner else None),
    )

    # ------------------------------------------------------------------ #
    # 8. Resolver - latest blessed model for comparison.
    # ------------------------------------------------------------------ #
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # ------------------------------------------------------------------ #
    # 9. Evaluator - threshold the new model on the eval split.
    # ------------------------------------------------------------------ #
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key="label"),
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="SparseCategoricalAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericMetricThreshold(
                                lower_bound={"value": 0.80}
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="SparseCategoricalCrossentropy"),
                    tfma.MetricConfig(class_name="SparseTopKCategoricalAccuracy"),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # ------------------------------------------------------------------ #
    # 10. Pusher - promote the model to the serving directory.
    # ------------------------------------------------------------------ #
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            destination=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    # ------------------------------------------------------------------ #
    # Assemble everything into a TFX pipeline.
    # ------------------------------------------------------------------ #
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    if tuner is not None:
        components.insert(5, tuner)

    return pipeline_pkg.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=None,  # use default local SQLite
        enable_cache=True,
        beam_pipeline_args=beam_pipeline_args,
    )


# --------------------------------------------------------------------------- #
# Public helper used by ``local_runner.py``.
# --------------------------------------------------------------------------- #
def create_pipeline():
    """Factory used by the local runner / notebook to instantiate the pipeline."""
    modules_root = os.path.join(os.path.dirname(__file__), "modules")
    return _create_pipeline(
        pipeline_name=configs.PIPELINE_NAME,
        pipeline_root=configs.PIPELINE_ROOT,
        data_path=configs.DATA_ROOT,
        eval_data_path=configs.EVAL_FILE,
        modules_root=modules_root,
        serving_model_dir=configs.SERVING_MODEL_DIR,
        metadata_path=configs.METADATA_PATH,
        beam_pipeline_args=configs.BEAM_PIPELINE_ARGS,
        enable_tuner=True,
    )
