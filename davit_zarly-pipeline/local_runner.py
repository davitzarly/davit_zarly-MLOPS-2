"""Runner for the NEU-DET TFX pipeline.

Ketentuan submission Kriteria 1 mewajibkan seluruh komponen pipeline
dijalankan menggunakan *Pipeline Orchestrator* bernama **Apache Beam**.
TFX menyediakan ``BeamDagRunner`` persis untuk kebutuhan ini -- ia
menerjemahkan graph komponen TFX menjadi sebuah Beam pipeline yang
dieksekusi oleh ``DirectRunner`` (lihat ``configs.BEAM_PIPELINE_ARGS``).

PENTING: gunakan ``BeamDagRunner``, BUKAN ``LocalDagRunner``.
``LocalDagRunner`` tidak diorkestrasi oleh Apache Beam sehingga tidak
memenuhi Kriteria 1 walaupun secara fungsional keduanya bisa menjalankan
pipeline yang sama.

Usage:
    python davit_zarly-pipeline/local_runner.py
"""
from __future__ import annotations

import os
import absl.logging

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

import configs
from pipeline import create_pipeline

absl.logging.set_verbosity(absl.logging.INFO)


def main():
    # Make the NEU-DET base dir explicit for the Transform/Trainer modules
    # regardless of the process' current working directory when Beam
    # spawns worker subprocesses.
    os.environ.setdefault("NEU_DET_BASE_DIR", configs.DATASET_BASE_DIR)

    # Make sure all output directories exist before starting.
    os.makedirs(configs.PIPELINE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(configs.METADATA_PATH), exist_ok=True)
    os.makedirs(configs.SERVING_MODEL_DIR, exist_ok=True)

    pipeline = create_pipeline()

    # Orkestrasi wajib menggunakan Apache Beam (Kriteria 1).
    # beam_pipeline_args (mis. --runner=DirectRunner) sudah didefinisikan
    # di configs.BEAM_PIPELINE_ARGS dan diteruskan lewat pipeline.py.
    BeamDagRunner().run(pipeline)

    absl.logging.info("Pipeline finished (orchestrated by Apache Beam). "
                       "Serving model at: %s", configs.SERVING_MODEL_DIR)


if __name__ == "__main__":
    main()
