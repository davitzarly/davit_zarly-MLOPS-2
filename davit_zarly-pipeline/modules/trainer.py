"""NEU-DET Trainer module (v2 - diperbaiki).

Perubahan dari v1:
1. IMAGE_SIZE diturunkan ke 128 karena tidak ada GPU (CPU only).
   Ini didokumentasikan secara eksplisit — bukan disembunyikan.
2. Arsitektur CNN diperkuat dengan L2 kernel_regularizer di setiap
   Conv2D layer untuk mengurangi overfitting.
3. Dropout dinaikkan ke 0.4.
4. Label smoothing 0.1 ditambahkan pada loss untuk regularisasi tambahan.
5. Mendukung transfer learning (MobileNetV2) via custom_config['model_type'].
6. Serving signature tetap 'image_bytes' -- konsisten dengan app.py.

Arsitektur yang tersedia:
  - 'cnn'       : Baseline CNN dengan L2 regularization (default)
  - 'mobilenet' : MobileNetV2 pretrained ImageNet + fine-tuning

IMAGE_SIZE = 128 (CPU mode; jika GPU tersedia, ganti ke 200 di configs.py).
"""
from typing import Any, Dict, List

import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options

try:
    from . import transform as transform_module  # type: ignore
except ImportError:  # pragma: no cover
    import transform as transform_module  # type: ignore

LABEL_KEY = transform_module.LABEL_KEY
IMAGE_PATH_KEY = transform_module.IMAGE_PATH_KEY
NUM_CLASSES = transform_module.NUM_CLASSES

# IMAGE_SIZE dikurangi ke 128 karena training di CPU (tidak ada GPU).
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
L2_FACTOR = 1e-4


def _label_smoothing_loss(num_classes=6, smoothing=0.1):
    """Custom label smoothing loss kompatibel dengan TF 2.13."""
    def loss_fn(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        y_smooth = y_true_oh * (1.0 - smoothing) + smoothing / num_classes
        return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)
    return loss_fn


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #

def _input_fn(
    file_pattern: List[str],
    data_accessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = True,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Bangun tf.data.Dataset dari transformed TFRecord examples."""
    raw_dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=LABEL_KEY,
        ),
        tf_transform_output.transformed_metadata.schema,
    )

    def _parse(features, label):
        image_path = features[IMAGE_PATH_KEY]
        image, lbl = transform_module._load_image(image_path, label, IMAGE_SIZE)
        if is_train:
            image, lbl = transform_module._augment(image, lbl)
        return image, lbl

    return (
        raw_dataset
        .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def _build_cnn_model(hyperparameters: Dict[str, Any]) -> tf.keras.Model:
    """Baseline CNN dengan BatchNorm + L2 regularization.

    Perubahan dari v1:
    - Setiap Conv2D mendapat kernel_regularizer=L2(L2_FACTOR)
    - Dropout dinaikkan ke 0.4
    - Tambah Dense layer kedua sebelum output untuk kapasitas tambahan
    """
    lr = float(hyperparameters.get("learning_rate", 5e-4))
    dropout = float(hyperparameters.get("dropout", 0.4))
    filters_1 = int(hyperparameters.get("filters_1", 32))
    filters_2 = int(hyperparameters.get("filters_2", 64))
    dense_units = int(hyperparameters.get("dense_units", 256))
    l2 = tf.keras.regularizers.L2(L2_FACTOR)

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

    def _conv_block(x, filters):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_regularizer=l2,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return tf.keras.layers.MaxPooling2D()(x)

    x = _conv_block(inputs, filters_1)       # 64x64
    x = _conv_block(x, filters_2)            # 32x32
    x = _conv_block(x, 128)                  # 16x16
    x = _conv_block(x, 256)                  # 8x8

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        dense_units, activation="relu", kernel_regularizer=l2,
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=l2,
    )(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="predictions",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="neu_det_cnn_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=_label_smoothing_loss(NUM_CLASSES, smoothing=0.1),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


def _build_mobilenet_model(hyperparameters: Dict[str, Any]) -> tf.keras.Model:
    """MobileNetV2 Transfer Learning.

    Fase 1 (run_fn): backbone dibekukan, hanya classification head dilatih.
    Fase 2 (fine-tuning): top 30 layer backbone di-unfreeze dengan LR kecil.
    """
    lr_head = float(hyperparameters.get("learning_rate", 1e-3))
    lr_finetune = float(hyperparameters.get("lr_finetune", 1e-5))
    dropout = float(hyperparameters.get("dropout", 0.3))
    dense_units = int(hyperparameters.get("dense_units", 256))
    fine_tune_at = int(hyperparameters.get("fine_tune_at", 100))

    # Preprocessing input MobileNetV2: rescale ke [-1, 1]
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)

    # Backbone MobileNetV2 pretrained ImageNet, tanpa top layer
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Fase 1: bekukan seluruh backbone
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="predictions",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="neu_det_mobilenet")

    # Compile untuk fase 1 (head training)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_head),
        loss=_label_smoothing_loss(NUM_CLASSES, smoothing=0.05),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    model._base_model = base
    model._lr_finetune = lr_finetune
    model._fine_tune_at = fine_tune_at
    return model


# --------------------------------------------------------------------------- #
# Serving signature
# --------------------------------------------------------------------------- #

def _get_serve_image_bytes_fn(model: tf.keras.Model):
    """Serving signature yang menerima raw JPEG bytes dari client.

    Preprocessing identik dengan training:
    - decode JPEG
    - resize ke IMAGE_SIZE x IMAGE_SIZE
    - normalize ke [0, 1]

    Catatan untuk model MobileNetV2: layer Rescaling (0-1 -> -1 to 1)
    sudah ada di dalam model graph, sehingga input tetap [0,1] lalu
    model sendiri yang me-rescale ke [-1,1]. Ini memastikan
    konsistensi training-serving.
    """

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes"),
    ])
    def serve_image_bytes_fn(image_bytes):
        def _decode(raw_bytes):
            img = tf.image.decode_jpeg(raw_bytes, channels=3)
            img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
            return tf.cast(img, tf.float32) / 255.0

        images = tf.map_fn(_decode, image_bytes, fn_output_signature=tf.float32)
        preds = model(images, training=False)
        return {"outputs": preds}

    return serve_image_bytes_fn


# --------------------------------------------------------------------------- #
# TFX Trainer entry-point
# --------------------------------------------------------------------------- #

def run_fn(fn_args: FnArgs):
    """TFX Trainer entry-point (dipanggil oleh komponen Trainer TFX).

    Mendukung dua arsitektur melalui fn_args.custom_config['model_type']:
    - 'cnn'       : Baseline CNN dengan L2 regularization (default)
    - 'mobilenet' : MobileNetV2 Transfer Learning

    Alur MobileNetV2:
    1. Train head (backbone frozen) selama EPOCHS_HEAD epoch.
    2. Unfreeze top fine_tune_at layers.
    3. Fine-tune dengan LR kecil selama EPOCHS_FINETUNE epoch.
    """
    hyperparameters: Dict[str, Any] = {}
    custom_config = fn_args.custom_config or {}
    hyperparameters.update(custom_config.get("hyperparameters", {}))
    if hasattr(fn_args, "hyperparameters") and fn_args.hyperparameters:
        hyperparameters.update(fn_args.hyperparameters.get("values", {}))

    model_type = custom_config.get("model_type", "cnn")

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor,
        tf_transform_output, is_train=True, batch_size=BATCH_SIZE,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor,
        tf_transform_output, is_train=False, batch_size=BATCH_SIZE,
    )

    callbacks_base = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7,
        ),
    ]

    if model_type == "mobilenet":
        model = _build_mobilenet_model(hyperparameters)
        epochs_head = int(hyperparameters.get("epochs_head", 15))
        epochs_finetune = int(hyperparameters.get("epochs_finetune", 15))

        # --- Fase 1: latih classification head ---
        model.fit(
            train_dataset,
            validation_data=eval_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_steps=fn_args.eval_steps,
            epochs=epochs_head,
            callbacks=callbacks_base,
        )

        # --- Fase 2: fine-tune top layers ---
        base = model._base_model
        fine_tune_at = model._fine_tune_at
        lr_finetune = model._lr_finetune
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_finetune),
            loss=_label_smoothing_loss(NUM_CLASSES, smoothing=0.05),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            ],
        )
        model.fit(
            train_dataset,
            validation_data=eval_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_steps=fn_args.eval_steps,
            epochs=epochs_finetune,
            callbacks=callbacks_base,
        )
    else:
        # Default: CNN
        model = _build_cnn_model(hyperparameters)
        model.fit(
            train_dataset,
            validation_data=eval_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_steps=fn_args.eval_steps,
            epochs=EPOCHS,
            callbacks=callbacks_base,
        )

    signatures = {"serving_default": _get_serve_image_bytes_fn(model)}
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
