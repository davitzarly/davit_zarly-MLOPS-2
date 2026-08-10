"""NEU-DET Transform module (v2 - diperbaiki).

Perubahan dari v1:
1. DATASET_BASE_DIR dibaca lazy (di dalam fungsi) bukan saat import,
   agar Apache Beam worker yang spawn di direktori berbeda tetap mendapat
   path yang benar.
2. _augment() diperkuat dengan lebih banyak variasi augmentation yang
   relevan untuk citra cacat permukaan baja:
   - random_flip_left_right (arah horizontal tidak mengubah arti cacat)
   - random_flip_up_down    (arah vertikal juga tidak signifikan untuk defect)
   - random_brightness      (max_delta dinaikkan 0.1 -> 0.2)
   - random_contrast        (range diperlebar 0.9-1.1 -> 0.75-1.25)
   - random_saturation      (tambahan baru)
   - random_hue             (tambahan baru, kecil, untuk variasi sensor)
"""
import os
import tensorflow as tf
import tensorflow_transform as tft

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #
LABEL_KEY = "label"
IMAGE_PATH_KEY = "image_path"
NUM_CLASSES = 6

VOCAB = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def _get_dataset_base_dir() -> str:
    """Lazy reader untuk base direktori dataset NEU-DET.

    Dibuat sebagai fungsi (bukan module-level constant) supaya Apache Beam
    worker yang diinisialisasi di direktori berbeda tetap mendapat nilai
    yang benar dari environment variable yang di-set oleh local_runner.py
    / notebook sebelum pipeline dijalankan.
    """
    return os.environ.get("NEU_DET_BASE_DIR", os.getcwd())


def _fill_in_missing(x):
    """Ganti missing value dengan string kosong."""
    if isinstance(x, tf.sparse.SparseTensor):
        x = tf.sparse.to_dense(x, default_value="")
    return x


def preprocessing_fn(inputs):
    """TFX Transform entry-point.

    Args:
        inputs: Dict mapping feature name -> batches of raw values.

    Returns:
        Dict mapping feature name -> batches of transformed values.
    """
    outputs = {}

    # 1. Image path - pertahankan sebagai dense string agar Trainer dapat
    #    memuat file JPEG secara lazy saat training (menghindari bloat
    #    pada artefak TFRecord Transform).
    image_path = _fill_in_missing(inputs[IMAGE_PATH_KEY])
    outputs[IMAGE_PATH_KEY] = image_path

    # 2. Label - encode ke integer 0..5 menggunakan vocabulary tetap.
    #    Menggunakan compute_and_apply_vocabulary agar vocabulary tersimpan
    #    di transform_graph dan dapat diinspeksi di kemudian hari.
    label = _fill_in_missing(inputs[LABEL_KEY])
    label_indices = tft.compute_and_apply_vocabulary(
        label,
        top_k=NUM_CLASSES,
        num_oov_buckets=0,
        vocab_filename="label_vocab",
        default_value=-1,
    )
    outputs[LABEL_KEY] = tf.cast(label_indices, tf.int64)

    return outputs


# --------------------------------------------------------------------------- #
# Helper functions yang dikonsumsi oleh Trainer module
# --------------------------------------------------------------------------- #

def _load_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    image_size: int = 128,
) -> tuple:
    """Muat dan decode satu gambar JPEG dari disk.

    Args:
        image_path: Tensor string berisi path relatif gambar.
        label:      Tensor integer kelas.
        image_size: Ukuran target (resize ke image_size x image_size).

    Returns:
        Tuple (image_float32, label).
    """
    base_dir = _get_dataset_base_dir()
    # Gunakan '/' sebagai separator agar portabel antara OS
    path = tf.strings.join([base_dir, image_path], separator="/")
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple:
    """Augmentation data training untuk citra cacat permukaan baja.

    Strategi augmentation dipilih berdasarkan karakteristik domain:
    - Flip horizontal/vertikal: orientasi cacat baja tidak memiliki makna
      arah, sehingga kedua flip aman digunakan.
    - Brightness & contrast: mensimulasikan variasi pencahayaan sensor.
    - Saturation & hue: variasi kecil untuk robustness warna.
    - TIDAK digunakan: rotation besar, zoom ekstrem, cutout —
      augmentation tersebut berpotensi mengubah pola tekstur cacat.

    PENTING: Fungsi ini HANYA dipanggil untuk split training.
             Evaluation/validation TIDAK mendapat augmentation.
    """
    # Spatial augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Photometric augmentations
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Clip ke [0,1] setelah augmentation photometric
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
