import numpy as np
import SimpleITK as sitk
from PIL import Image

from dataset_preprocessing.CAMUS.volume_calculation import compute_left_ventricle_volumes

LV_LABEL = 1


def sitk_load(path):
    img = sitk.ReadImage(str(path))
    arr = np.squeeze(sitk.GetArrayFromImage(img))
    info = {"spacing": img.GetSpacing()}
    return arr, info


def to_uint8_0_255(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255).astype(np.uint8)


def resize_2d(x, size=(112, 112), is_mask=False):
    mode = Image.NEAREST if is_mask else Image.BILINEAR
    img = Image.fromarray(x)
    img = img.resize((size[1], size[0]), mode)
    return np.asarray(img)


def load_split(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def load_img_or_mask(patient_dir, patient_id, view, phase, gt=False):
    suffix = "_gt" if gt else ""
    path = patient_dir / f"{patient_id}_{view}_{phase}{suffix}.nii.gz"
    return sitk_load(path)


def get_spacing(info):
    # wie im Notebook: spacing[:2][::-1]
    return info["spacing"][:2][::-1]


def compute_ef_from_volumes(edv, esv):
    return round(100 * (edv - esv) / edv)


def compute_monoplane_volume_from_mask(mask, spacing):
    """
    A4C-only Volumenapproximation aus LV-cavity-Maske.

    Formel:
        V = 8 * A^2 / (3 * pi * L)

    mask: binary LV-cavity mask, original resolution
    spacing: (dy, dx) in mm
    return: volume in ml
    """
    import numpy as np

    mask = mask > 0

    if not np.any(mask):
        return np.nan

    dy, dx = spacing

    area_px = np.sum(mask)
    area_mm2 = area_px * dy * dx

    ys, xs = np.where(mask)

    # einfache Längenapproximation
    length_px = ys.max() - ys.min() + 1
    length_mm = length_px * dy

    if length_mm <= 0:
        return np.nan

    volume_mm3 = (8.0 * area_mm2 ** 2) / (3.0 * np.pi * length_mm)
    volume_ml = volume_mm3 / 1000.0

    return volume_ml


def load_patient_camus(patient_root, patient_id, target_size=(112, 112), debug=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    patient_dir = Path(patient_root) / patient_id

    # --------------------------------------------------
    # 1) 4CH Bilder laden: EchoNet-ähnliches ED/ES-Paar
    # --------------------------------------------------
    ed_img_raw, _ = load_img_or_mask(patient_dir, patient_id, "4CH", "ED", gt=False)
    es_img_raw, _ = load_img_or_mask(patient_dir, patient_id, "4CH", "ES", gt=False)

    ed_img = resize_2d(to_uint8_0_255(ed_img_raw), target_size, is_mask=False)
    es_img = resize_2d(to_uint8_0_255(es_img_raw), target_size, is_mask=False)

    # --------------------------------------------------
    # 2) 4CH GT-Masken laden und LV-only Masken erzeugen
    # --------------------------------------------------
    ed_mask_raw, _ = load_img_or_mask(patient_dir, patient_id, "4CH", "ED", gt=True)
    es_mask_raw, _ = load_img_or_mask(patient_dir, patient_id, "4CH", "ES", gt=True)

    ed_mask_lv_raw = (ed_mask_raw == LV_LABEL).astype(np.uint8)
    es_mask_lv_raw = (es_mask_raw == LV_LABEL).astype(np.uint8)

    ed_mask_lv = resize_2d(ed_mask_lv_raw, target_size, is_mask=True).astype(np.uint8)
    es_mask_lv = resize_2d(es_mask_lv_raw, target_size, is_mask=True).astype(np.uint8)

    # --------------------------------------------------
    # 3) A4C GT-Masken in Originalauflösung für A4C-only EF
    # --------------------------------------------------
    a4c_ed, a4c_info = load_img_or_mask(patient_dir, patient_id, "4CH", "ED", gt=True)
    a4c_es, _ = load_img_or_mask(patient_dir, patient_id, "4CH", "ES", gt=True)

    # --------------------------------------------------
    # 4) Debug-Plot
    # --------------------------------------------------
    if debug:
        print(f"\nPatient: {patient_id}")
        print("Unique labels 4CH ED:", np.unique(a4c_ed))
        print("Unique labels 4CH ES:", np.unique(a4c_es))
        print("Current LV_LABEL:", LV_LABEL)
        print("LV area ED:", np.sum(a4c_ed == LV_LABEL))
        print("LV area ES:", np.sum(a4c_es == LV_LABEL))

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))

        axes[0, 0].imshow(ed_img_raw, cmap="gray")
        axes[0, 0].set_title("ED image raw")

        axes[0, 1].imshow(a4c_ed, cmap="jet", interpolation="nearest")
        axes[0, 1].set_title("ED all labels")

        axes[0, 2].imshow(ed_img_raw, cmap="gray")
        axes[0, 2].contour(a4c_ed == LV_LABEL, levels=[0.5], linewidths=1.5)
        axes[0, 2].set_title(f"ED LV_LABEL={LV_LABEL}")

        axes[0, 3].imshow(ed_img, cmap="gray")
        axes[0, 3].contour(ed_mask_lv == 1, levels=[0.5], linewidths=1.5)
        axes[0, 3].set_title("ED saved LV mask resized")

        axes[1, 0].imshow(es_img_raw, cmap="gray")
        axes[1, 0].set_title("ES image raw")

        axes[1, 1].imshow(a4c_es, cmap="jet", interpolation="nearest")
        axes[1, 1].set_title("ES all labels")

        axes[1, 2].imshow(es_img_raw, cmap="gray")
        axes[1, 2].contour(a4c_es == LV_LABEL, levels=[0.5], linewidths=1.5)
        axes[1, 2].set_title(f"ES LV_LABEL={LV_LABEL}")

        axes[1, 3].imshow(es_img, cmap="gray")
        axes[1, 3].contour(es_mask_lv == 1, levels=[0.5], linewidths=1.5)
        axes[1, 3].set_title("ES saved LV mask resized")

        for ax in axes.flatten():
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 5) A4C-only Volumen + EF berechnen
    # --------------------------------------------------
    a4c_spacing = get_spacing(a4c_info)

    edv = compute_monoplane_volume_from_mask(
        a4c_ed == LV_LABEL,
        spacing=a4c_spacing,
    )

    esv = compute_monoplane_volume_from_mask(
        a4c_es == LV_LABEL,
        spacing=a4c_spacing,
    )

    ef_moplane = compute_ef_from_volumes(edv, esv)

    a2c_ed, a2c_info = load_img_or_mask(patient_dir, patient_id, "2CH", "ED", gt=True)
    a2c_es, _ = load_img_or_mask(patient_dir, patient_id, "2CH", "ES", gt=True)
    a4c_ed, a4c_info = load_img_or_mask(patient_dir, patient_id, "4CH", "ED", gt=True)
    a4c_es, _ = load_img_or_mask(patient_dir, patient_id, "4CH", "ES", gt=True)

    a2c_spacing = get_spacing(a2c_info)
    a4c_spacing = get_spacing(a4c_info)

    edv, esv = compute_left_ventricle_volumes(
        a2c_ed == LV_LABEL,
        a2c_es == LV_LABEL,
        a2c_spacing,
        a4c_ed == LV_LABEL,
        a4c_es == LV_LABEL,
        a4c_spacing,
    )

    ef_biplane = compute_ef_from_volumes(edv, esv)






    # --------------------------------------------------
    # 6) Speichern im gewünschten Format
    # --------------------------------------------------
    frames = np.stack([ed_img, es_img], axis=-1)  # H, W, 2
    frames = frames[None, :, :, :]               # 1, H, W, 2

    masks = np.stack([ed_mask_lv, es_mask_lv], axis=-1)  # H, W, 2
    masks = masks[None, :, :, :]                        # 1, H, W, 2

    return ef_moplane,ef_biplane, frames, patient_id, np.array([edv, esv]), masks

def build_split(patient_root, split_txt, target_size=(112, 112)):
    patient_ids = load_split(split_txt)

    ef_moplanes = []
    ef_biplanes = []
    max_min_frames = []
    p_ids = []
    max_min_volumes = []
    max_min_masks = []

    for pid in patient_ids:
        ef_moplane,ef_biplane, frames, p_id, volumes, masks = load_patient_camus(
            patient_root,
            pid,
            target_size=target_size,
        )

        ef_moplanes.append(ef_moplane)
        ef_biplanes.append(ef_biplane)
        max_min_frames.append(frames)
        p_ids.append(p_id)
        max_min_volumes.append(volumes)
        max_min_masks.append(masks)

    return (
        np.asarray(ef_moplanes, dtype=np.float32),
        np.asarray(ef_biplanes, dtype=np.float32),
        np.asarray(max_min_frames, dtype=np.uint8),
        np.asarray(p_ids),
        np.asarray(max_min_volumes, dtype=np.float32),
        np.asarray(max_min_masks, dtype=np.uint8),
    )


def create_camus_npz(
    patient_root,
    train_split,
    val_split,
    test_split,
    output_file,
    target_size=(112, 112),
):
    train = build_split(patient_root, train_split, target_size)
    val = build_split(patient_root, val_split, target_size)
    test = build_split(patient_root, test_split, target_size)

    np.savez_compressed(
        output_file,

        train_ef_motargets=train[0],
        train_ef_bitargets=train[1],
        train_max_min_frame=train[2],
        train_p_id=train[3],
        train_max_min_volumes=train[4],
        train_max_min_masks=train[5],

        val_ef_motargets=val[0],
        val_ef_bitargets=val[1],
        val_max_min_frame=val[2],
        val_p_id=val[3],
        val_max_min_volumes=val[4],
        val_max_min_masks=val[5],

        test_ef_motargets=test[0],
        test_ef_bitargets=test[1],
        test_max_min_frame=test[2],
        test_p_id=test[3],
        test_max_min_volumes=test[4],
        test_max_min_masks=test[5],
    )
    print('save')