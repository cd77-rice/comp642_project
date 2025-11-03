from PIL import Image
from PIL import ImageFile
import logging
import sys
import random
import numpy as np

from config import OUTPUT_DIR
from config import Point
from config import Backdrop
from config import Overlay
from config import Subject
from config import Inventory


logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_stream_handler = logging.StreamHandler(sys.stdout)
stdout_stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stdout_stream_handler)


def random_point(overlay: Overlay, background: bool = False) -> tuple[Point, float]:
    """
    Compute a random point within a bounding box.

    `background` = True if object should be placed in background
    """
    bnd_box_p1, bnd_box_p2 = overlay.bound_box

    # y-distance in the bound box is normally distributed with most occurences
    # towards the upper or lower edge (foreground or background)
    if not background:
        y_factor = np.random.normal(loc=0.0, scale=0.3)
        y_factor = min(abs(y_factor), 1.0)
    else:
        y_factor = abs(np.random.normal(loc=0.0, scale=0.05))
        y_factor = 1 - min(abs(y_factor), 1.0)

    distance = int((bnd_box_p2.y - bnd_box_p1.y) * y_factor)
    x = random.randint(bnd_box_p1.x, bnd_box_p2.x)
    y = bnd_box_p1.y + distance

    start_factor = overlay.front_factor
    end_factor = overlay.back_factor
    length = abs(bnd_box_p2.y - bnd_box_p1.y)
    factor = start_factor - distance * (start_factor - end_factor) / length

    return Point(x, y), factor


def overlay_images(backdrop: ImageFile.ImageFile,
                   subject: ImageFile.ImageFile,
                   rotation_deg: float, pos_low_left_center: Point,
                   resize_factor: float) -> Image.Image:
    """
    Overlay subject onto base image.

    CSYS origin is at bottom left corner; x pointing right and y up.
    NOTE: this is different from `PIL.Image` and done for simpler positioning.
    """

    overlay_width, overlay_height = subject.size
    overlay_new_size = (int(overlay_width * resize_factor),
                        int(overlay_height * resize_factor))
    overlay_img_mod = subject.copy()
    overlay_img_mod = overlay_img_mod.resize(overlay_new_size)
    overlay_img_mod = overlay_img_mod.rotate(-rotation_deg)

    base_copy_img = backdrop.copy()
    pos_mod = (
        pos_low_left_center.x - int(overlay_new_size[0] / 2),
        backdrop.size[1] - pos_low_left_center.y - overlay_new_size[1]
    )
    base_copy_img.paste(overlay_img_mod, pos_mod, overlay_img_mod)
    return base_copy_img


def random_overlay(subject: Subject, backdrop_img: ImageFile.ImageFile,
                   backdrop_dpm: float, overlay: Overlay,
                   background: bool = False) -> Image.Image:
    """
    Selects random subject of specified `subject_type_name`, 
    overlays with `backdrop` and returns a tuple of (suject, image).
    """
    subject_image = Image.open(subject.file)
    dpm_subject = subject_image.size[1] / subject.height
    factor_base = backdrop_dpm / dpm_subject

    point, factor_position = random_point(overlay, background)
    angle = random.uniform(-1.0, 1.0)
    if random.random() > 0.5:
        subject_image = subject_image.transpose(Image.FLIP_LEFT_RIGHT)

    return overlay_images(backdrop_img, subject_image, angle, point,
                          factor_base * factor_position)


def generate():
    """
    Generate pictures with random subjects.

    The backdrop is static, subjects are randomly positioned and scaled.
    """

    # TODO: image scaling per Overlay definition
    # TODO: select 1/3 animals, 2/3 human ... stratified

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
    else:
        for file in OUTPUT_DIR.iterdir():
            if file.is_file():
                file.unlink()

    logger.info('>>>> grouping images by subject type')
    subjects_by_type: dict[str, list[Subject]] = {}
    for img_dat in Inventory.subjects:
        images_of_type = subjects_by_type.setdefault(img_dat.type_, list())
        images_of_type.append(img_dat)
    logger.info(f'found distinct subject types: {
                list(subjects_by_type.keys())}')

    logger.info(f'>>>> reading backdrop image {Backdrop.file}')
    backdrop = Backdrop()
    backdrop_image = Image.open(backdrop.file)
    backdrop_dpm = backdrop_image.size[1] / backdrop.height

    total_images = sum(t.count for t in Overlay.groups)
    zfill_width = len(str(total_images))
    logger.info(f'>>>> generating {total_images} images')
    start_idx = 0
    overlay = Overlay()
    for group in overlay.groups:

        if group.subject_type not in subjects_by_type:
            err_msg = f'subject type "{group.subject_type}" not in inventory'
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info(f'{group.count} images of "{group.subject_type}"')

        for idx in range(group.count):
            image_number = f'{idx + 1 + start_idx}'.zfill(zfill_width)
            out_file = OUTPUT_DIR / f'{image_number}-{group.subject_type}.png'

            subject_count = 1
            if group.subject_type == "human" and random.random() < 0.15:
                # background subject
                subject_count = 2
            subjects = random.sample(subjects_by_type[group.subject_type],
                                     k=subject_count)
            overlayed_img = backdrop_image
            if subject_count == 2:
                overlayed_img = random_overlay(subjects[1], overlayed_img,
                                               backdrop_dpm, overlay, True)
            overlayed_img = random_overlay(subjects[0], overlayed_img,
                                           backdrop_dpm, overlay)

            overlayed_img.save(out_file)
            logger.info(
                f'{[subj.file.name for subj in subjects]} -> {out_file.name}')

        start_idx += group.count

    logger.info('>>>> done generating')


if __name__ == '__main__':
    try:
        generate()
    except ValueError as err:
        print(err)
    except Exception as err:
        print(f'Unexpected error: {err}')
        raise err
