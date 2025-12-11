from PIL import Image
from PIL import ImageFile
from PIL.PngImagePlugin import PngInfo
import random
import pathlib
import shutil
import numpy as np

from config import logger
from config import Process
from config import OUTPUT_DIR, SubjectGroup
from config import Point
from config import Backdrop
from config import Overlay
from config import Subject
from config import Inventory


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
    base_copy_img.filename = backdrop.filename      # type: ignore
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
    angle = random.uniform(overlay.angle_min, overlay.angle_max)
    if random.random() > overlay.flip_probability:
        subject_image = subject_image.transpose(Image.FLIP_LEFT_RIGHT)

    return overlay_images(backdrop_img, subject_image, angle, point,
                          factor_base * factor_position)


def crop_and_scale_image(img: Image.Image, overlay: Overlay):
    """
    Crop and scale the image and return new image.

    - Frist crop to lower aligned, centered bound box of width and height 
      equal to `overlay.output_image_crop_width`
    - Then, if `overlay.output_image_width != overlay.output_image_crop_width`
      resize image to width and height `overlay.output_image_width`
    """
    min_img_dim = min(img.size)
    if min_img_dim < overlay.output_image_crop_width:
        logger.error(f'unable to crop {img.filename}: min dimension {min_img_dim} '   # type: ignore
                     f'< output image width {overlay.output_image_crop_width}')

    width, height = img.size
    left = int(width - overlay.output_image_crop_width) / 2
    upper = height - overlay.output_image_crop_width
    right = left + overlay.output_image_crop_width
    lower = height
    img_mod = img.crop(box=(left, upper, right, lower))

    if overlay.output_image_crop_width != overlay.output_image_width:
        img_mod = img_mod.resize(
            (overlay.output_image_width, overlay.output_image_width))

    return img_mod


def _make_image(idx: int, start_idx: int, zfill_width: int,
                overlay: Overlay, group: SubjectGroup, 
                backdrop_image: ImageFile.ImageFile, 
                backdrop_dpm: float, 
                subjects_by_type: dict[str, list[Subject]],
                image_map: dict[SubjectGroup, list[pathlib.Path]],
                subdir_name: str | None = None):
    """
    Creates the image and writes to file.

    :param subdir_name: if specified will not use `group.subject_type` for the sub-directory name
    """

    image_number = f'{idx + 1 + start_idx}'.zfill(zfill_width)
    if subdir_name is not None and subdir_name != '':
        out_file = OUTPUT_DIR / subdir_name / f'{image_number}-{group.subject_type}.png'
    else:
        out_file = OUTPUT_DIR / group.subject_type / f'{image_number}-{group.subject_type}.png'

    image_map[group].append(out_file)

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

    overlayed_img = crop_and_scale_image(overlayed_img, overlay)
    png_info = PngInfo()
    png_info.add_text('subject', group.subject_type)

    overlayed_img.save(out_file, pnginfo=png_info)
    if Process.verbose:
        logger.info(
            f'{[subj.file.name for subj in subjects]} -> {out_file.name}')


def generate() -> dict[SubjectGroup, list[pathlib.Path]]:
    """
    Generate pictures with random subjects.

    The backdrop is static, subjects are randomly positioned and scaled.
    """

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
    else:
        for file in OUTPUT_DIR.iterdir():
            if file.is_file() and not file.is_dir():
                file.unlink()
            if file.is_dir():
                shutil.rmtree(file)

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
    image_map: dict[SubjectGroup, list[pathlib.Path]] = {}

    for class_name in [g.subject_type for g in overlay.groups] + ['real']:
        class_path = OUTPUT_DIR / class_name
        if not class_path.exists():
            class_path.mkdir(exist_ok=False)

    for group in overlay.groups:
        image_map[group] = []

        if group.subject_type not in subjects_by_type:
            err_msg = f'subject type "{group.subject_type}" not in inventory'
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info(f'{group.count} images of "{group.subject_type}"')

        for idx in range(group.count):
            _make_image(idx, start_idx, zfill_width,
                        overlay,
                        group,
                        backdrop_image,
                        backdrop_dpm,
                        subjects_by_type,
                        image_map)

        start_idx += group.count

    # real images
    if overlay.real_images > 0:
        start_idx = 0
        for idx, group in enumerate(overlay.groups):
            count = int(group.count/total_images * overlay.real_images)
            if idx == len(overlay.groups) - 1:
                # to ensure total number of real images is as specified
                # take the remainder if this is the last iteration
                # this is required because of flooring counts from floating point
                count = overlay.real_images - start_idx
            if count < 1:
                logger.error(f'invalid count {count} for real images {overlay.real_images} '
                             f'of "{group.subject_type}"')
            for img_idx in range(count):
                _make_image(img_idx, start_idx, zfill_width,
                            overlay,
                            group,
                            backdrop_image,
                            backdrop_dpm,
                            subjects_by_type,
                            image_map,
                            subdir_name='real')

            start_idx += count
        logger.info(f'{start_idx} real images of {[g.subject_type for g in overlay.groups]}')

    logger.info('>>>> done generating')
    return image_map


if __name__ == '__main__':
    try:
        generate()
    except ValueError as err:
        print(err)
    except Exception as err:
        print(f'Unexpected error: {err}')
        raise err
