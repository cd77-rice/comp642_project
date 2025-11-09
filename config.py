import pathlib
from dataclasses import dataclass

from numpy.lib import angle


IMG_DIR = pathlib.Path('./images')
OUTPUT_DIR = pathlib.Path('./generated')


@dataclass(frozen=True)
class Point:
    """
    Point on image in pixels.

    CSYS start bottom, left corner: x pointing left, y pointing up
    """
    x: int
    y: int


class Backdrop:
    """
    Backdrop image.
    """
    file: pathlib.Path = IMG_DIR / 'trail1_color1.png'
    height: float = 2.5


@dataclass(frozen=True)
class Subject:
    """
    A subject to be placed on the backdrop.
    """
    file: pathlib.Path
    height: float
    type_: str


@dataclass(frozen=True)
class SubjectGroup:
    """
    Group of subjects.
    """
    subject_type: str
    count: int


class Overlay:
    """
    Definition of the overlay parameters.
    """
    bound_box = Point(360, 20), Point(500, 270)
    front_factor = 1.0
    back_factor = 0.35
    flip_probability = 0.5
    angle_min = -1.0
    angle_max = 1.0
    groups: list[SubjectGroup] = [
        SubjectGroup('animal', 10),
        SubjectGroup('human', 45)
    ]
    output_image_width = 256


class Inventory:
    """
    Inventory of subjects to select from.
    """
    subjects = [
        Subject(IMG_DIR / 'bear_black_front.png',  1.2, 'animal'),
        Subject(IMG_DIR / 'bob_cat_front1.png', 0.65, 'animal'),
        Subject(IMG_DIR / 'bob_cat_side.png', 0.65, 'animal'),
        Subject(IMG_DIR / 'bobcat.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'boy_walking.png', 1.75, 'human'),
        Subject(IMG_DIR / 'cyote_back1.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'cyote_back2.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'cyote_front.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'cyote_front1.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'cyote_front2.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'deer_back.png', 1.4, 'animal'),
        Subject(IMG_DIR / 'deer_back1.png', 1.3, 'animal'),
        Subject(IMG_DIR / 'deer_side.png', 1.3, 'animal'),
        Subject(IMG_DIR / 'grey_wolf_back.png', 0.8, 'animal'),
        Subject(IMG_DIR / 'male_elk_front.png', 2.5, 'animal'),
        Subject(IMG_DIR / 'man_walking.png', 1.9, 'human'),
        Subject(IMG_DIR / 'hiker_back1.png', 1.65, 'human'),
        Subject(IMG_DIR / 'hiker_back2.png', 1.7, 'human'),
        Subject(IMG_DIR / 'person_back.png', 1.75, 'human'),
        Subject(IMG_DIR / 'person_back4.png', 1.65, 'human'),
        Subject(IMG_DIR / 'person_front.png', 1.8, 'human'),
        Subject(IMG_DIR / 'person_front1.png', 1.8, 'human'),
        Subject(IMG_DIR / 'person_front2.png', 1.87, 'human'),
        Subject(IMG_DIR / 'person_front3.png', 1.75, 'human'),
        Subject(IMG_DIR / 'person_front5.png', 1.65, 'human'),
        Subject(IMG_DIR / 'person_dog_front.png', 1.65, 'human')
    ]

