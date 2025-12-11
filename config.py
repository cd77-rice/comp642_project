import pathlib
import logging
import sys
from dataclasses import dataclass

from tensorflow.python.framework.ops import enum


logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_stream_handler = logging.StreamHandler(sys.stdout)
stdout_stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stdout_stream_handler)

IMG_DIR = pathlib.Path('./images')
OUTPUT_DIR = pathlib.Path('./generated')
LOG_DIR = pathlib.Path('./logs')
PLOT_DIR = pathlib.Path('./plots')


@dataclass(frozen=True)
class Process:
    """
    Processing configuration
    """
    verbose: bool = False
    '''True = verbose output, False = non-verbose output'''


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
    '''File path to the backdrop image.'''

    height: float = 2.5
    '''Physical or true height of the backdrop, e.g. 2.5m.
    The unit must be consistent with `Subject.height`.'''


@dataclass(frozen=True)
class Subject:
    """
    A subject to be placed on the backdrop.
    """
    file: pathlib.Path
    '''File path to subject image file. 
    This should be a PNG with transparent background.'''

    height: float
    '''True or physical height of the subject, e.g. 1.8m (m = meter).
    The unit of this value must be consistent with `Backdrop.height`.
    '''

    type_: str
    '''Type name of the subject. 
    This must be consistent with `SubjectGroup.subject_type` and `Inventory`.'''


@dataclass(frozen=True)
class SubjectGroup:
    """
    Group of subjects.
    """

    subject_type: str
    '''Subject type which will be the label for that class in the model.'''

    count: int
    '''Number of images to generate for the group / class.'''


class Overlay:
    """
    Definition of the overlay parameters.
    """

    bound_box = Point(360, 20), Point(500, 270)
    '''Bounding box for positioning of subjects'''

    front_factor = 1.0
    '''Subject scale factor at front (lower edge of bounding box).'''

    back_factor = 0.35
    '''Subject scale factor at back (upper edge of bounding box).'''

    flip_probability = 0.5
    '''Probability to flip subject image.'''

    angle_min = -1.0
    '''Lower range of random angle [rad] for subject rotation.'''

    angle_max = 1.0
    '''Upper range of random angle [rad] for subject rotation.'''

    groups: list[SubjectGroup] = [
        SubjectGroup('animal', 400),
        SubjectGroup('human', 600)
    ]

    real_images = 20
    '''Number of "real" images (as taken by a camera) to be generated. 
    These images are neither in training, validation nor test set.
    The split of classes will be roughly (due to rounding error) the same 
    as defined by `groups`.
    '''

    output_image_crop_width = 512
    '''Width & height of center cropping area including the 
    bounding box with subject.'''

    output_image_width = 244
    '''Final image width & height, scaled to after cropping of the image.'''


class Inventory:
    """
    Inventory of subjects to select from.
    """
    subjects = [
        Subject(IMG_DIR / 'bear_black_front.png',  1.2, 'animal'),
        Subject(IMG_DIR / 'bobcat.png', 0.7, 'animal'),
        Subject(IMG_DIR / 'bob_cat_front1.png', 0.65, 'animal'),
        Subject(IMG_DIR / 'bob_cat_side.png', 0.65, 'animal'),
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
        Subject(IMG_DIR / 'hiker_back1.png', 1.65, 'human'),
        Subject(IMG_DIR / 'hiker_back2.png', 1.7, 'human'),
        Subject(IMG_DIR / 'male_elk_front.png', 2.5, 'animal'),
        Subject(IMG_DIR / 'man_walking.png', 1.9, 'human'),
        Subject(IMG_DIR / 'person_back.png', 1.75, 'human'),
        Subject(IMG_DIR / 'person_back3.png', 1.8, 'human'),
        Subject(IMG_DIR / 'person_back4.png', 1.65, 'human'),
        Subject(IMG_DIR / 'person_dog_front.png', 1.65, 'human'),
        Subject(IMG_DIR / 'person_front.png', 1.8, 'human'),
        Subject(IMG_DIR / 'person_front1.png', 1.8, 'human'),
        Subject(IMG_DIR / 'person_front2.png', 1.87, 'human'),
        Subject(IMG_DIR / 'person_front3.png', 1.75, 'human'),
        Subject(IMG_DIR / 'person_front5.png', 1.75, 'human'),
        Subject(IMG_DIR / 'person_front6.png', 1.8, 'human'),
        Subject(IMG_DIR / 'person_front7.png', 1.75, 'human'),
        Subject(IMG_DIR / 'person_front8.png', 1.8, 'human')
    ]


class Training:
    """
    Configuration of training runs
    """
    class EarlyStopMonitor(enum.Enum):
        LOSS = 'val_loss'
        ACCURACY = 'val_accuracy'

    fraction_training_set = 0.6
    '''Fraction of the training set from complete dataset. The test set
    fraction will be `1 - fraction_training_set - fraction_validation_set`.
    Therefore `fraction_training_set + fraction_validation_set` should be < 1.0'''

    fraction_validation_set = 0.2
    '''Fraction of the validation set from complete dataset. The test set 
    fraction will be `1 - fraction_training_set - fraction_validation_set`.
    Therefore `fraction_training_set + fraction_validation_set` should be < 1.0'''

    tolerance_class_ratio = 1e-2
    epochs = 50
    '''Number of epoch to train, if early stopping is not applied.'''

    early_stopping = True
    '''`True` = apply early stopping, `False` otherwise.'''

    early_stopping_monitor = EarlyStopMonitor.LOSS
    '''Property to be monitored for early stopping.'''

    early_stopping_patience = 5
    '''Patience for early stopping, 
    see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping'''

    early_stopping_restore_best = True
    '''Whether to restore weights from best epoch,
    see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping'''


class Plot:
    """
    Plot settings
    """

    sample_grid_row_size = 3
    '''Number of rows for plot of classification samples.'''

    sample_grid_column_size = 4
    '''Number of columns for plot of classification samples.'''

