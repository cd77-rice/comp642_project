import os

# Remove these os.environ lines if CUDA and/or ROCM work on your machine
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

from config import Training
from config import Overlay
from config import Plot
from config import OUTPUT_DIR
from config import LOG_DIR
from config import PLOT_DIR
from config import logger

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.math import confusion_matrix
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow import version


def load_dataset(directory: Path) -> Dataset:
    """
    Load images into tensorflow dataset and return a tensorflow dataset.
    """
    overlay = Overlay()
    logger.info('>>>> loading images into dataset')
    class_names = [g.subject_type for g in overlay.groups]
    img_width = overlay.output_image_width
    data = image_dataset_from_directory(directory, 
                                        class_names=class_names, 
                                        image_size=(img_width, img_width),
                                        shuffle=True)
    # there is not simple function to show the total number of 
    # datapoints in a tensroflow dataset, thus only batch related counts
    logger.info(f'total data points : {sum(g.count for g in overlay.groups)}')
    logger.info(f'number of batches : {len(data)}')
    logger.info(f'number of classes : {len(data.class_names)}')
    logger.info(f'classes           : {data.class_names}')
    logger.info('<<<< done loading dataset')
    return data


def preprocess_dataset(data: Dataset) -> Dataset:
    """
    Apply preprocessing to dataset, like normalization from max `255` to `1.0`.
    """
    return data.map(lambda x,y: (x / 255.0, y))


def split_dataset(data) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split the tensorflow dataset per configuration.

    This will create tensorflow datasets and return as tuple of (training, validation, test) sets.
    """
    logger.info('>>>> splitting training, validation & test set')
    train_set_size = int(Training.fraction_training_set * len(data))
    validation_set_size = int(Training.fraction_validation_set * len(data))
    test_set_size = len(data) - train_set_size - validation_set_size

    train_set = data.take(train_set_size)
    validation_set = data.skip(train_set_size).take(validation_set_size)
    test_set = data.skip(train_set_size + validation_set_size).take(test_set_size)

    logger.info('Set sizes in number of batches')
    logger.info(f'train set size      : {len(train_set)}')
    logger.info(f'validation set size : {len(validation_set)}')
    logger.info(f'test set size       : {len(test_set)}')
    logger.info(f'total set size      : {len(data)}')
    logger.info('<<<< done splitting')

    return train_set, validation_set, test_set


def cnn_model1(shape: tuple[int, int, int]):
    """
    CNN1 architecture.
    """
    name = "CNN1"
    logger.info(f'>>>> Creating {name}')
    logger.info(cnn_model1.__doc__)
    model = Sequential(
        [
            Input(shape=shape),
            Conv2D(16, (3,3), activation='relu', padding='valid'),
            MaxPool2D(2,2),
            Conv2D(32, (3,3), activation='relu', padding='valid'),
            MaxPool2D(2,2),
            Conv2D(64, (3,3), activation='relu', padding='valid'),
            MaxPool2D(2,2),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name=name.lower()
    )
    logger.info(model.summary())
    logger.info(f'<<<< done creating {name}')
    return model


def cnn_model2(shape: tuple[int, int, int]):
    """
    CNN2 architecture.
    """
    name = "CNN2"
    logger.info(f'>>>> Creating {name}')
    logger.info(cnn_model2.__doc__)

    model = Sequential(
        [
            Input(shape=shape),
            Conv2D(32, (3,3), activation='relu', padding='valid'),
            MaxPool2D(2,2),
            Conv2D(64, (3,3), activation='relu', padding='valid'),
            MaxPool2D(2,2),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name=name.lower()
    )
    logger.info(model.summary())
    logger.info(f'<<<< done creating {name}')
    return model


def cnn_model3(shape: tuple[int, int, int]):
    """
    CNN model 3 architecture.
    """
    name = "CNN3"
    logger.info(f'>>>> Creating {name}')

    model = Sequential(
        [
            Input(shape=shape),
            Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='valid'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name=name.lower()
    )
    logger.info(model.summary())
    logger.info(f'<<<< done creating {name}')
    return model


def train(model: Model, train_set: Dataset, validation_set: Dataset) -> History:
    """
    Train model using the provided training and validation set.

    :returns: history of training process
    """
    logger.info(f'>>>> training {model.name.upper()}')
    logger.info('Set sizes in number of batches')
    logger.info(f'train set size      : {len(train_set)}')
    logger.info(f'validation set size : {len(validation_set)}')

    logger.info(f'compile {model.name}')
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    logger.info(f'train model & validate, running {Training.epochs} epochs total')
    logger.info(f'logging TensorBoard to {LOG_DIR.absolute()}')
    callbacks = [TensorBoard(log_dir=LOG_DIR)]
    if Training.early_stopping:
        logger.info(f'applying early stopping: '
                    f'on {Training.early_stopping_monitor.value}, '
                    f'patience {Training.early_stopping_patience}, '
                    f'restore best weights {Training.early_stopping_restore_best}')
        es = EarlyStopping(monitor=Training.early_stopping_monitor.value, 
                           patience=Training.early_stopping_patience, 
                           restore_best_weights=Training.early_stopping_restore_best)
        callbacks.append(es)
    else:
        logger.info('no early stopping applied')
    history = model.fit(train_set,
                        validation_data=validation_set, 
                        epochs=Training.epochs,
                        callbacks=callbacks)
    logger.info(f'<<<< done training')
    return history


def test(model: Sequential, test_set: Dataset):
    """
    Test the model
    """
    logger.info(f'>>>> testing model {model.name}')

    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    labels = []
    predicted = []
    for X, y in test_set.as_numpy_iterator():
        y_ = model.predict(X)
        labels += list(y)
        predicted += list(y_)
        precision.update_state(y, y_)
        recall.update_state(y, y_)
        accuracy.update_state(y, y_)

    precision_value = precision.result().numpy()
    recall_value = recall.result().numpy()
    accuracy_value = accuracy.result().numpy()
    logger.info(f'precision : {precision.result().numpy()}')
    logger.info(f'recall    : {recall.result().numpy()}')
    logger.info(f'accuracy  : {accuracy.result().numpy()}')

    result_df = pd.DataFrame(
        [[precision_value, recall_value, accuracy_value]],
        columns=['precision', 'recall', 'accuracy'],
        index=['value']
    )
    result_file = PLOT_DIR / f'{model.name}_results.md'
    result_df.to_markdown(result_file)
    logger.info(f'wrote test results to {result_file}')

    conf_matrix = confusion_matrix(labels, [v > 0.5 for v in predicted])
    logger.info(f'confusion matrix {confusion_matrix}')
    logger.info(f'<<<< done testing {model.name}')
    return conf_matrix


def plot(history: History, file: Path):
    """
    Plot `history` of training process and write to `file`.
    """

    logger.info('>>>> plotting history of training process')

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss & Accuracy {history.model.name.upper()}')
    x = [i + 1 for i in range(len(history.history['loss']))]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    lines = []
    lines += ax.plot(x, history.history['loss'], 
                     label='training loss', 
                     color='blue')
    lines += ax.plot(x, history.history['val_loss'],   
                     label='validation loss', 
                     color='green')

    ax_ = ax.twinx()
    ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
    lines += ax_.plot(x, history.history['accuracy'], 
                      label='training acc', 
                      color='lightgrey', 
                      linestyle='dashed')
    lines += ax_.plot(x, history.history['val_accuracy'], 
                      label='validation acc', 
                      color='black', 
                      linestyle='dashed')
    ax_.set_ylabel('Accuracy')

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    fig.tight_layout()

    file_ = file.resolve()
    if not file_.parent.exists():
        file_.parent.mkdir()
        logger.info(f'created output directory {file_.parent.absolute()}')
    fig.savefig(file_)
    logger.info(f'saved plot to {file_.absolute()}')

    logger.info('<<<< done plotting')


def plot_conf_matrix(history: History, conf_matrix: Tensor, file: Path):
    """
    Plot the confusion matrix and write to `file`.
    """
    plot_labels = ['positive', 'negative']
    cm_df = pd.DataFrame(conf_matrix, index=plot_labels, columns=plot_labels)
    plt.figure(figsize = (3, 3))
    plt.title(f'Confusion Matrix {history.model.name.upper()}')
    sn.heatmap(cm_df, annot=True, fmt='d', cbar=False)
    plt.xlabel('PREDICTED')
    plt.ylabel('ACTUAL')
    plt.tight_layout()
    plt.savefig(file)


def predict_real_images(model: Sequential, classes: list[str], plot_file: Path):
    """
    Predict using real images for visual inspection and plot result to `plot_file`.
    """
    image_dir = OUTPUT_DIR / 'real'
    fig, ax = plt.subplots(Plot.sample_grid_row_size, 
                           Plot.sample_grid_column_size)
    fig.suptitle(f'{model.name} Sample Classifications')

    for idx, file in enumerate(image_dir.iterdir()):
        if idx >= Plot.sample_grid_row_size * Plot.sample_grid_column_size:
            break;
        if file.suffix.lower() != '.png':
            continue

        img = Image.open(file)
        y = int(img.info['subject'] == 'human')
        x = np.asarray(img)

        y_ = model.predict(np.expand_dims(x/255.0, 0))
        class_idx = 1 if y_ > 0.5 else 0
        indicator = '+' if class_idx == y else '-'

        plot_row = idx // Plot.sample_grid_column_size
        plot_col = idx % Plot.sample_grid_column_size
        ax[plot_row, plot_col].imshow(img)
        ax[plot_row, plot_col].set_title(f'{classes[class_idx]} {indicator}', fontsize=10)
        ax[plot_row, plot_col].xaxis.set_visible(False)
        ax[plot_row, plot_col].yaxis.set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_file)


def write_model_summary(model: Sequential):
    """
    Writes the model summary to a file in PLOT_DIR.
    """

    file = PLOT_DIR / f'{model.name}_summary.txt'
    if not PLOT_DIR.is_dir():
        PLOT_DIR.mkdir()
    if file.exists():
        file.unlink()
    file.touch()
    def myprint(s):
        with open(file, 'a', encoding='utf-8') as f:
            print(s, file=f)

    model.summary(print_fn=myprint)


def run():
    logger.info('########################################')
    logger.info('             >>>> START <<<<')
    logger.info('########################################')
    logger.info(f'Tensorflow version {version.VERSION}')

    data = load_dataset(OUTPUT_DIR)
    class_names = data.class_names
    shape = data.element_spec[0].shape[1:]
    data = preprocess_dataset(data)
    train_set, val_set, test_set = split_dataset(data)

    model_functions = [cnn_model1, cnn_model2, cnn_model3]
    logger.info(f'Processing {len(model_functions)} models / architectures')
    for func in model_functions:
        model = func(shape)
        write_model_summary(model)

        history = train(model, train_set, val_set)
        conf_matrix = test(model, test_set)

        plot(history, PLOT_DIR / f'{model.name}.png')
        plot_conf_matrix(history, conf_matrix, 
                         PLOT_DIR / f'{model.name}_conf_matrix.png')
        predict_real_images(model, class_names,
                            PLOT_DIR / f'{model.name}_predict_real.png')

    logger.info('########################################')
    logger.info('             <<<< done <<<<')
    logger.info('########################################')


if __name__ == '__main__':
    run()
