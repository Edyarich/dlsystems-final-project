import numpy as np
from .autograd import Tensor
import os
import pickle
import struct
import gzip
import cv2
from typing import Iterator, Optional, List, Tuple, Sized, Union, Iterable, Any, \
    Callable
from needle import backend_ndarray as nd
from needle.backend_ndarray.ndarray import BackendDevice


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p

        if flip_img:
            return img[:, ::-1, :]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
            img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding,
                                             high=self.padding + 1, size=2)
        padded_img = np.pad(img,
                            ((self.padding, self.padding),
                             (self.padding, self.padding),
                             (0, 0)),
                            constant_values=0)

        x_start = shift_x + self.padding
        y_start = shift_y + self.padding
        height, width, _ = img.shape

        return padded_img[x_start:x_start + height, y_start:y_start + width, :]


class Lambda(Transform):
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, img):
        return self.func(img)


class Resize(Transform):
    def __init__(self, sizes: Tuple[int]):
        self.height, self.width = sizes

    def __call__(self, img, **cv2_resize_kwargs):
        """
        Resize image
        Args:
            img: H x W x C np.ndarray of an image
        Return:
            self.height x self.width x C NDArray of resized image
        """
        resized_img = cv2.resize(
            img,
            (self.height, self.width),
            **cv2_resize_kwargs
        )
        return resized_img


class ToTensor(Transform):
    def __init__(self, device: Optional[BackendDevice] = None,
                 dtype: str = "float32", requires_grad: bool = False):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

    def __call__(self, img):
        """
        type(img): NDArray ==> Tensor
        Args:
            img: H x W x C NDArray (or np.ndarray) of an image
        Return:
            H x W x C Tensor of an image
        """
        return Tensor(img, device=self.device, dtype=self.dtype,
                      requires_grad=self.requires_grad)


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
            self,
            dataset: Dataset,
            batch_size: Optional[int] = 1,
            shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        if not self.shuffle:
            indices = np.arange(len(dataset))
            self.ordering = self._get_order(indices)
        else:
            indices = np.random.permutation(len(dataset))
            self.ordering = self._get_order(indices)

    def _get_order(self, indices):
        return np.array_split(indices,
                              range(self.batch_size,
                                    len(indices),
                                    self.batch_size))

    def __iter__(self):
        for ind_batch in self.ordering:
            batch = [self.dataset[ind] for ind in ind_batch]
            k_outputs = len(self.dataset[0])

            yield [Tensor([val[k] for val in batch]) for k in range(k_outputs)]

        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
            self.ordering = self._get_order(indices)

        return self

    def __next__(self):
        pass

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MNISTDataset(Dataset):
    def __init__(
            self,
            image_filename: str,
            label_filename: str,
            transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.images = None
        self.labels = None

        self._parse_mnist(image_filename, label_filename)

    def _parse_mnist(self, image_filename: str, label_filename: str) -> None:
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 4D numpy array containing the loaded
                    data.  The dimensionality of the data should be
                    (num_examples x height x width x num_channels) where 'input_dim' is the full
                    dimension of the data, e.g., since MNIST images are 28x28, it
                    will be 784.  Values should be of type np.float32, and the data
                    should be normalized to have a minimum value of 0.0 and a
                    maximum value of 1.0.

                y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.int8 and
                    for MNIST will contain the values 0-9.
        """
        with gzip.open(image_filename, 'rb') as fd:
            samples = fd.read()

        with gzip.open(label_filename, 'rb') as fd:
            labels = fd.read()

        _, imgs_cnt, rows_cnt, cols_cnt = struct.unpack('>iiii', samples[:16])
        _, labels_cnt = struct.unpack('>ii', labels[:8])

        assert imgs_cnt == labels_cnt

        decoded_samples = struct.unpack(f'>{imgs_cnt * rows_cnt * cols_cnt}B',
                                        samples[16:])
        decoded_labels = struct.unpack(f'>{labels_cnt}B', labels[8:])

        samples = np.array(decoded_samples, dtype=np.float32)
        samples = samples.reshape(imgs_cnt, rows_cnt, cols_cnt, 1)
        samples -= samples.min()
        samples /= (samples.max() - samples.min())

        self.images = samples
        self.labels = np.array(decoded_labels, dtype=np.uint8)

    def __getitem__(self, index) -> object:
        img, label = self.images[index], self.labels[index]
        transformed_img = self.apply_transforms(img)

        return transformed_img, label

    def __len__(self) -> int:
        return len(self.labels)


class CIFAR10Dataset(Dataset):
    def __init__(
            self,
            base_folder: str,
            train: bool,
            transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        images - numpy array of images
        labels - numpy array of labels
        """
        super().__init__(transforms)
        images, labels = self._parse_cifar(base_folder, train)
        self.images = images
        self.labels = labels

    def _parse_cifar(self, base_folder: str, train: bool):
        batch_names = [f'data_batch_{i}' for i in range(1, 6)] if train \
            else ['test_batch']

        images, labels = [], []
        for batch_name in batch_names:
            filename = base_folder + '/' + batch_name

            with open(filename, 'rb') as fd:
                dct = pickle.load(fd, encoding='bytes')

            images.append(dct[b'data'])
            labels.append(dct[b'labels'])

        np_images = np.array(images, dtype=np.float32) / 255
        np_images = np_images.reshape(-1, 3, 32, 32)
        np_labels = np.array(labels).reshape(-1)

        return np_images, np_labels

    def __getitem__(self, index: int) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        if index >= len(self):
            raise IndexError('CIFAR10 index out of range')

        return self.apply_transforms(self.images[index]), self.labels[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.labels.size


class LandscapesDataset(Dataset):
    def __init__(self, files: List[str], img_size: int = 256,
                 extra_transforms: Optional[List] = None):
        basic_transforms = [
            Lambda(lambda img: img.astype(np.float32)),
            Lambda(lambda img: img / 255 * 2 - 1),
            Resize((img_size, img_size))
        ]
        if extra_transforms is not None:
            basic_transforms += extra_transforms

        super().__init__(basic_transforms)
        self.files = sorted(files)

    def load_sample(self, filename: str) -> np.ndarray:
        img = cv2.imread(filename)
        return img

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('LandscapesDataset index out of range')

        return self.apply_transforms(self.load_sample(self.files[index]))

    def __len__(self):
        return len(self.files)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.idx2word)


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    EOS_TOKEN = '<eos>'

    def __init__(self, base_dir: str, max_lines: Optional[int] = None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'),
                                   max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path: str, max_lines: Optional[int] = None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        max_lines = float('inf') if max_lines is None else max_lines
        eos_id = self.dictionary.add_word(self.EOS_TOKEN)

        with open(path, 'r') as fd:
            for i, sent in enumerate(fd):
                if i >= max_lines:
                    break

                for word in sent.strip().split():
                    word_id = self.dictionary.add_word(word)
                    ids.append(word_id)

                ids.append(eos_id)

        return ids


def batchify(data: list, batch_size: int, device: BackendDevice, dtype: type):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    k_batches = len(data) // batch_size
    data = data[:k_batches * batch_size]

    return np.array(
        [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    )


def get_batch(batches: np.ndarray, i: int, bptt: int,
              device: BackendDevice = None, dtype: type = None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    return Tensor(batches[i:i + bptt], device=device, dtype=dtype), \
           Tensor(batches[i + 1:i + bptt + 1].reshape(-1), device=device,
                  dtype=dtype)
