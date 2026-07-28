import os
import collections
import pandas as pd
import numpy as np
import skimage.draw
import tensorflow as tf
import cv2

class EchoTF:
    """
    TensorFlow version of EchoNet-Dynamic dataset.
    Returns (video, target) as TensorFlow tensors.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):



        self.root = root
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            data = pd.read_csv(os.path.join(self.root, "FileList.csv"))
            data["Split"] = data["Split"].map(lambda x: x.upper())
            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = [fn + ".avi" if os.path.splitext(fn)[1] == "" else fn
                           for fn in data["FileName"].tolist()]
            self.outcome = data.values.tolist()

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(lambda: collections.defaultdict(list))

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))

            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # Drop videos with < 2 traced frames
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __len__(self):
        return len(self.fnames)

    def _load_item(self, index):
        # --- VIDEO PATH ---
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        # --- LOAD VIDEO ---
        video = loadvideo(video_path).astype(np.float32)

        # --- Add noise ---
        if self.noise is not None:
            n = np.prod(video.shape[1:])
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # --- Normalize ---
        '''
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)
        '''
        video=video/255
        # --- Frame selection ---
        c, f, h, w = video.shape
        length = f // self.period if self.length is None else self.length
        if self.max_length is not None:
            length = min(length, self.max_length)
        if f < length * self.period:
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1
            )
            c, f, h, w = video.shape

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # --- Targets ---
        target = []
        key = self.fnames[index]
        for t in self.target_type:
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                trace = self.trace[key][self.frames[key][-1 if t == "LargeTrace" else 0]]
                x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                r, c_ = skimage.draw.polygon(np.rint(y).astype(np.int32),
                                             np.rint(x).astype(np.int32),
                                             (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c_] = 1
                target.append(mask)
            else:
                target.append(np.float32(self.outcome[index][self.header.index(t)]))
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        # --- Select clips ---
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        # --- Padding ---
        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        # Convert to (L, H, W, C) for TF
        video = np.transpose(video[0:1], (1, 2, 3, 0))
        return video, target/100

    def get_tf_dataset(self, batch_size=50, shuffle=False):
        def gen():
            for i in range(len(self)):
                yield self._load_item(i)

        output_signature = (
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)  # adjust if multi-target
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))

    return v