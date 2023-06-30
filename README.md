# Asciify
A Python Script to "asciify" photos and videos.

Before Asciify| After Asciify
:-:|:-:
![Pre-Asciify](https://github.com/TheNebulo/Asciify/blob/main/photo_pre.jpg?raw=true) | ![Post-Asciify](https://github.com/TheNebulo/Asciify/blob/main/photo_post.png?raw=true) 



## Installation

Asciify uses a couple dependencies. All of these can be installed via Python's package manager PIP.

```bash
pip install opencv-python
pip install Pillow
pip install Numpy
```

Any other dependencies should come standard with Python 3.6+.

## Usage

To asciify a photo, use `ascii_photo()` like so:

```python
ascii_photo(in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness= 2.25)
"""
in_path: The path of the photo to asciify (Required).
final_path: The path where the asciified photo will be saved (Required).
scaleFactor: The scale multiplier at which the frame is processed. Defaults to 0.15. (Optional)
oneCharWidth: How wide a character should be in pixels. Defaults to 7. (Optional)
oneCharHeight: How tall a character should be in pixels. Defaults to 9. (Optional)
brightness: How bright the image should be. Defaults to 2.25. (Optional)
"""
```

To asciify a video, use `ascii_video()` like so:

```python
ascii_video(in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness= 2.25, num_workers = None)
"""
in_path: The path of the video to asciify (Required).
final_path: The path where the asciified video will be saved (Required).
scaleFactor: The scale multiplier at which the frames are processed. Defaults to 0.15. (Optional)
oneCharWidth: How wide a character should be in pixels. Defaults to 7. (Optional)
oneCharHeight: How tall a character should be in pixels. Defaults to 9. (Optional)
brightness: How bright the image should be. Defaults to 2.25. (Optional)
num_workers = The amount of subprocess workers work on processing. Defaults to the amount of CPU cores in the system. (Optional)
"""
```

Do note that in the script that is run first (`__name__ == "__main__"`), if `ascii_video()` is used at all (in any script), the following lines should be added.

```python
from multiprocessing import freeze_support

if __name__ == "__main__":
  freeze_support()
```

This is only for Windows machines, and is used to prevent subprocesses freezing from new creations.

## License
This project uses the [MIT License](https://choosealicense.com/licenses/mit/).
