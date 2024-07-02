# Asciify
A Python Script to "asciify" photos and videos.

Before Asciify| After Asciify
:-:|:-:
![Pre-Asciify](https://github.com/TheNebulo/Asciify/blob/main/photo_pre.jpg?raw=true) | ![Post-Asciify](https://github.com/TheNebulo/Asciify/blob/main/photo_post.png?raw=true) 



## Installation

Asciify uses a couple dependencies. All of these can be installed via Python's package manager PIP from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Any other dependencies should come standard with Python 3.6+.

## Usage

To asciify a photo, use `ascii_photo()` like so:

```python
def ascii_photo(
        in_path, out_path, scale_factor=0.15, return_to_original_size=False, one_char_width=8, one_char_height=8, color_brightness=1, pixel_brightness=2.15, char_set = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
        monochrome=False, filters=None, overlay_contours=True, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, progress_bar=False
    ):
    """
    Asciifies a photo from a path, and saves it to a file.

    Parameters:
    - in_path: String.
        Relative path to the input image.
    - out_path: String.
        Relative path to the output image. Doesn't need to exist.
    - scale_factor: Float.
        Controls the image quality in the ASCII image.
        Default is 0.15.
    - return_to_original_size: Bool.
        If True, the frame is resized back to original size after conversion (loses quality).
        Default is False.
    - one_char_width: Int.
        Width of one character in the ASCII representation.
        Default is 8.
    - one_char_height: Int.
        Height of one character in the ASCII representation.
        Default is 8.
    - color_brightness: Float.
        Specify brightness multiplier of colors in calculations.
        Default is 1.
    - pixel_brightness: Float.
        Specify brightness multiplier of the drawn pixels in the frame.
        Default is 2.15.
    - char_set: String.
        A string containing all the ASCII/Unicode characters to represent pixels (going from lightest to darkest.)
        Default is a predertimened string for an ASCII set.
    - monochrome: Bool.
        If True, a frame are rendered using only grayscale colors.
        Default is False.
    - filters: Dict.
        A dictionary containing filters to use. crt, sepia, and tint are boolean keys, and tint requires a tint_color key with a color tuple (0-255).
        Default is None.
    - overlay_contours: Bool.
        If True, overlays contours on the image.
        Default is True.
    - contour_depth_minimum_threshold: Float.
        Specify the minimum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 0.
    - contour_depth_maximum_threshold: Float.
        Specify the maximum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 255.
    - progress_bar: Bool.
        If True, displays a progress bar in the console.
        Default is False.
        
    Returns: Void.
    """
```

To asciify a video, use `ascii_video()` like so:

```python
def ascii_video(
        in_path, out_path, scale_factor = 0.15, return_to_original_size=False, one_char_width = 8, one_char_height = 8, color_brightness=1, pixel_brightness=2.15, char_set = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
        monochrome=False, filters=None, overlay_contours = True, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, low_res_audio = True, progress_bar = True, num_workers=None
    ):
    """
    Asciifies a video from a path, and saves it to a file.

    Parameters:
    - in_path: String.
        Relative path to the input video.
    - out_path: String.
        Relative path to the output video. Doesn't need to exist.
    - scale_factor: Float.
        Controls the video quality in the ASCII video.
        Default is 0.15.
    - return_to_original_size: Bool.
        If True, frames is resized back to original size after conversion (loses quality).
        Default is False.
    - one_char_width: Int.
        Width of one character in the ASCII representation.
        Default is 8.
    - one_char_height: Int.
        Height of one character in the ASCII representation.
        Default is 8.
    - color_brightness: Float.
        Specify brightness multiplier of colors in calculations.
        Default is 1.
    - pixel_brightness: Float.
        Specify brightness multiplier of the drawn pixels in the frame.
        Default is 2.15.
    - char_set: String.
        A string containing all the ASCII/Unicode characters to represent pixels (going from lightest to darkest.)
        Default is a predertimened string for an ASCII set.
    - monochrome: Bool.
        If True, frames are rendered using only grayscale colors.
        Default is False.
    - filters: Dict.
        A dictionary containing filters to use. crt, sepia, and tint are boolean keys, and tint requires a tint_color key with a color tuple (0-255).
        Default is None.
    - overlay_contours: Bool.
        If True, overlays contours on the video.
        Default is True.
    - contour_depth_minimum_threshold: Float.
        Specify the minimum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 0.
    - contour_depth_maximum_threshold: Float.
        Specify the maximum threshold of the depth map for when point contours are drawn. Must be between (0-255).
        Default is 255.
    - low_res_audio: Bool.
        If True, audio quality will be lowered to 8Khz to match the video.
        Default is true.
    - progress_bar: Bool.
        If True, displays a progress bar in the console.
        Default is True.
    - num_workers: Int or None.
        Number of concurrent workers for multiprocessing.
        If None, uses the number of system's CPU cores.
        Default is None.
        
    Returns: Void
    """
```

Do not that if `ascii_video()` is used at all (in any script), the script that is initially run must contain the following lines.

```python
from multiprocessing import freeze_support

if __name__ == "__main__":
  freeze_support()
  # Continue the code execution here ( i.e. call main() ).
```

This is only for Windows machines, and is used to prevent subprocesses freezing from new creations.

## License
This project uses the [MIT License](https://choosealicense.com/licenses/mit/).
