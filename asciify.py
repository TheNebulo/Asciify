import cv2
import math
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def asciify_frame(
        frame, scaleFactor=0.15, oneCharWidth=7, oneCharHeight=9, color_brightness=1, pixel_brightness=2.15, monochrome=False, 
        overlay_contours=False, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, progress_bar=False
    ):
    chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
    charArray = list(chars)
    charLength = len(charArray)
    interval = charLength / 256

    contour_chars = "|/-\\_ "
    
    def getChar(inputInt):
        return charArray[math.floor(inputInt * interval)]
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    im = Image.fromarray(frame_rgb)
    fnt = ImageFont.truetype('lucon.ttf', 10)

    width, height = im.size
    im = im.resize((int(scaleFactor * width), int(scaleFactor * height * (oneCharWidth / oneCharHeight))), Image.NEAREST)
    width, height = im.size
    pix = im.load()

    outputImage = Image.new('RGB', (int(oneCharWidth * width), int(oneCharHeight * height)), color=(0, 0, 0))
    d = ImageDraw.Draw(outputImage)

    gray = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)

    depth = cv2.Laplacian(gray, cv2.CV_64F)
    depth = cv2.convertScaleAbs(depth)
    
    _, depth_mask = cv2.threshold(depth, max(0, contour_depth_minimum_threshold), min(255, contour_depth_maximum_threshold), cv2.THRESH_BINARY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(gray, 50, 70)

    for i in tqdm(range(height), disable=not progress_bar, leave=False):
        for j in tqdm(range(width), disable=not progress_bar, leave=False):
            r, g, b = pix[j, i]
            
            mult_r = 0.299 * color_brightness
            mult_g = 0.587 * color_brightness
            mult_b = 0.114 * color_brightness
            
            h = int(mult_r * r + mult_g * g + mult_b * b)
            h = min(255, int(h * pixel_brightness))
            
            char = getChar(h)
            
            if overlay_contours and edges[i, j] > 0 and depth_mask[i, j] > 0:
                angle = cv2.phase(grad_x[i, j], grad_y[i, j], angleInDegrees=True)[0] % 180
                if angle < 45:
                    char = contour_chars[0]  # |
                elif angle < 90:
                    char = contour_chars[1]  # /
                elif angle < 135:
                    char = contour_chars[2]  # -
                else:
                    char = contour_chars[3]  # \
                        
            rgb = (r,g,b)
            
            if monochrome:
                rgb = (h,h,h)

            d.text((j * oneCharWidth, i * oneCharHeight), char, font=fnt, fill=rgb, align="center")
    
    outputImage = outputImage.convert('RGB')
    output_frame = np.array(outputImage)
    return output_frame

def convert_frame(args):
    return asciify_frame(*args)

def ascii_photo(
        in_path, final_path, scaleFactor=0.15, oneCharWidth=7, oneCharHeight=9, color_brightness=1, pixel_brightness=2.15, monochrome=False, 
        overlay_contours=False, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, progress_bar=False
    ):
    """
    Asciifies a photo from a path, and saves it to a file.

    Parameters:
    - in_path: String.
        Relative path to the input image.
    - final_path: String.
        Relative path to the output image. Doesn't need to exist.
    - scaleFactor: Float.
        Controls the image quality in the ASCII image.
        Default is 0.15.
    - oneCharWidth: Int.
        Width of one character in the ASCII representation.
        Default is 7.
    - oneCharHeight: Int.
        Height of one character in the ASCII representation.
        Default is 9.
    - color_brightness: Float.
        Specify brightness multiplier of colors in calculations.
        Default is 1.
    - pixel_brightness: Float.
        Specify brightness multiplier of the drawn pixels in the frame.
        Default is 2.15.
    - monochrome: Bool.
        If True, a frames are rendered using only grayscale colors.
        Default is False.
    - overlay_contours: Bool.
        If True, overlays contours on the image.
        Default is False.
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
    
    if not os.path.exists(in_path):
        print("Invalid input path.")
        return

    cap = cv2.imread(in_path)
    frame_args = (
        cap, scaleFactor, oneCharWidth, oneCharHeight, color_brightness, pixel_brightness, monochrome, 
        overlay_contours, contour_depth_minimum_threshold, contour_depth_maximum_threshold, progress_bar
    )
    ascii_frame = Image.fromarray(convert_frame(frame_args), 'RGB')
    ascii_frame.save(final_path)
    if progress_bar: 
        print(f"Saved asciified photo to {final_path}")


def ascii_video(
        in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, color_brightness=1, pixel_brightness=2.15, monochrome=False, 
        overlay_contours = False, contour_depth_minimum_threshold = 0, contour_depth_maximum_threshold = 255, low_res_audio = True, progress_bar = True, num_workers=None
    ):
    """
    Asciifies a video from a path, and saves it to a file.

    Parameters:
    - in_path: String.
        Relative path to the input video.
    - final_path: String.
        Relative path to the output video. Doesn't need to exist.
    - scaleFactor: Float.
        Controls the video quality in the ASCII video.
        Default is 0.15.
    - oneCharWidth: Int.
        Width of one character in the ASCII representation.
        Default is 7.
    - oneCharHeight: Int.
        Height of one character in the ASCII representation.
        Default is 9.
    - color_brightness: Float.
        Specify brightness multiplier of colors in calculations.
        Default is 1.
    - pixel_brightness: Float.
        Specify brightness multiplier of the drawn pixels in the frame.
        Default is 2.15.
    - monochrome: Bool.
        If True, a frames are rendered using only grayscale colors.
        Default is False.
    - overlay_contours: Bool.
        If True, overlays contours on the video.
        Default is False.
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
    
    if not os.path.exists(in_path):
        print("Invalid input path.")
        return

    cap = cv2.VideoCapture(in_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print('Cap is not open')
        return

    if num_workers is None:
        num_workers = cpu_count()

    pool = Pool(num_workers)
    frame_args = []

    for current_frame in tqdm(range(1, frame_count + 1), desc="Processing frames", disable = not progress_bar, leave=False):
        ret, frame = cap.read()
        if ret:
            frame_args.append((
                frame, scaleFactor, oneCharWidth, oneCharHeight, color_brightness, pixel_brightness, monochrome, 
                overlay_contours, contour_depth_minimum_threshold, contour_depth_maximum_threshold, False
            ))
        else:
            break

    cap.release()

    ascii_frames = list(tqdm(pool.imap(convert_frame, frame_args), total=len(frame_args), desc="Converting frames", disable = not progress_bar, leave=False))
    pool.close()
    pool.join()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter("temp.mp4", fourcc, frame_rate, (ascii_frames[0].shape[1], ascii_frames[0].shape[0]))

    for frame in tqdm(ascii_frames, desc="Saving video", disable = not progress_bar, leave=False):
        output_video.write(frame)

    output_video.release()
    
    original_clip = VideoFileClip(in_path)
    original_audio = original_clip.audio
    
    clip = VideoFileClip("temp.mp4")
    clip = clip.to_RGB()
    fps = clip.fps
        
    if low_res_audio:
        low_audio = original_audio.set_fps(8000)
        low_audio.write_audiofile("temp_audio_low.mp3", verbose=False, logger=None)
        
        audio = AudioFileClip("temp_audio_low.mp3")
        audio = audio.set_fps(16000) 
        audio.write_audiofile("temp_audio.mp3", verbose=False, logger=None)
    else:
        original_audio.write_audiofile("temp_audio.mp3", verbose=False, logger=None)

    with FFMPEG_VideoWriter(final_path, fps=fps, size=clip.size, codec='libx264', logfile=None, threads=num_workers, audiofile="temp_audio.mp3", ffmpeg_params=['-strict', '-2']) as writer:
        frame_iterator = tqdm(clip.iter_frames(fps), total=int(clip.duration*fps), desc = "Adding audio", disable = not progress_bar, leave=False)

        for frame in frame_iterator:
            writer.write_frame(frame)

        writer.close()
        
    clip.close()

    os.remove('temp.mp4')
    if low_res_audio: os.remove('temp_audio_low.mp3')
    os.remove('temp_audio.mp3')
