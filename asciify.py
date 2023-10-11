import cv2
import math
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Pool, freeze_support, cpu_count
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pydub import AudioSegment

def asciify_frame(frame, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness = 2.15, progress_bar = False):
    chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
    charArray = list(chars)
    charLength = len(charArray)
    interval = charLength / 256

    def getChar(inputInt):
        return charArray[math.floor(inputInt * interval)]

    im = Image.fromarray(frame)
    fnt = ImageFont.truetype('C:\\Windows\\Fonts\\lucon.ttf', 15)

    width, height = im.size
    im = im.resize((int(scaleFactor * width), int(scaleFactor * height * (oneCharWidth / oneCharHeight))), Image.NEAREST)
    width, height = im.size
    pix = im.load()

    outputImage = Image.new('RGB', (int(oneCharWidth * width), int(oneCharHeight * height)), color=(0, 0, 0))
    d = ImageDraw.Draw(outputImage)

    for i in tqdm(range(height), disable = not progress_bar, leave=False):
        for j in tqdm(range(width), disable = not progress_bar, leave=False):
            r, g, b = pix[j, i]
            h = int(0.299 * r + 0.587 * g + 0.114 * b)
            h = min(255, int(h * brightness))

            pix[j, i] = (h, h, h)
            d.text((j * oneCharWidth, i * oneCharHeight), getChar(h), font=fnt, fill=(r, g, b))

    outputImage = outputImage.convert('RGB')
    output_frame = np.array(outputImage)
    return output_frame

def convert_frame(args):
    frame, scaleFactor, oneCharWidth, oneCharHeight, brightness, progress_bar = args
    return asciify_frame(frame,  scaleFactor, oneCharWidth, oneCharHeight, brightness, progress_bar)

def ascii_photo(in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness= 2.15, progress_bar = False):
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
    - brightness: Float.
        Specify brightness of the frame.
        Default is 2.15.
    - progress_bar: Bool.
        If True, displays a progress bar in the console.
        Default is False.
        
    Returns: Void.
    """
    
    if not os.path.exists(in_path):
        print("Invalid input path.")
        return
    
    cap = cv2.imread(in_path)
    frame_args = (cap, scaleFactor, oneCharWidth, oneCharHeight, brightness, progress_bar)
    ascii_frame = Image.fromarray(convert_frame(frame_args),'RGB')
    ascii_frame.save(final_path)
    if progress_bar: print(f"Saved asciified photo to {final_path}")


def ascii_video(in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness= 2.15, low_res_audio = True, progress_bar = True, num_workers=None):
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
    - brightness: Float.
        Specify brightness of the frame.
        Default is 2.15.
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
            frame_args.append((frame, scaleFactor, oneCharWidth, oneCharHeight, brightness, False))
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

    if low_res_audio:
        low_audio = AudioSegment.from_file(in_path)
        low_audio = low_audio.set_frame_rate(8000) 
        low_audio.export("temp_audio_low.mp3", format="mp3")
        
        audio = AudioSegment.from_file("temp_audio_low.mp3")
        audio = audio.set_frame_rate(16000) 
        audio.export("temp_audio.mp3", format="mp3")
    else:
        audio = AudioSegment.from_file(in_path)
        audio.export("temp_audio.mp3", format="mp3")

    clip = VideoFileClip("temp.mp4")
    clip = clip.to_RGB()
    fps = clip.fps

    with FFMPEG_VideoWriter(final_path, fps=fps, size=clip.size, codec='libx264', logfile=None, threads=num_workers, audiofile="temp_audio.mp3", ffmpeg_params=['-strict', '-2']) as writer:
        frame_iterator = tqdm(clip.iter_frames(fps), total=int(clip.duration*fps), desc = "Adding audio", disable = not progress_bar, leave=False)

        for frame in frame_iterator:
            writer.write_frame(frame)

        writer.close()
        
    clip.close()

    os.remove('temp.mp4')
    if low_res_audio: os.remove('temp_audio_low.mp3')
    os.remove('temp_audio.mp3')
