import cv2
import math
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, cpu_count
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def asciify_frame(frame, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness = 2.25, progress_bar = False):
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

    if not progress_bar:
        for i in range(height):
            for j in range(width):
                r, g, b = pix[j, i]
                h = int(0.299 * r + 0.587 * g + 0.114 * b)
                h = min(255, int(h * brightness))

                pix[j, i] = (h, h, h)
                d.text((j * oneCharWidth, i * oneCharHeight), getChar(h), font=fnt, fill=(r, g, b))
    else:
        for i in tqdm(range(height)):
            for j in tqdm(range(width), leave=False):
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

def ascii_photo(in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness= 2.25):
    
    if not os.path.exists(in_path):
        print("Invalid input path.")
        return
    
    cap = cv2.imread(in_path)
    frame_args = (cap, scaleFactor, oneCharWidth, oneCharHeight, brightness, True)
    ascii_frame = Image.fromarray(convert_frame(frame_args),'RGB')
    ascii_frame.save(final_path)


def ascii_video(in_path, final_path, scaleFactor = 0.15, oneCharWidth = 7, oneCharHeight = 9, brightness= 2.25, num_workers=None):
    
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

    for current_frame in tqdm(range(1, frame_count + 1), desc="Processing frames"):
        ret, frame = cap.read()
        if ret:
            frame_args.append((frame, scaleFactor, oneCharWidth, oneCharHeight, brightness))
        else:
            break

    cap.release()

    ascii_frames = list(tqdm(pool.imap(convert_frame, frame_args), total=len(frame_args), desc="Converting frames"))
    pool.close()
    pool.join()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter("temp.mp4", fourcc, frame_rate, (ascii_frames[0].shape[1], ascii_frames[0].shape[0]))

    for frame in tqdm(ascii_frames, desc="Saving video"):
        output_video.write(frame)

    output_video.release()

    final_video = cv2.VideoCapture("temp.mp4")
    audio = cv2.VideoCapture(in_path)
    audio_frames = int(audio.get(cv2.CAP_PROP_FRAME_COUNT))

    final_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    final_output = cv2.VideoWriter(final_path, final_fourcc, frame_rate, (ascii_frames[0].shape[1], ascii_frames[0].shape[0]), isColor=True)

    for _ in tqdm(range(audio_frames), desc="Adding audio"):
        ret, audio_frame = audio.read()
        ret, final_frame = final_video.read()
        final_output.write(final_frame)
    
    final_video.release()
    final_output.release()
    
    os.remove('temp.mp4')

    print("Process completed!")
