"""Preprocess images by finding and removing text boxes and borders."""
import argparse
import cv2
import os
import numpy as np
import scipy
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
from memes.utils import construct_output_filename, DATA_DIR, read_id_to_path, read_hash_to_ids, filename, is_img
import pdb

def find_text(gray, gaussian_size_px=8, text_threshold=10):
    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)
    thresh = cv2.dilate(thresh, None, iterations=8)
    return thresh

def remove_text(gray, img, minpool_size=3):
    is_text = find_text(gray)
    
    text_rows = np.argwhere(is_text.sum(axis=1) > 0).squeeze()
    if len(text_rows) == 0:
        return gray
    isolated_text = img[text_rows]
    # we do this by checking the variance of hue

    is_text[text_rows[(np.argwhere(isolated_text[:,:,1].mean(axis=1) > 10)).squeeze()]] = 0
    
    image_only = np.where(is_text, 0, gray)
    # filter out small bright pixels with convolved minimum
    image_only = scipy.ndimage.minimum_filter(image_only, size=minpool_size)
    return image_only


def find_subimage(img_orig, max_size=300):
    height, width, _ = img_orig.shape
    if min(width, height) > max_size:
        scale_percent = max_size / min(width, height)
        width = int(width * scale_percent)
        height = int(height * scale_percent)
        img_orig = cv2.resize(img_orig, (width, height))

    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
    # check if we need to invert colors:
    # cheeky: we just take an average lightness of the border colors
    # invert colors so white = 0 and black = 255
    avg_lightness = hsv[:,:,2].mean(axis=1)
    std_hue = hsv[:,:,0].std(axis=1)

    blank_indices = np.argwhere(std_hue < 10)
    if blank_indices.shape[0] == 0:
        # no detected border, so we return the original
        return img_orig
    if avg_lightness[np.argwhere(std_hue < 10)].mean() > 112:
        # light background, so we invert
        gray = np.max(gray) - gray

    image_only = remove_text(gray, hsv)

    _, thresh = cv2.threshold(image_only,0,255, cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 2)
    erosion = cv2.dilate(erosion, kernel, iterations = 2)

    coords = cv2.findNonZero(erosion)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_orig[y:y+h, x:x+w]
    return cropped

def process_image(data, out_dir, keep_original=False):
    post_id, path = data
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image {path}")
        return post_id, None
    if keep_original:
        try:
            cv2.imwrite(str(out_dir / f"{filename(path)}-original{os.path.splitext(path)[1]}"), img)
        except:
            print(f"Failed to save original image {path}")
            return post_id, out_path
    cropped = find_subimage(img.copy())
    out_path = out_dir / os.path.basename(path)
    if 0 in cropped.shape:
        # this means we didn't find a subimage, so we return the original
        try:
            cv2.imwrite(str(out_path), img)
            return post_id, out_path
        except:
            print(f"Failed to save original image {path}")
    try:
        cv2.imwrite(str(out_path), cropped)
    except Exception as e:
        print(f"Failed to save cropped image {path}")
    return post_id, out_path


def main(filepaths, hash_file, keep_original, out_dir, num_procs):
    os.makedirs(out_dir, exist_ok=True)
    hash_to_ids = read_hash_to_ids(hash_file)
    ids_to_hash = dict()
    for phash, post_ids in hash_to_ids.items():
        for post_id in post_ids:
            ids_to_hash[post_id] = phash
    id_to_path = read_id_to_path(filepaths)

    def data():
        for post_id, path in id_to_path.items():
            if post_id not in ids_to_hash or not is_img(path):
                continue
            yield post_id, path

    outfilename = construct_output_filename(subdir=DATA_DIR / "filepaths", prefix="processed", suffix="id_to_file", ext="tsv")
    outfile = open(outfilename, "w")

    buf = []
    pool = Pool(num_procs)
    for ind, (post_id, outpath) in tqdm(
        enumerate(pool.imap(partial(process_image, out_dir=out_dir, keep_original=keep_original), data(), chunksize=100)),
        total=len(ids_to_hash),
    ):
        if outpath is not None:
            buf.append(post_id + "\t" + str(outpath) + "\n")
        if ind % 100_000 == 0:
            outfile.write("".join(buf))
            buf = []
    outfile.write("".join(buf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", help="Path to input id-filepath map file")
    parser.add_argument("hash_file", help="Path to hash file")
    parser.add_argument("--keep_original", help="Keep original images", action="store_true", default=False)
    parser.add_argument("--out_dir", help="Path to output directory", default=DATA_DIR / "processed_images")
    parser.add_argument(
        "--num_procs", default=64, type=int, help="Number of processes in pool"
    )

    main(**vars(parser.parse_args()))