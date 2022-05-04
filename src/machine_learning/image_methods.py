import cv2
import numpy as np
import math
import yaml
import random as rd
import matplotlib.pyplot as plt
from PIL import Image

CONFIG_PATH = "../../../../src/config_distance.yml"  # ../chip_img_movement/config_distance.yml"


def overlay_image_alpha(l_img, s_img, x_offset, y_offset):
    if not (l_img.shape[0] > s_img.shape[0] and l_img.shape[1] > s_img.shape[1]):
        print("size error on the input images")
        exit()

    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c]
                                  )
    smallerAlpha = s_img[:, :, 3]
    existingAlpha = l_img[y1:y2, x1:x2, 3]
    l_img[y1:y2, x1:x2, 3] = np.maximum(smallerAlpha, existingAlpha)
    alp = l_img[:, :, 3]
    mask = cv2.resize(alp, [500, 500], interpolation=cv2.INTER_AREA)
    return l_img, mask


def RotateImageSafe(image, rotAngle):
    (h, w) = image.shape[:2]
    hypo = (h * h + w * w) ** 0.5
    canvasWidth = math.ceil(hypo)
    print(canvasWidth)
    canvas = np.zeros((canvasWidth, canvasWidth, 4), dtype="uint8") + 255
    canvas[:, :, 3] = canvas[:, :, 3] * 0

    xoff = math.floor((canvasWidth - w) / 2)
    yoff = math.floor((canvasWidth - h) / 2)
    print(image.shape, canvas.shape)
    (addedIm, a) = overlay_image_alpha(canvas, image, xoff, yoff)
    M = cv2.getRotationMatrix2D((canvasWidth / 2, canvasWidth / 2), rotAngle, 1.0)
    rotated = cv2.warpAffine(addedIm, M, (canvasWidth, canvasWidth))
    return rotated


def warpImage(img, pts=np.float32([[821, 482], [1574, 578], [691, 1950], [1907, 1917], ]), x=512, y=512):
    pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
    mat = cv2.getPerspectiveTransform(pts, pts2)
    out = cv2.warpPerspective(img, mat, (x, y))
    return out


def imageWarpCalibration(image):
    right = 100
    left = 97
    up = 119
    down = 115
    nxt = 32
    kill = 99

    pts = []
    w, h, _ = image.shape
    print(w, h)
    img = image.copy()
    x = 0
    y = 0
    for i in range(4):
        while True:
            cv2.imshow(f"calibrating....(w,d,s,a)", cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA))
            key = cv2.waitKey(1)
            img = image.copy()

            if key == kill:
                return
            elif key == nxt:
                break
            elif key == up:
                y = y - 1 if (y - 1 >= 0) else y
            elif key == down:
                y = y + 1 if (y + 1 <= h) else y
            elif key == left:
                x = x - 1 if (x - 1 >= 0) else x
            elif key == right:
                x = x + 1 if (x + 1 <= w) else x
            img = cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        print("[", x, ",", y, "]")
        pts.append([x, y])
    return np.float32(pts)


def ps_imageWarpCalibration():
    import matplotlib.cbook as cbook

    image_file = cbook.get_sample_data(
        '/Users/paulstephenmalta/Documents/GitHub/NanoView_G33/dev/machine_learning/yoloML/t1.jpg')
    img = plt.imread(image_file)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Plot images to detemine mapping coordinates
    ax1.imshow(img, origin='upper')
    ax2.imshow(img, origin='lower')
    ax3.imshow(img.transpose(1, 0, 2), origin='upper')
    ax4.imshow(img.transpose(1, 0, 2), origin='lower')

    ax3.set_xlabel('upper')
    ax4.set_xlabel('lower')

    ax1.set_ylabel('Not transposed')
    ax3.set_ylabel('Transposed')

    plt.show()

    # All points are in format [cols, rows]
    pt_A = [751, 588]
    pt_B = [1050, 401]
    pt_C = [657, 105]
    pt_D = [464, 160]

    # Here, L2 norm is used to calculate the distances.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Specify image mapping
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Final mapping
    out = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    fig, (output) = plt.subplots(1, 1)
    output.imshow(out, origin='upper')
    plt.show()


def cv2_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def fig_to_cv2(fig):
    # convert canvas to image
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def stitch_image(list_of_image, DEBUG=False):
    row_num = len(list_of_image[0])
    col_num = len(list_of_image)
    img = list_of_image[0][0]
    if DEBUG:
        cv2.imshow("haha", img)
        cv2.waitKey(1000)
    h, w, _ = img.shape
    canvas = cv2.resize(img, (w * row_num, h * col_num))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2RGBA)
    for i, rows in enumerate(list_of_image):
        for j, img in enumerate(rows):
            starth = h * i
            startw = w * j
            rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            canvas, _ = overlay_image_alpha(canvas, rgba, startw, starth)
    return canvas


def Intersecting(x1, y1, w1, h1, x2, y2, w2, h2):

    Xs =   ((x2 + w2) >= x1 >= x2)  or ((x2 + w2) >= (x1+w1)>= x2) \
            or \
            ((x1 + w1) >= x2 >= x1)  or ((x1 + w1) >= (x2+w2)>= x1)

    Ys = ((y2 + h2)>= y1 >= y2) or ((y2 + h2)>=(y1+h1) >= y2) or \
         ((y1 + h1)>= y2 >= y1) or ((y1 + h1)>=(y2+h2) >= y1)
    return (Xs and Ys)



def makeCanvas(w, h, src="",img = None):  # working
    if (src == "" and ( type(img) == type(None))):
        canvas = np.zeros((w, h, 4), dtype="uint8") + 100
        canvas[:, :, 3] = np.zeros((w, h), dtype="uint8") + 50
        return canvas
    elif (not type(img) == type(None)):

        canvas = np.zeros((w, h, 4), dtype="uint8") + 100
        imRead = cv2.resize(img, [w, h], interpolation=cv2.INTER_AREA)
        if imRead.shape[2] == 3:
            canvas[:, :, 0] = imRead[:, :, 0]
            canvas[:, :, 1] = imRead[:, :, 1]
            canvas[:, :, 2] = imRead[:, :, 2]
        else:
            canvas = imRead
        return canvas
    canvas = np.zeros((w, h, 4), dtype="uint8") + 100
    imRead = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    imRead = cv2.resize(imRead, [w, h], interpolation=cv2.INTER_AREA)
    if imRead.shape[2] == 3:
        canvas[:, :, 0] = imRead[:, :, 0]
        canvas[:, :, 1] = imRead[:, :, 1]
        canvas[:, :, 2] = imRead[:, :, 2]
    else:
        canvas = imRead
    return canvas


def blur(img, amt=3):
    return cv2.blur(img, (amt, amt))


def add_noise(img, amt=3, types=1):
    if types == 1:  # "gauss":
        row, col, ch = img.shape
        mean = 0
        var = amt * 5
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = noisy.astype("uint8")
        return noisy
    elif types == 2:  # "s&p"
        row, col, ch = img.shape
        s_vs_p = 0.5
        amount = amt / 1000
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        out[out > 255] = 255
        out[out < 0] = 0
        out = out.astype("uint8")
        return out
    elif types == 3:  # "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals * amt) / float(vals)
        noisy = noisy - np.mean(noisy) / 2
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = noisy.astype("uint8")

        return noisy
    elif types == 4:  # "speckle:
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)

        noisy = img + (img * gauss * (amt) / 10)
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = noisy.astype("uint8")
        return noisy
    print("bad args")


def randomGradientLight(img, amt=3):
    h, w, d = img.shape
    gradient = np.zeros((h, w, d))
    angle = rd.randint(0, 180)

    for i in range(h):
        gradient[i, :, :] = (i / h) * 255
        # gradient[h-1,:,-1] = 1

    gradient = gradient.astype("uint8")

    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1.0)
    rotated = cv2.warpAffine(gradient, rotation_matrix, (h, w))

    biggerGradient = cv2.resize(rotated, [math.ceil(1.5 * h), math.ceil(1.5 * w)], interpolation=cv2.INTER_AREA)
    cropped = biggerGradient[math.ceil(h / 4):h, math.ceil(w / 4):w, :]
    croppedToScale = cv2.resize(cropped, [w, h], interpolation=cv2.INTER_AREA)
    factor = amt / 10
    floatVal = croppedToScale / 255.0
    scaledGray = img * (1 - factor) + img * floatVal * factor + rd.randint(0, 50)
    scaledGray[scaledGray > 255] = 255
    scaledGray[scaledGray < 0] = 0
    final = scaledGray.astype("uint8")
    return final


def randomPass(img):
    amt = rd.randint(0, 10)
    blured = blur(img)
    amt = rd.randint(0, 5)
    noise = add_noise(blured, amt)
    amt = rd.randint(0, 10)
    lit_wierd = randomGradientLight(noise, amt)
    return lit_wierd


def formatImage(img=None, path=None, dest=None, dims=(512, 512)):
    if img is not None:
        pass
    if img is None and (path is not None and dest is not None):
        img = cv2.imread(path)
    if img is None and path is None and dest is None:
        print("No Args")
        return
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    angle = -35
    rot = RotateImageSafe(img, angle)
    warp = warpImage(rot, x=dims[0], y=dims[1])
    adjusted = cv2.cvtColor(warp, cv2.COLOR_RGBA2RGB)
    if path is not None and dest is not None:
        cv2.imwrite(dest, adjusted)
    return adjusted


def white_balance(img):
    if len(img.shape) <= 0:
        raise Exception("No input Image")
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def crop(img, top, bottom, right, left):
    h, w, _ = img.shape
    top_amt = int(h * (top / 100.0))
    bottom_amt = h - int(h * (bottom / 100.0))
    right_amt = int(w * (right / 100.0))
    left_amt = w - int(w * (left / 100.0))
    print(top_amt, bottom_amt, right_amt, left_amt)
    return img[top_amt:bottom_amt, right_amt:left_amt, :]


def increaseContrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    out = [['CLAHE output', cl],

           ]
    return final, out


def make_image_list(fig, img_tuple_list):
    root = int(np.sqrt(len(img_tuple_list))) + 1
    plotSize = 1000
    for i, imgTup in enumerate(img_tuple_list):
        img = imgTup[1]
        largeDim = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]
        scalFac = (plotSize / root) / largeDim
        scaledImg = cv2.resize(img, (int(img.shape[1] * scalFac), int(img.shape[0] * scalFac)))
        fig.add_subplot(root, root, i + 1)
        plt.imshow(scaledImg)
        plt.title(imgTup[0])
        plt.yticks([])
        plt.xticks([])
    return fig


def return_open_cv_img_from_list(img_tuple_list):
    fig = plt.figure()
    fig = make_image_list(fig, img_tuple_list)
    img = fig_to_cv2(fig)
    return img


def showImageList(imgTupleList):
    fig = plt.figure()
    fig = make_image_list(fig, imgTupleList)
    plt.show()


def edge_detection(img):
    im_out = img.copy()
    pre_blur = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    img_blur = cv2.GaussianBlur(pre_blur, (3, 3), 0)

    im_out = cv2.Canny(im_out, 200, 300)
    # Sobel Edge Detection
    # img_blur = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # cv2.imshow("c",cv2.resize(im_out,(500,500)))
    # cv2.imshow("s",cv2.resize(sobelx,(500,500)))
    # cv2.imshow("d",cv2.resize(sobely,(500,500)))
    # cv2.imshow("a",cv2.resize(sobelxy,(500,500)))
    # cv2.waitKey(0)
    out = sobelx[:, :, 0] + sobelx[:, :, 1] + sobelx[:, :, 2]

    out[out > 255] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)

    return out


def green_to_black(img, u_params, l_params):
    output = img.copy()
    u_green = np.array(u_params)
    l_green = np.array(l_params)
    mask = cv2.inRange(output, l_green, u_green)
    res = cv2.bitwise_and(output, output, mask=mask)
    output = output - res
    return output


def green_to_white(img, u_params, l_params):
    img = green_to_black(img, u_params, l_params)
    img[img <= 0] = 255
    return img


def formate_edge_guess(q1, q2, q3, q4):
    # --- 4= |#
    # --- 2= _#
    # --- 6= #|
    # --- 8= #^_
    if q1 and q2 and q3 and q4:
        return 0
    if q3 and not q1 and not q2 and not q4:
        return 1
    if q3 and q4 and not q1 and not q2:
        return 2
    if q4 and not q1 and not q2 and not q3:
        return 3
    if q2 and q3 and not q1 and not q4:
        return 4
    if not q1 and not q2 and not q3 and not q4:
        return 5
    if q2 and not q1 and not q3 and not q4:
        return 7
    if q1 and q2 and not q3 and not q4:
        return 8
    if q1 and not q2 and not q3 and not q4:
        return 9
    return False


def edge_window_x(img, window_length, section_length):
    section_length = int(section_length)
    h, w = img.shape[:2]
    interest_region = img[:, int((w - window_length) / 2):int((w + window_length) / 2)]
    h2, w2 = interest_region.shape[:2]
    value = 0
    final_value = 0
    final_col = 0
    total_windows = window_length / section_length
    for i in range(w - section_length - 1):
        section = interest_region[:, i + (section_length):(i + 1) + (section_length)]
        value = np.mean(section)
        if final_value < value:
            final_value = value
            final_col = i + (w - window_length) / 2

    return final_col


def edge_window_y(img, window_length, section_length):
    section_length = int(section_length)
    h, w = img.shape[:2]
    interest_region = img[int((h - window_length) / 2):int((h + window_length) / 2), :]
    h2, w2 = interest_region.shape[:2]
    value = 0
    final_value = 0
    final_row = 0
    total_windows = window_length / section_length
    for i in range(h2 - section_length - 1):
        section = interest_region[i + section_length:(i + 1) + section_length, :]
        value = np.mean(section)
        if final_value < value:
            final_value = value
            final_row = i + (h - window_length) / 2

    return final_row


# debug methods
def readConfig(filename="config_distance.yml"):
    data = {}
    with open(filename, "r") as stream:
        try:

            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return data


def formate(img):
    if __name__ != "__main__":
        raise "You cant run this in another program"
    config = readConfig(CONFIG_PATH)
    right_round = cv2.rotate(img, cv2.ROTATE_180)
    # contrast,_ =increaseContrast(right_round)
    white_shift = white_balance(right_round)

    cropped = crop(white_shift,
                   config["camera"][0]["image"]["crop"]["top"],
                   config["camera"][0]["image"]["crop"]["bottom"],
                   config["camera"][0]["image"]["crop"]["left"],
                   config["camera"][0]["image"]["crop"]["right"])

    return cropped


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    import time

    print(__file__)
    #    print(os.listdir(CONFIG_PATH))
    time.sleep(2)
    ret, image = cap.read()
    cv2.imshow("frame", formate(image))
    cv2.waitKey(1000)
    print(__file__)
    cap.release()
