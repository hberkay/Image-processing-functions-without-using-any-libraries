import math
import numpy
from tkinter import *
from PIL import Image
from matplotlib import pyplot as plt
from tkinter import ttk
from tkinter.filedialog import askopenfilename

def resimsec():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    im_ = Image.open(filename)
    im_.show()
    return im_
im = resimsec()
np_im = numpy.array(im)
original_im = np_im

sharp_filter = [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]]

averaging_filter = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]

Edge_x = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

Edge_y = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]

mean_filter = [[1 / 9, 1 / 9, 1 / 9],
               [1 / 9, 1 / 9, 1 / 9],
               [1 / 9, 1 / 9, 1 / 9]]

morphology_filter = [['/', 1, '/'],
                     [1, 1, 1],
                     ['/', 1, '/']]


def rgb2gray(np_image):
    imageHeight = len(np_image)
    imageWidth = len(np_image[0])
    grayImage = numpy.empty([imageHeight, imageWidth], dtype=numpy.uint8)
    for i in range(len(np_image)):
        for j in range(len(np_image[i])):
            grayImage[i][j] = int(np_image[i][j][0] * 0.2126 + np_image[i][j][1] * 0.7152 + np_image[i][j][2] * 0.0722)
    im2 = Image.fromarray(grayImage)
    im2.show()
    return grayImage


np_im = rgb2gray(np_im)
list_im = np_im.tolist()


def bandw(list_image):
    esik = int(input('esik degeri gir:'))
    i, j = [0, 0]
    for i in range(len(list_image)):
        for j in range(len(list_image[i])):
            if list_image[i][j] < esik:
                list_image[i][j] = 0
            else:
                list_image[i][j] = 255
    list_image = numpy.array(list_image)
    im2 = Image.fromarray(list_image)
    im2.show()
    return list_image


def cut(list_image):
    print('Resim boyutları X:', len(list_image[0]), 'Y:', len(list_image))
    while 1:
        x, y = [int(x) for x in input("Baslangic Kordinatlari Girin  (X Y): ").split()]
        x2, y2 = [int(x) for x in input("Bitis Kordinatlari Girin  (X Y): ").split()]
        if x < 0 or x > len(list_image[0]) or y < 0 or y > len(list_image) or x2 < x or x2 < 0 or x2 > len(
                list_image[0]) or y2 < y or y2 < 0 or y2 > len(list_image):
            print('Hatali kordinat girdisi')
        else:
            break
    new = []
    cutted_image = []
    temp = [[0] * (x2 - x + 1) for i in range(y2 - y + 1)]
    i,j = [0,0]
    temp_x = x
    while y < y2 - 1:
        while x < x2:
            temp[i][j] = list_image[y][x]
            x += 1
            j += 1
        x = temp_x
        j = 0
        i += 1
        y += 1
    cutted_image = temp
    cutted_image = numpy.array(cutted_image)
    im2 = Image.fromarray(cutted_image.astype(numpy.uint8))
    im2.show()
    return cutted_image

def zoom(zoom_list):
    zoom_list = numpy.array(zoom_list)
    zoom_list = zoom_list.tolist()
    zoom_height = len(zoom_list)
    zoom_weight = len(zoom_list[0])
    a = []
    k = 0
    new = []
    temp = []
    while k < zoom_height:
        i = 0
        while i < zoom_weight:
            zoom_list[k].insert(i + 1, (zoom_list[k][i] + zoom_list[k][i + 1]) / 2)
            i += 2
        k += 1
    k = 0
    i = 0
    while k < len(zoom_list) - 1:
        a.append(zoom_list[k])
        j = 0
        while j < len(zoom_list[k]) and k + 1 < len(zoom_list):
            temp.append((zoom_list[k][j] + zoom_list[k + 1][j]) / 2)
            j += 1
        new = list(temp)
        a.append(new)
        temp.clear()
        k += 1
    k = 0
    zoom_list = a
    zoom_list = numpy.array(zoom_list)
    im2 = Image.fromarray(zoom_list)
    im2.show()
    a.clear()
    return zoom_list.tolist()


def zoomout(uzak_list):
    uzak_list = numpy.array(uzak_list)
    uzak_list = uzak_list.tolist()
    a = []
    b = []
    k = 0
    new = []
    temp = []
    while k < len(uzak_list)-1:
        i = 0
        while i < len(uzak_list[0])-1:
            temp.append((uzak_list[k][i] + uzak_list[k][i + 1]) / 2)
            i += 2
        new = list(temp)
        a.append(new)
        temp.clear()
        k += 1
    k = 0
    while k < len(uzak_list) - 2:
        j = 0
        while j < len(a[k]) and k + 1 < len(uzak_list):
            temp.append((a[k][j] + a[k + 1][j]) / 2)
            j += 1
        new = list(temp)
        b.append(new)
        temp.clear()
        k += 2
    uzak_list = numpy.array(b)
    im2 = Image.fromarray(uzak_list)
    im2.show()
    b.clear()
    return uzak_list.tolist()

def histogram(histogram_list):
    histogram_ = numpy.zeros((256), numpy.uint64)
    i, j = [0, 0]
    for i in range(len(histogram_list)):
        for j in range(len(histogram_list[0])):
            histogram_[histogram_list[i][j]] += 1
    x = numpy.arange(0, 256)
    plt.bar(x, histogram_, color="gray", align="center")
    plt.show()
    return histogram_.tolist()


def histogram_equalization(eq_list, histogram_eq):
    sum_his = numpy.zeros((256), numpy.uint64)
    sum_his[0] = histogram_eq[0]
    new_image = eq_list
    i = 1
    MN = len(eq_list) * len(eq_list[0])
    for i in range(256):
        sum_his[i] = sum_his[i - 1] + histogram_eq[i]
    for j in range(256):
        sum_his[j] = (sum_his[j] * 255) / MN
    i, j = [0, 0]
    for i in range(len(eq_list)):
        for j in range(len(eq_list[0])):
            new_image[i][j] = sum_his[eq_list[i][j]]
    new_image = numpy.array(new_image)
    im2 = Image.fromarray(new_image.astype(numpy.uint8))
    im2.show()
    histogram(new_image)


def quantization(list_quanti):
    a = int(input('Ton Gir : '))
    n = 0
    while 2 ** n < a:
        n += 1
    a = a << 8 - n
    for j in range(len(list_quanti)):
        for k in range(len(list_quanti[0])):
            list_quanti[j][k] = list_quanti[j][k] & a
            list_quanti[j][k] = list_quanti[j][k] >> 1
    list_quanti = numpy.array(list_quanti)
    im2 = Image.fromarray(list_quanti)
    im2.show()


def filter_elements_sum(sel_filter):
    elements_sum = 0
    for i in range(len(sel_filter)):
        for y in range(len(sel_filter[0])):
            elements_sum = elements_sum + sel_filter[i][y]
    return elements_sum


def place(list_place, place_average):
    count = 0
    for k in range(1, len(list_place) - 1):
        for j in range(1, len(list_place[0]) - 1):
            list_place[k][j] = place_average[count]
            count += 1
    return list_place


def filter(list_filter, selected_filter):
    average = []
    fil_elements_sum = filter_elements_sum(selected_filter)
    average_sum, temp_x, temp_y = [0, 0, 0]
    for y in range(len(list_filter) - 2):
        temp_y = y
        for x in range(len(list_filter[0]) - 2):
            temp_x = x
            for v in range(len(selected_filter)):
                for u in range(len(selected_filter[0])):
                    average_sum = average_sum + (list_filter[y][x] * selected_filter[u][v])
                    x += 1
                y += 1
                x = temp_x
            if fil_elements_sum != 0:
                average.append(average_sum / fil_elements_sum)
            else:
                average.append(average_sum)
            average_sum = 0
            y = temp_y
    list_filter = place(list_filter, average)
    list_filter = numpy.array(list_filter)
    im2 = Image.fromarray(list_filter)
    im2.show()
    average.clear()
    return list_filter.tolist()


def find_median(median_list):
    median_list.sort()
    return median_list[int((len(median_list) / 2) + 1)]


def median_filter(list_median):
    median_temp = []
    median_numbers = []
    filter_size = 3
    for i in range(len(list_median) - 2):
        temp_i = i
        for j in range(len(list_median[0]) - 2):
            temp_j = j
            for v in range(filter_size):
                for u in range(filter_size):
                    median_temp.append(list_median[i][j])
                    j += 1
                i += 1
                j = temp_j
            temp = median_temp
            median_numbers.append(find_median(temp))
            median_temp.clear()
            i = temp_i
    list_median = place(list_median, median_numbers)
    list_filter = numpy.array(list_median)
    im2 = Image.fromarray(list_filter)
    im2.show()
    return list_filter.tolist()


def gaussian_blur(list_gaussian):
    kernel = [[0] * len(list_gaussian[0]) for i in range(len(list_gaussian))]
    sigma = int(input('Standart sapma degeri gir:'))
    for i in range(len(list_gaussian) - 1):
        for j in range(len(list_gaussian[0]) - 1):
            Gx = (1 / (2 * math.pi * sigma ** 2)) * math.e ** (
                -(((i - (len(list_gaussian) - 1) / 2) ** 2 + (j - (len(list_gaussian[0]) - 1) / 2)) / (2 * sigma ** 2)))
            kernel[i][j] = Gx
    list_gaussian = sum_Array(list_gaussian, kernel)
    list_gaussian = numpy.array(list_gaussian)
    im2 = Image.fromarray(list_gaussian)
    im2.show()


def contraharmonic(list_contraharmonic):
    Q = 1.5
    filter_size = 3
    contraharmonic_values = []
    contraharmonic_value = 0
    for i in range(len(list_contraharmonic) - 2):
        temp_i = i
        for j in range(len(list_contraharmonic[0]) - 2):
            temp_j = j
            for v in range(filter_size):
                for u in range(filter_size):
                    if list_contraharmonic[i][j] != 0:
                        contraharmonic_value = contraharmonic_value + (
                                (list_contraharmonic[i][j] ** (Q + 1)) / (list_contraharmonic[i][j] ** Q))
                    else:
                        contraharmonic_value = contraharmonic_value + (((list_contraharmonic[i][j] + 1) ** (Q + 1)) / (
                                (list_contraharmonic[i][j] + 1) ** Q))
                    j += 1
                i += 1
                j = temp_j
            i = temp_i
            contraharmonic_values.append(contraharmonic_value / 9)
            contraharmonic_value = 0
    list_contraharmonic = place(list_contraharmonic, contraharmonic_values)
    list_contraharmonic = numpy.array(list_contraharmonic)
    im2 = Image.fromarray(list_contraharmonic)
    im2.show()
    return list_contraharmonic.tolist()


def sum_Array(list1, list2):
    sum_list = [[0] * len(list1[0]) for i in range(len(list2))]
    for i in range(len(list1) - 1):
        for j in range(len(list1[0]) - 1):
            sum_list[i][j] = list1[i][j] + list2[i][j]
    return sum_list

def sub_Array(list1, list2):
    sub_list = [[0] * len(list1[0]) for i in range(len(list2))]
    for i in range(len(list1) - 1):
        for j in range(len(list1[0]) - 1):
            sub_list[i][j] = list1[i][j] - list2[i][j]
    return sub_list

def sum_Array_edge(listx, listy):
    sum_list = [[0] * len(listx[0]) for i in range(len(listx))]
    for i in range(len(listx) - 1):
        for j in range(len(listx[0]) - 1):
            if listx[i][j] > 230:
                listx[i][j] = 255
            else:
                listx[i][j] = 0
            if listy[i][j] > 230:
                listy[i][j] = 255
            else:
                listy[i][j] = 0
            sum_list[i][j] = listx[i][j] + listy[i][j]
    return sum_list


def morp_check(checklist):
    result, black, white = [1, 0, 255]
    for i in range(len(checklist)):
        result *= checklist[i]
    if result < 2:
        return black
    else:
        return white


def morphologicial(list_morp, mode):
    morp_changed_list = numpy.zeros((len(list_morp), len(list_morp[0])), dtype=numpy.uint8)
    for y in range(1,len(list_morp) - 1):
        for x in range(1,len(list_morp[0]) - 1):
            if list_morp[y][x] == 0 and mode == 1:
                morp_changed_list[y-1:y+2,x-1:x+2] = 0
            elif list_morp[y][x] != 0 and mode == 1:
                morp_changed_list[y][x] = list_morp[y][x]
            elif list_morp[y][x] == 255 and mode == 0:
                morp_changed_list[y - 1:y + 2, x - 1:x + 2] = 255
    list_morp = morp_changed_list
    list_morp = numpy.array(list_morp)
    im2 = Image.fromarray(list_morp)
    im2.show()
    return list_morp

def skeletonize(list_skel):
    for i in range(10):
        list_skel =  morphologicial(list_skel,0)
    list_skel = numpy.array(list_skel)
    im2 = Image.fromarray(list_skel)
    im2.show()



