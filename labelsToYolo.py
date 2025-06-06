"""
Eina d'etiquetatge per passar keypoints a bbox.
"""


import glob
import json
import math
import os
import sys
import cv2
from InferenceMaskCrop import Mask
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QRect, QRectF, QThread
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtWidgets import QSlider, QApplication, QWidget, QDialog, QLabel, QGridLayout, QFileDialog, QPushButton, QMessageBox

maskModel = Mask('G')
bboxSizes = [20, 40, 80, 100, 120, 140, 160]    #Mides del crop per passar a la mascara
punts = []
pathSessions = "\\\\DESKTOP-I3TDHEL\\sessionImages\\"


class ImageLabel(QLabel):
    """QLabel que permet dibuixar bbox"""
    bbox_drawn = QtCore.pyqtSignal(tuple)

    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.pixmap_original = None
        self.drawing = False
        self.start_point = QtCore.QPoint()
        self.end_point = QtCore.QPoint()
        self.current_rect = QRect()
        self.manual_bbox = None
        self.selected = False

    def setPixmap(self, pixmap):
        self.pixmap_original = pixmap
        super(ImageLabel, self).setPixmap(pixmap)

    def setManualBbox(self, bbox):
        self.manual_bbox = bbox
        self.update()

    def clearManualBbox(self):
        self.manual_bbox = None
        self.selected = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.selected = False
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.current_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            self.current_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

            if self.pixmap_original:
                scale_x = self.pixmap_original.width() / self.width()
                scale_y = self.pixmap_original.height() / self.height()
                x = int(self.current_rect.x() * scale_x)
                y = int(self.current_rect.y() * scale_y)
                w = int(self.current_rect.width() * scale_x)
                h = int(self.current_rect.height() * scale_y)
                self.manual_bbox = (x, y, w, h)
                self.selected = False
                self.bbox_drawn.emit((x, y, w, h))

    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QPainter(self)
        if self.drawing:
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)
        if self.manual_bbox is not None:
            # Dibuixar la bbox manual
            x, y, w, h = self.manual_bbox
            scale_x = self.pixmap_original.width() / self.width()
            scale_y = self.pixmap_original.height() / self.height()
            bx = x / scale_x
            by = y / scale_y
            bw = w / scale_x
            bh = h / scale_y
            pen_color = Qt.green if not self.selected else Qt.yellow
            pen = QPen(pen_color, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(bx, by, bw, bh))


class BBoxViewer(QDialog):
    """View amb les opcions de bbox"""

    def __init__(self, imgOriginal, filtered_bboxs, center_point):
        super().__init__()
        self.setWindowTitle("Visualització de Bounding Boxes")
        self.imgOriginal = imgOriginal
        self.filtered_bboxs = filtered_bboxs
        self.center_point = center_point
        self.selected_bbox_index = None
        self.manual_bbox = None  # bbox manual (x,y,w,h)
        self.goBack = False
        self.initUI()

    def initUI(self):
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.display_bboxes()
        self.display_manual_area()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        brightness_label = QLabel("Brillantor:")
        self.brightness_slider.setFocusPolicy(Qt.NoFocus)
        self.grid.addWidget(brightness_label, 2, 3)
        self.grid.addWidget(self.brightness_slider, 3, 3)
        self.grid.addWidget(QLabel(""))

    def update_brightness(self, value):
        adjusted_img = cv2.convertScaleAbs(self.manual_region, alpha=1, beta=value)
        pixmap = self.convert_cv_to_pixmap(adjusted_img)
        self.manual_label.setPixmap(pixmap.scaled(
            self.manual_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_bboxes(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for idx, bbox in enumerate(self.filtered_bboxs[:5]):
            subimg, x_marge, y_marge = self.get_subimage_with_margin(bbox, margin_percent=1)
            pixmap = self.convert_cv_to_pixmap(subimg)
            pixmap = self.draw_bbox_on_pixmap(pixmap, bbox, idx+1, subimg.shape[1], subimg.shape[0], x_marge, y_marge)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setFixedSize(200, 200)
            label.setScaledContents(True)
            self.grid.addWidget(QLabel(str(idx)), (idx // 3) * 2, idx % 3)
            self.grid.addWidget(label, ((idx // 3) * 2) + 1, idx % 3)

    def display_manual_area(self):
        manual_label = ImageLabel()
        manual_label.setFixedSize(1000, 1000)
        manual_pixmap = self.get_manual_region_pixmap()
        manual_label.setPixmap(manual_pixmap.scaled(
            manual_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        manual_label.bbox_drawn.connect(self.handle_manual_bbox_drawn)
        manual_label.manual_bbox = self.manual_bbox
        self.manual_label = manual_label
        self.grid.addWidget(QLabel("9"), 0, 3)
        self.grid.addWidget(self.manual_label, 1, 3)

    def handle_manual_bbox_drawn(self, bbox):
        self.manual_bbox = (int(bbox[0] / 2) + self.offsetManual[0],
                            int(bbox[1] / 2) + self.offsetManual[1],
                            int(bbox[2] / 2),
                            int(bbox[3] / 2))
        self.manual_label.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.goBack = True
            self.close()
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4,
                           Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9]:
            num = int(event.text())
            total_bboxes = len(self.filtered_bboxs)
            has_manual = self.manual_bbox is not None
            if num >= 0 and num <= total_bboxes:
                self.selectedBbox = self.filtered_bboxs[num]
            elif has_manual and num == 9:
                if self.manual_bbox is None:
                    return
                self.selectedBbox = self.manual_bbox
            else:
                print("incorrect number")
                return
            self.close()

    def getSelectedBbox(self):
        return self.selectedBbox

    def askToGoBack(self):
        return self.goBack

    def reset(self):
        self.goBack = False


    def get_subimage_with_margin(self, bbox, margin_percent=0.5):
        x, y, w, h = bbox
        marge = int(margin_percent * max(w, h))
        x_marge = max(x - marge, 0)
        y_marge = max(y - marge, 0)
        x2_marge = min(x + w + marge, self.imgOriginal.shape[1])
        y2_marge = min(y + h + marge, self.imgOriginal.shape[0])
        subimg = self.imgOriginal[y_marge:y2_marge, x_marge:x2_marge]
        return subimg, x_marge, y_marge

    def get_manual_region_pixmap(self):
        cx, cy = self.center_point
        x1 = max(cx - 250, 0)
        y1 = max(cy - 250, 0)
        x2 = x1 + 500
        y2 = y1 + 500
        if x2 > self.imgOriginal.shape[1]:
            x2 = self.imgOriginal.shape[1]
            x1 = x2 - 500
            if x1 < 0:
                x1 = 0
        if y2 > self.imgOriginal.shape[0]:
            y2 = self.imgOriginal.shape[0]
            y1 = y2 - 500
            if y1 < 0:
                y1 = 0
        aux = np.copy(self.imgOriginal)
        cv2.circle(aux, (cx, cy), 2, (0, 0, 255), -1)
        self.offsetManual = (x1, y1)
        manual_region = aux[y1:y2, x1:x2]
        self.manual_region = manual_region
        return self.convert_cv_to_pixmap(manual_region)

    def convert_cv_to_pixmap(self, cv_img):
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap

    def draw_bbox_on_pixmap(self, pixmap, bbox, index, sub_width, sub_height, x_marge, y_marge):
        x, y, w, h = bbox
        rel_x = x - x_marge
        rel_y = y - y_marge
        scaleX = pixmap.width() / sub_width
        scaleY = pixmap.height() / sub_height
        bx = rel_x * scaleX
        by = rel_y * scaleY
        bw = w * scaleX
        bh = h * scaleY
        p = QPainter(pixmap)
        pen = QPen(Qt.green, 1, Qt.SolidLine)
        p.setPen(pen)
        p.drawRect(QRectF(bx, by, bw, bh))
        p.end()
        return pixmap


def boxes_overlap(box1, box2, margin=5):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    diff_left = abs(x1 - x2)
    diff_top = abs(y1 - y2)
    diff_right = abs((x1 + w1) - (x2 + w2))
    diff_bottom = abs((y1 + h1) - (y2 + h2))
    if diff_left > margin or diff_top > margin or diff_right > margin or diff_bottom > margin:
        return False
    else:
        return True


class BBoxPreloadThread(QThread):
    resultReady = QtCore.pyqtSignal(int, list)  # Emet (índex, bboxFinal)

    def __init__(self, index, imgOriginal, img, punt, amplada, altura):
        super().__init__()
        self.index = index
        self.imgOriginal = imgOriginal
        self.img = img
        self.punt = punt
        self.amplada = amplada
        self.altura = altura

    def run(self):
        result = MainApp.compute_bbox_final_static(self.imgOriginal, self.img, self.punt, self.amplada, self.altura)
        self.resultReady.emit(self.index, result)


class MainApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = None
        with open("Nuria_labels.json", "r") as f:
            self.data = json.load(f)

    @staticmethod
    def compute_bbox_final_static(imgOriginal, img, punt, amplada, altura):
        bboxsSelected = []
        for size in bboxSizes:
            x1 = int(punt[0] - size / 2)
            y1 = int(punt[1] - size / 2)
            x2 = int(punt[0] + size / 2)
            y2 = int(punt[1] + size / 2)

            moveX = 0
            if x1 < 0:
                moveX = -x1
            elif x2 > amplada:
                moveX = amplada - x2

            moveY = 0
            if y1 < 0:
                moveY = -y1
            elif y2 > altura:
                moveY = altura - y2

            x1 += moveX
            x2 += moveX
            y1 += moveY
            y2 += moveY

            if x1 < 0:
                x1 = 0
                x2 = size
            if y1 < 0:
                y1 = 0
                y2 = size
            if x2 > amplada:
                x2 = amplada
                x1 = amplada - size
            if y2 > altura:
                y2 = altura
                y1 = altura - size

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, amplada)
            y2 = min(y2, altura)

            crop = imgOriginal[y1:y2, x1:x2]
            bboxs = maskModel.getAllBbox(crop, offset=(x1, y1))
            minDistance = 1000
            indexSelected = -1
            for i, bbox in enumerate(bboxs):
                if bbox[2] > 5 and bbox[3] > 5:
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2
                    distancia = math.hypot(cx - punt[0], cy - punt[1])
                    if distancia < minDistance:
                        minDistance = distancia
                        indexSelected = i
            if indexSelected != -1:
                bboxsSelected.append(bboxs[indexSelected])

        bboxsSelected_sorted = sorted(bboxsSelected, key=lambda b: b[2] * b[3], reverse=True)
        filtered_bboxs_count = {}
        for bbox in bboxsSelected_sorted:
            overlap = False
            for kept_bbox in filtered_bboxs_count:
                if boxes_overlap(bbox, kept_bbox, margin=2):
                    overlap = True
                    filtered_bboxs_count[kept_bbox] += 1
                    break
            if not overlap:
                filtered_bboxs_count[bbox] = 1

        bboxFinal = []
        for val in filtered_bboxs_count:
            bboxFinal.append(val)
        return bboxFinal

    def run(self):
        total = 0
        fets = 0

        for label in self.data:
            lesions = 0
            labelPath = ".\\labels\\" + label[:-3] + "txt"
            if os.path.exists(labelPath):
                if self.data[label]["done"] == 1:
                    total += len(self.data[label]['labelData'])
                    fets += len(self.data[label]['labelData'])
                    lesions += len(self.data[label]['labelData'])
            elif self.data[label]["done"] == 1:
                total += len(self.data[label]['labelData'])
                lesions += len(self.data[label]['labelData'])
            print("Patient", labelPath, "lesions", lesions)

        for label in self.data:
            info = label.split("\\")
            labelPath = ".\\labels\\" + label[:-3] + "txt"
            labelPathJson = ".\\labels\\" + label[:-3] + "json"
            if os.path.exists(labelPath):
                continue

            print("Processing:", labelPath)
            possiblePath = pathSessions + label
            imgsPaths = glob.glob(possiblePath)
            imgOriginal = cv2.imread(imgsPaths[0])
            img = np.copy(imgOriginal)
            altura, amplada = img.shape[:2]
            bboxsResult = []
            dictionaryExtraInfo = {"small": [], "big": []}

            if self.data[label]["done"] == 1:
                lesions_list = self.data[label]['labelData']
                precomputed_bbox = {}
                preload_threads = {}

                # Inici pre-calcul per a totes les lesions en threads
                for idx, lesion in enumerate(lesions_list):
                    punt = lesion['points']
                    thread = BBoxPreloadThread(idx, imgOriginal, img, punt, amplada, altura)
                    thread.resultReady.connect(lambda index, result: precomputed_bbox.update({index: result}))
                    thread.start()
                    preload_threads[idx] = thread

                print("N:", len(lesions_list))

                ij = 0
                while ij < len(lesions_list):
                    lesion = lesions_list[ij]
                    punt = lesion['points']
                    # Esperem el calcul
                    while ij not in precomputed_bbox:
                        QApplication.processEvents()
                    bboxFinal = precomputed_bbox[ij]

                    viewer = BBoxViewer(imgOriginal, bboxFinal, punt)
                    viewer.exec()


                    if viewer.askToGoBack():
                        viewer.reset()
                        if ij > 0:
                            ij -= 1
                            if bboxsResult:
                                bboxsResult.pop()
                    else:
                        bboxsResult.append(viewer.getSelectedBbox())
                        ij += 1
                        fets += 1
                    print("Fets:", fets, total, fets/total)
            else:
                continue

            carpeta = label.split("\\")
            if not os.path.exists(".\\labels\\" + carpeta[0]):
                os.makedirs(".\\labels\\" + carpeta[0])

            with open(labelPath, "w") as f:
                for bbox in bboxsResult:
                    cx = (bbox[0] + bbox[2] / 2) / amplada
                    cy = (bbox[1] + bbox[3] / 2) / altura
                    w = bbox[2] / amplada
                    h = bbox[3] / altura
                    f.write("0 " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + "\n")
            with open(labelPathJson, "w") as f:
                json.dump(dictionaryExtraInfo, f)
        print("Fets:", fets, total, fets / total)


if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()
