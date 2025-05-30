import sys
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *


class Volume3DViewer(QGLWidget):
    def __init__(self, volume, mask=None, parent=None):
        super(Volume3DViewer, self).__init__(parent)
        self.volume = volume
        self.mask = mask
        self.skip = 1  # Finer rendering
        self.angle_x = 0
        self.angle_y = 0
        self.last_pos = None
        self.vertex_vbo = None
        self.color_vbo = None
        self.num_points = 0

    def prepare_voxels(self):
        points = []
        colors = []
        vmax = np.max(self.volume)

        for z in range(0, self.volume.shape[2], self.skip):
            for y in range(0, self.volume.shape[1], self.skip):
                for x in range(0, self.volume.shape[0], self.skip):
                    val = self.volume[x, y, z] / vmax
                    if val > 0.05:
                        points.append([x, y, z])
                        if self.mask is not None and self.mask[x, y, z] > 0:
                            colors.append([1.0, 0.0, 0.0])  # red for mask
                        else:
                            colors.append([val, val, val])  # grayscale

        points_np = np.array(points, dtype=np.float32)
        colors_np = np.array(colors, dtype=np.float32)
        self.num_points = len(points_np)

        self.vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, points_np.nbytes, points_np, GL_STATIC_DRAW)

        self.color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glBufferData(GL_ARRAY_BUFFER, colors_np.nbytes, colors_np, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glPointSize(1.0)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)  # Anti-aliasing if supported
        self.prepare_voxels()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 0.1, 2000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        scale = 2.0 / max(self.volume.shape)
        glTranslatef(0.0, 0.0, -2.5)
        glScalef(scale, scale, scale)
        glTranslatef(-self.volume.shape[0] / 2,
                     -self.volume.shape[1] / 2,
                     -self.volume.shape[2] / 2)

        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glColorPointer(3, GL_FLOAT, 0, None)

        glDrawArrays(GL_POINTS, 0, self.num_points)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.angle_x += dy
            self.angle_y += dx
            self.last_pos = event.pos()
            self.update()


class MainWindow(QMainWindow):
    def __init__(self, volume_path, mask_path=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle("3D Medical Volume Viewer with Fine Detail")

        vol_img = nib.load(volume_path)
        volume_data = vol_img.get_fdata().astype(np.float32)

        mask_data = None
        if mask_path:
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(np.uint8)

        self.viewer = Volume3DViewer(volume_data, mask=mask_data)

        layout = QVBoxLayout()
        layout.addWidget(self.viewer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    volume_path = "./BraTS19_2013_7_1_flair.nii"        # Replace with actual path
    mask_path = "./BraTS19_2013_7_1_seg.nii"            # Replace with actual path
    window = MainWindow(volume_path, mask_path)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
