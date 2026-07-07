import sys
import cv2
import numpy as np
import socket
from ultralytics import YOLO

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QCheckBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

# ================= UDP CONFIGURATION =================
UDP_IP = "192.168.0.4"
UDP_PORT = 5005

# ================= BACKGROUND VIDEO THREAD =================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    posture_signal = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.pose_model = YOLO("best.pt")

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.pose_model.predict(frame, stream=True, classes=[0], verbose=False)

            is_posture_abnormal = False
            reason = "Normal"
            person_count = 0

            for r in results:
                frame = r.plot()
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    kpts = r.keypoints.xy.cpu().numpy()
                    for i, kpt in enumerate(kpts):
                        person_count += 1
                        code, reason = self.classify_posture(kpt)
                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])

                        cv2.putText(
                            frame, f"Person {person_count}",
                            (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 0), 2
                        )

                        if code != 0:
                            label_map = {
                                1: "Abnormal (Head Forward)",
                                2: "Abnormal (Head Backward)",
                                3: "Abnormal (Lean Left)",
                                4: "Abnormal (Lean Right)"
                            }
                            cv2.putText(
                                frame, label_map[code],
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2
                            )
                            is_posture_abnormal = True
                        print(f"Posture Code: {code}, Reason: {reason}")

            self.change_pixmap_signal.emit(frame)
            self.posture_signal.emit(is_posture_abnormal, reason)

            if not self._run_flag:
                break

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def classify_posture(self, k):
        try:
            head_tilt = k[0][1] - (k[1][1] + k[2][1]) / 2
            shoulder_diff = k[5][1] - k[6][1]

            if head_tilt > 25: return 1, "Head Forward"
            if head_tilt < 8: return 2, "Head Backward"
            if shoulder_diff > 10: return 3, "Lean Left"
            if shoulder_diff < -25: return 4, "Lean Right"
            return 0, "Normal"
        except Exception:
            return 0, "Normal"

# ================= MAIN GUI WINDOW =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Monitor: AI Pose & ECG")
        self.resize(1100, 650)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sent = None

        self.posture_bad = False
        self.ecg_bad = False
        self.locked_false = False   # latch flag

        self.init_ui()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.posture_signal.connect(self.update_posture)
        self.thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("border: 3px solid #333; background: black;")
        main_layout.addWidget(self.video_label)

        controls_group = QGroupBox("System Controls")
        controls_layout = QVBoxLayout()

        self.lbl_posture = QLabel("Posture: Initializing...")
        self.lbl_posture.setFont(QFont("Arial", 14))
        self.lbl_posture.setStyleSheet("color: blue; padding: 10px;")

        self.chk_ecg = QCheckBox("Simulate ECG Abnormality")
        self.chk_ecg.setFont(QFont("Arial", 14))
        self.chk_ecg.setStyleSheet("padding: 10px;")
        self.chk_ecg.toggled.connect(self.update_ecg)

        self.lbl_final = QLabel("FINAL STATUS:\nTRUE (NORMAL)")
        self.lbl_final.setAlignment(Qt.AlignCenter)
        self.lbl_final.setMinimumHeight(200)
        self.lbl_final.setStyleSheet("background: green; color: white; font-size: 20pt; font-weight: bold; border-radius: 15px;")

        controls_layout.addWidget(self.lbl_posture)
        controls_layout.addWidget(self.chk_ecg)
        controls_layout.addStretch()
        controls_layout.addWidget(self.lbl_final)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

    @pyqtSlot(np.ndarray)
    def update_image(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, w * c, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    @pyqtSlot(bool, str)
    def update_posture(self, is_bad, reason):
        self.posture_bad = is_bad
        self.lbl_posture.setText(f"AI Detection: {reason}")
        self.lbl_posture.setStyleSheet(f"font-size: 14pt; color: {'red' if is_bad else 'green'};")
        self.calculate_final_result()

    def update_ecg(self, checked):
        self.ecg_bad = checked
        self.calculate_final_result()

    def calculate_final_result(self):
        # Once locked_false is set, never send true again
        if self.locked_false:
            current_signal = "false"
        else:
            is_abnormal = self.posture_bad and self.ecg_bad
            current_signal = "false" if is_abnormal else "true"

        if current_signal == "false":
            self.lbl_final.setText("FINAL STATUS:\nFALSE (ABNORMAL)")
            self.lbl_final.setStyleSheet("background: red; color: white; font-size: 20pt; font-weight: bold; border-radius: 15px;")
            self.locked_false = True  # latch permanently
        else:
            self.lbl_final.setText("FINAL STATUS:\nTRUE (NORMAL)")
            self.lbl_final.setStyleSheet("background: green; color: white; font-size: 20pt; font-weight: bold; border-radius: 15px;")

        if current_signal != self.last_sent:
            try:
                message = f"Final State: {current_signal}"
                self.sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                print(f"UDP Packet Sent: {message}")
                self.last_sent = current_signal
            except Exception as e:
                print(f"Transmission Error: {e}")

    def closeEvent(self, event):
        self.thread.stop()
        self.sock.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())