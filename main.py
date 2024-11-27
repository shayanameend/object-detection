import logging
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2


def check_dependencies():
  dependencies = ['ffmpeg', 'opencv_createsamples', 'opencv_traincascade']
  missing_deps = []

  for dep in dependencies:
    try:
      subprocess.run([dep, '--help'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
      missing_deps.append(dep)

  if not missing_deps:
    messagebox.showerror("Dependency Error",
                         f"Missing dependencies: {', '.join(missing_deps)}")
    sys.exit(1)


class HaarCascadeTrainer:
  def __init__(self):
    check_dependencies()

    self.video_path = None
    self.output_dir = None
    self.frames_dir = None

    self.logger = self._setup_logging()

    self.current_frame = None
    self.current_frame_index = 0
    self.total_frames = 0
    self.selected_frames = []

    self.rectangles = []
    self.drawing = False
    self.ix, self.iy = -1, -1

  @staticmethod
  def _setup_logging():
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s: %(message)s',
      handlers=[
        logging.FileHandler('haar_cascade.log'),
        logging.StreamHandler(sys.stdout)
      ]
    )
    return logging.getLogger(__name__)

  def convert_video_to_avi(self):
    output_path = os.path.splitext(self.video_path)[0] + '.avi'
    try:
      subprocess.run(['ffmpeg', '-i', self.video_path,
                      '-c:v', 'mjpeg', output_path],
                     check=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
      return output_path
    except subprocess.CalledProcessError as e:
      messagebox.showerror("Conversion Error", str(e))
      return None

  def extract_frames(self, video_path):
    self.frames_dir = os.path.join(self.output_dir, 'frames')
    os.makedirs(self.frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_skip = max(1, total_frames // 60)
    extracted_frames = 0

    for frame_num in range(0, total_frames, frame_skip):
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
      ret, frame = cap.read()
      if not ret:
        break

      frame_path = os.path.join(self.frames_dir, f'frame_{extracted_frames:04d}.bmp')
      cv2.imwrite(frame_path, frame)
      extracted_frames += 1

    cap.release()
    self.selected_frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.bmp')])
    self.total_frames = len(self.selected_frames)

    return extracted_frames

  def interactive_object_marking(self):
    def mouse_event(event, x, y, flags, param):
      if event == cv2.EVENT_LBUTTONDOWN:
        self.drawing = True
        self.ix, self.iy = x, y

      elif event == cv2.EVENT_MOUSEMOVE:
        if self.drawing:
          frame_copy = self.current_frame.copy()
          cv2.rectangle(frame_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
          cv2.imshow('Object Marking', frame_copy)

      elif event == cv2.EVENT_LBUTTONUP:
        self.drawing = False
        x1, y1 = min(self.ix, x), min(self.iy, y)
        x2, y2 = max(self.ix, x), max(self.iy, y)

        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width > 20 and height > 20:
          cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          self.rectangles.append((x1, y1, width, height))
          cv2.imshow('Object Marking', self.current_frame)

    def load_frame():
      frame_path = os.path.join(self.frames_dir, self.selected_frames[self.current_frame_index])
      self.current_frame = cv2.imread(frame_path)
      self.rectangles.clear()
      cv2.imshow('Object Marking', self.current_frame)

    def save_annotations():
      with open(os.path.join(self.output_dir, 'positive.txt'), 'a') as f:
        if self.rectangles:
          frame_path = os.path.join(self.frames_dir, self.selected_frames[self.current_frame_index])
          annotation_line = [frame_path, str(len(self.rectangles))]

          for rect in self.rectangles:
            x, y, w, h = rect
            annotation_line.extend([str(x), str(y), str(w), str(h)])

          f.write(' '.join(annotation_line) + '\n')

    def next_frame():
      save_annotations()
      if self.current_frame_index < self.total_frames - 1:
        self.current_frame_index += 1
        load_frame()
        frame_label.config(text=f'Frame {self.current_frame_index + 1}/{self.total_frames}')

    def prev_frame():
      save_annotations()
      if self.current_frame_index > 0:
        self.current_frame_index -= 1
        load_frame()
        frame_label.config(text=f'Frame {self.current_frame_index + 1}/{self.total_frames}')

    cv2.namedWindow('Object Marking')
    cv2.setMouseCallback('Object Marking', mouse_event)

    control_window = tk.Tk()
    control_window.title('Frame Navigation')

    prev_btn = tk.Button(control_window, text='Previous Frame', command=prev_frame)
    prev_btn.pack(side=tk.LEFT)

    next_btn = tk.Button(control_window, text='Next Frame', command=next_frame)
    next_btn.pack(side=tk.RIGHT)

    finish_btn = tk.Button(control_window, text='Finish Marking', command=control_window.quit)
    finish_btn.pack()

    frame_label = tk.Label(control_window, text=f'Frame 1/{self.total_frames}')
    frame_label.pack()

    load_frame()
    cv2.imshow('Object Marking', self.current_frame)
    control_window.mainloop()

    save_annotations()
    cv2.destroyAllWindows()
    control_window.destroy()

    return self._count_positive_samples()

  def _count_positive_samples(self):
    with open(os.path.join(self.output_dir, 'positive.txt'), 'r') as f:
      return len(f.readlines())

  def create_samples(self, num_pos):
    samples_dir = os.path.join(self.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    cmd = [
      'opencv_createsamples',
      '-info', os.path.join(self.output_dir, 'positive.txt'),
      '-num', str(num_pos),
      '-w', '24',
      '-h', '24',
      '-vec', os.path.join(samples_dir, 'samples.vec')
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
      return os.path.join(samples_dir, 'samples.vec')
    return None

  def create_negative_samples_txt(self):
    negative_samples_dir = os.path.join(self.output_dir, 'negative_samples')

    if not os.path.exists(negative_samples_dir):
      messagebox.showerror("Error", "Negative samples directory not found")
      return 0

    neg_images = [
      os.path.join(negative_samples_dir, f)
      for f in os.listdir(negative_samples_dir)
      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    with open(os.path.join(self.output_dir, 'negatives.txt'), 'w') as f:
      for img in neg_images:
        f.write(f"{img}\n")

    return len(neg_images)

  def train_cascade(self, num_pos, num_neg):
    cascade_dir = os.path.join(self.output_dir, 'cascade')
    os.makedirs(cascade_dir, exist_ok=True)

    samples_vec = self.create_samples(num_pos)
    if not samples_vec:
      messagebox.showerror("Error", "Could not create samples vector")
      return None

    cmd = [
      'opencv_traincascade',
      '-data', cascade_dir,
      '-vec', samples_vec,
      '-bg', os.path.join(self.output_dir, 'negatives.txt'),
      '-numPos', str(num_pos),
      '-numNeg', str(num_neg),
      '-numStages', '10',
      '-w', '24',
      '-h', '24',
      '-featureType', 'HAAR',
      '-minHitRate', '0.999',
      '-maxFalseAlarmRate', '0.5',
      '-mode', 'ALL'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
      cascade_xml = os.path.join(cascade_dir, 'cascade.xml')
      return cascade_xml
    return None

  def detect_objects(self, avi_video, cascade_path):
    classifier = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(avi_video)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(self.output_dir, 'detection_result.avi')
    out = cv2.VideoWriter(
      out_video_path,
      cv2.VideoWriter_fourcc(*'MJPG'),
      10,
      (frame_width, frame_height)
    )

    total_detections = 0
    detection_frames = 0

    while True:
      ret, frame = cap.read()
      if not ret:
        break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      objects = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(24, 24)
      )

      if len(objects) > 0:
        detection_frames += 1
        total_detections += len(objects)

        for (x, y, w, h) in objects:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      out.write(frame)

    cap.release()
    out.release()

    messagebox.showinfo(
      "Detection Results",
      f"Total Objects Detected: {total_detections}\n"
      f"Frames with Detections: {detection_frames}\n"
      f"Output Video: {out_video_path}"
    )

    subprocess.Popen(['start', out_video_path], shell=True)

    return total_detections, detection_frames

  def run(self):
    root = tk.Tk()
    root.withdraw()

    self.video_path = filedialog.askopenfilename(
      title="Select Video File",
      filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not self.video_path:
      return

    self.output_dir = os.path.join(os.path.dirname(self.video_path), 'haar_cascade_output')
    os.makedirs(self.output_dir, exist_ok=True)

    avi_video = self.convert_video_to_avi()
    if not avi_video:
      return

    frame_count = self.extract_frames(avi_video)
    if frame_count == 0:
      messagebox.showerror("Error", "No frames extracted")
      return

    positive_samples = self.interactive_object_marking()
    if positive_samples == 0:
      messagebox.showerror("Error", "No positive samples marked")
      return

    negative_samples = self.create_negative_samples_txt()
    if negative_samples == 0:
      messagebox.showerror("Error", "No negative samples found")
      return

    cascade_path = self.train_cascade(
      num_pos=min(positive_samples, 50),
      num_neg=min(negative_samples, 100)
    )

    if not cascade_path:
      messagebox.showerror("Error", "Cascade training failed")
      return

    self.detect_objects(avi_video, cascade_path)


def main():
  try:
    trainer = HaarCascadeTrainer()
    trainer.run()
  except Exception as e:
    messagebox.showerror("Critical Error", str(e))


if __name__ == "__main__":
  main()
