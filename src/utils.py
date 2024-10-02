import cv2


def read_image(path: str):
  image = cv2.imread(path)

  if image is None:
    exit(0)

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return image, gray_image


def show_image(title: str, image):
  cv2.imshow(winname=title, mat=image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def detect_objects(cascade_path: str, image, gray_image):
  cascade = cv2.CascadeClassifier(cascade_path)

  if cascade.empty():
    exit(0)

  objects = cascade.detectMultiScale(gray_image)

  if len(objects) == 0:
    print("No objects detected.")
    exit(0)

  for (x, y, width, height) in objects:
    cv2.rectangle(
      img=image,
      pt1=(x, y),
      pt2=(x + width, y + height),
      color=(0, 255, 0),
      thickness=2
    )

  return image, gray_image, objects


def read_video(path: str):
  video = cv2.VideoCapture(path)

  if not video.isOpened():
    exit(0)

  return video


def show_video(title: str, video, cascade_path: str):
  cascade = cv2.CascadeClassifier(cascade_path)

  if cascade.empty():
    exit(0)

  while video.isOpened():
    ret, frame = video.read()

    if not ret:
      break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    objects = cascade.detectMultiScale(gray_frame)

    for (x, y, width, height) in objects:
      cv2.rectangle(
        img=frame,
        pt1=(x, y),
        pt2=(x + width, y + height),
        color=(0, 255, 0),
        thickness=2
      )

    cv2.imshow(title, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video.release()
  cv2.destroyAllWindows()
