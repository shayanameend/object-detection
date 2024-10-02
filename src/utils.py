import cv2


def read_image(path: str):
  image = cv2.imread(path)

  if image is None:
    print("Error: Could not load image.")
    exit(0)

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return image, gray_image


def show_image(title: str, image):
  cv2.imshow(winname=title, mat=image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def detect_objects(cascade_path: str, image, gray_image, height=24, width=24):
  cascade = cv2.CascadeClassifier(cascade_path)

  if cascade.empty():
    print("Error: Could not load cascade classifier.")
    exit(0)

  objects = cascade.detectMultiScale(
    image=gray_image,
    minSize=(height, width),
  )

  if len(objects) == 0:
    print("Error: Could not detect any objects.")
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
