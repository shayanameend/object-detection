import cv2


def main():
  print("Object Detection Demo")

  cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  image = cv2.imread("data/demo/car.jpg")
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  objects = cascade.detectMultiScale(
    image=gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
  )

  for (x, y, w, h) in objects:
    cv2.rectangle(
      img=image,
      pt1=(x, y),
      pt2=(x + w, y + h),
      color=(0, 255, 0),
      thickness=2
    )

  cv2.imshow(winname="Detected Objects", mat=image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
