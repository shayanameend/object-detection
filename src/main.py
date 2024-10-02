from utils import read_image, detect_objects, show_image


def main():
  print("Object Detection Demo")

  image, gray_image = read_image('data/demo/cars.bmp')

  show_image(title="Original Image", image=image)

  image, gray_image, objects = detect_objects(
    cascade_path='classifier/cascade.xml',
    image=image,
    gray_image=gray_image,
    height=18,
    width=32,
  )

  show_image(title="Detected Objects", image=image)


if __name__ == "__main__":
  main()
