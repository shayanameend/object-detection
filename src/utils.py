import cv2


def show_image(title='', path='demo.jpg'):
  image = cv2.imread(path)

  if title == '':
    parsed_filename = path.split('/')[-1].replace('_', ' ')
    parsed_filename = parsed_filename.split('.')[0]

    filename = parsed_filename.split()
    for i in range(len(filename)):
      filename[i] = filename[i].capitalize()
    filename = ' '.join(filename)

    title = filename

  if image is None:
    print("Error: Could not load image.")
  else:
    cv2.imshow(winname=title, mat=image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
