from utils import read_video, show_video


def main():
  video_path = 'data/demo/cars.avi'
  cascade_path = 'classifier/cascade.xml'
  video = read_video(video_path)
  show_video(title="Detected Objects in Video", video=video, cascade_path=cascade_path)


if __name__ == "__main__":
  main()
