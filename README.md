```markdown
# Hand and Face Tracking with OpenCV and MediaPipe

This project demonstrates hand and face tracking using `OpenCV` and `MediaPipe` libraries. It captures live video from your webcam, processes the video frames for hand and face landmarks, and overlays the detected landmarks on the live feed. The project also calculates the distance between the thumb and finger tips in real-time.

## Features

- Hand tracking using `MediaPipe Hands`
- Face mesh tracking using `MediaPipe Face Mesh`
- Real-time FPS display
- Distance calculation between thumb and fingertip landmarks
- Overlay of landmarks and connections for both hands and face on the video feed

## Requirements

To run this project, you need Python 3.7 or later and the following Python packages:

- `opencv-python`
- `mediapipe`
- `numpy`

You can install the required dependencies using `pip`:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Toji-cmd/HandAndFaceRecognition.git
cd HandAndFaceRecognition
```

2. Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
```

3. Run the script:

```bash
python hand_face_tracking.py
```

This will open a webcam window that displays the hand and face tracking in real-time. The landmarks for hands and faces will be drawn, and the distances between finger tips and the thumb will be displayed.

## How it works

- **Hand Tracking**: The script uses `MediaPipe`'s hand model to detect key landmarks on the hand, such as the thumb, index, middle, ring, and pinky tips. The distances between the thumb and these finger tips are calculated and displayed on the video.
  
- **Face Mesh**: It uses `MediaPipe`'s face mesh model to detect and draw landmarks on the face. The contours and connections between landmarks are drawn to visualize the detected facial features.

- **Real-time Processing**: The script calculates the frames per second (FPS) to give real-time feedback on the processing speed.

## Troubleshooting

### ModuleNotFoundError: No module named 'mediapipe'

If you encounter this error, make sure you have installed the `mediapipe` package correctly:

```bash
pip install mediapipe
```

If you still face issues, check your Python version (`mediapipe` supports Python 3.7 to 3.10).

### Errors related to camera access

Ensure that your webcam is not being used by other applications. If you're on macOS, you might need to grant permission for the terminal or IDE to access the webcam.

## Contributing

Feel free to fork this project, submit issues, or create pull requests if you want to contribute.

## License

This project is open-source and available under the MIT License.
