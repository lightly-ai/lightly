import av
import json
from pathlib import Path

dataset_dir = Path('/datasets/my_dataset')
video_path = dataset_dir / 'video.mp4'
predictions_dir = dataset_dir / '.lightly' / 'predictions' / 'my_prediction_task'
predictions_dir.mkdir(parents=True, exist_ok=True)

def model_predict(frame):
    # This function must be overwritten to generate predictions for a 
    # frame using a prediction model of your choice.
    return []

# get predictions for frames
predictions = []
with av.open(str(video_path)) as container:
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        predictions.append(model_predict(frame.to_image()))

# save predictions
num_frames = len(predictions)
zero_padding = len(str(num_frames))
for frame_index, frame_predictions in enumerate(predictions):
    video_name = video_path.relative_to(dataset_dir).with_suffix('')
    frame_name = Path(f'{video_name}-{frame_index:0{zero_padding}}-{video_path.suffix[1:]}.png')
    prediction = {
        'file_name': str(frame_name),
        'predictions': frame_predictions,
    }
    out_path = predictions_dir / frame_name.with_suffix('.json')
    with open(out_path, 'w') as file:
        json.dump(prediction, file)

# results in the following file structure
# .
# ├── .lightly
# │   └── predictions
# │       └── my_prediction_task
# │           ├── video-000-mp4.json
# │           ├── video-001-mp4.json
# │           ├── video-002-mp4.json
# │           └── ...
# └── video.mp4
