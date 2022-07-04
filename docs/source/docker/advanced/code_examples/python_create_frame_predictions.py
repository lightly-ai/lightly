import av
import json
from pathlib import Path

dataset_dir = Path('/datasets/my_dataset')
predictions_dir = dataset_dir / '.lightly' / 'predictions' / 'my_prediction_task'

def model_predict(frame):
    # This function must be overwritten to generate predictions for a frame using
    # a prediction model of your choice. Here we just return an example prediction.
    return [{'category_id': 0, 'bbox': [0, 10, 100, 30], 'score': 0.8}]

for video_path in dataset_dir.glob('**/*.mp4'):
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as file:
            json.dump(prediction, file)


# example directory structure before
# .
# ├── test
# │   └── video_0.mp4
# └── train
#     └── video_1.mp4
#
# example directory structure after
# .
# ├── .lightly
# │   └── predictions
# │       └── my_prediction_task
# │           ├── test
# │           │   ├── video_0-000-mp4.json
# │           │   ├── video_0-001-mp4.json
# │           │   ├── video_0-002-mp4.json
# │           │   └── ...
# │           └── train
# │               ├── video_1-000-mp4.json
# │               ├── video_1-001-mp4.json
# │               ├── video_1-002-mp4.json
# │               └── ...
# ├── test
# │   └── video_0.mp4
# └── train
#     └── video_1.mp4
