from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os 

log_path = "results/step1/lightning_logs/version_4/"
print(log_path)

av_files = os.listdir(log_path)
for files in av_files:
    if "events" in files:
        event_file_path = log_path + files
print(event_file_path)

event_file = EventAccumulator(
    event_file_path
)

event_file.Reload()
event_file.Tags()

scalar_tags = event_file.Tags()['scalars']

'''
for tag in scalar_tags:
    scalar_events = event_file.Scalars(tag)
    print(f"\nData for tag '{tag}':")
    for event in scalar_events:
        print(f"Step {event.step}, Wall time {event.wall_time}: {event.value}")
#step 199
'''

step_file = []
#for tag in ['epoch','val_0_iou_0', 'val_0_iou_1', 'val_0_iou_2', 'val_0_iou_3', 'val_0_iou_4', 'val_0_iou_5', 'val_0_iou_6', 'val_0_iou_7', 'val_0_iou_8', 'val_0_iou_9', 'val_0_miou']:
for tag in scalar_tags:
    scalar_events = event_file.Scalars(tag)
    print(f"Data for tag '{tag}':")
    for event in scalar_events:
        if event.step==18: #39
          print(f"Step {event.step}, Wall time {event.wall_time}: {event.value}")
