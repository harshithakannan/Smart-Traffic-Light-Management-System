
!pip install ultralytics opencv-python
from google.colab import drive
drive.mount('/content/drive')
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('/content/drive/MyDrive/yolov8s.pt')

image_path = '/content/drive/MyDrive/sample_image7.jpg'


results = model(image_path)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

model = YOLO('/content/drive/MyDrive/yolov8s.pt')


video_path = '/content/drive/MyDrive/test4.mp4'


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


output_path = '/content/drive/MyDrive/Trafficdataset/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no more frames

    results = model(frame)


    annotated_frame = results[0].plot()


    out.write(annotated_frame)


    cv2_imshow(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import pandas as pd
from google.colab import drive
import cv2


drive.mount('/content/drive')

model = YOLO('/content/drive/MyDrive/yolov8s.pt')


image_paths = [
    '/content/drive/MyDrive/sample_image.jpg',
    '/content/drive/MyDrive/sample_image4.jpg',
    '/content/drive/MyDrive/sampe_image10.jpg'
]


results_list = []


vehicle_time = {
    'Small': 2,       # For cars, autos
    'Two Wheeler': 1.5,
    'Large': 3        # For buses, trucks
}


density_threshold = 10  # Baseline number of vehicles for low density
density_multiplier = 0.5  # Multiplier to adjust the density impact


base_density_factor = 1.0


road_width_adjustment_factor = 1.1  # Example dynamic factor

for cycle_no, img_path in enumerate(image_paths, start=1):
    results = model(img_path)

    vehicle_count = {'Small': 0, 'Two Wheeler': 0, 'Large': 0}

    for box in results[0].boxes:
        class_id = int(box.cls.item())  # Get the class ID
        class_name = results[0].names[class_id]  # Get the class name

        if class_name in ['car', 'auto', 'truck']:
            vehicle_count['Small'] += 1
        elif class_name in ['two_wheeler']:
            vehicle_count['Two Wheeler'] += 1
        elif class_name in ['bus', 'large_vehicle']:
            vehicle_count['Large'] += 1

    total_vehicles = sum(vehicle_count.values())

    if total_vehicles > density_threshold:
        density_factor = base_density_factor + ((total_vehicles - density_threshold) / density_threshold) * density_multiplier
    else:
        density_factor = base_density_factor

    print(f"Cycle {cycle_no}:")
    print(f"  Image: {img_path.split('/')[-1]}")
    print(f"  Small Vehicles: {vehicle_count['Small']}")
    print(f"  Two Wheelers: {vehicle_count['Two Wheeler']}")
    print(f"  Large Vehicles: {vehicle_count['Large']}")
    print(f"  Density Factor: {density_factor}")

    signal_time = sum(
        vehicle_count[category] * vehicle_time[category]
        for category in vehicle_count
    ) * density_factor * road_width_adjustment_factor

    signal_time = min(round(signal_time, 2), 60)

    print(f"  Calculated Signal Time: {signal_time} seconds")


    results_list.append({
        'Cycle Number': cycle_no,
        'Image': img_path.split('/')[-1],
        'Small Vehicles': vehicle_count['Small'],
        'Two Wheelers': vehicle_count['Two Wheeler'],
        'Large Vehicles': vehicle_count['Large'],
        'Total Vehicles': total_vehicles,
        'Density Factor': density_factor,
        'Signal Time (s)': signal_time
    })


df_results = pd.DataFrame(results_list)


output_excel_path = '/content/drive/MyDrive/vehicle_signal_times_real_life.xlsx'


df_results.to_excel(output_excel_path, index=False)


print(f"Results saved to: {output_excel_path}")
from ultralytics import YOLO
import pandas as pd
from google.colab import drive
import os

drive.mount('/content/drive')


model = YOLO('/content/drive/MyDrive/yolov8s.pt')


image_paths = [
    '/content/drive/MyDrive/sample_image5.jpg',
    '/content/drive/MyDrive/sample_image7.jpg',
    '/content/drive/MyDrive/sample_image8.jpg'
]


cycle_file = '/content/drive/MyDrive/cycle_number.txt'


if os.path.exists(cycle_file):
    with open(cycle_file, 'r') as file:
        cycle_number = int(file.read().strip())
else:
    cycle_number = 1

base_density_factor = 1.0
density_multiplier = 0.02  # More conservative increase in time with density

vehicle_time = {
    'Small': 1.5,  # Cars, autos
    'Two Wheeler': 1.0,  # Motorcycles
    'Large': 3.0  # Buses, trucks
}

results_list = []

def calculate_road_width_adjustment_factor(road_width):
    return max(1.0, 1.2 - (road_width / 10.0))  

road_width = 15  
road_width_adjustment_factor=calculate_road_width_adjustment_factor(road_width)

for idx, img_path in enumerate(image_paths):
    results = model(img_path)

    vehicle_count = {'Small': 0, 'Two Wheeler': 0, 'Large': 0}

    for box in results[0].boxes:
        class_id = int(box.cls.item())  # Get the class ID
        class_name = results[0].names[class_id]  # Get the class name

        if class_name in ['car', 'auto', 'truck']:
            vehicle_count['Small'] += 1
        elif class_name in ['two_wheeler']:
            vehicle_count['Two Wheeler'] += 1
        elif class_name in ['bus', 'large_vehicle']:
            vehicle_count['Large'] += 1

    total_vehicles = sum(vehicle_count.values())

    density_factor = base_density_factor + (density_multiplier * total_vehicles)

    base_signal_time = (
        (vehicle_count['Small'] * vehicle_time['Small']) +
        (vehicle_count['Two Wheeler'] * vehicle_time['Two Wheeler']) +
        (vehicle_count['Large'] * vehicle_time['Large'])
    )

    adjusted_signal_time = base_signal_time * density_factor * road_width_adjustment_factor

    signal_time = max(15, min(round(adjusted_signal_time, 2), 90))

    direction = ['A', 'B', 'C'][idx % 3]

    print(f"Cycle {cycle_number}:")
    print(f"  Image: {img_path.split('/')[-1]}")
    print(f"  Small Vehicles: {vehicle_count['Small']}")
    print(f"  Two Wheelers: {vehicle_count['Two Wheeler']}")
    print(f"  Large Vehicles: {vehicle_count['Large']}")
    print(f"  Total Vehicles: {total_vehicles}")
    print(f"  Density Factor: {density_factor}")
    print(f"  Base Signal Time: {base_signal_time}")
    print(f"  Road Width Adjustment Factor: {road_width_adjustment_factor}")
    print(f"  Adjusted Signal Time: {adjusted_signal_time}")
    print(f"  Final Calculated Signal Time: {signal_time} seconds")

    results_list.append({
        'Cycle Number': cycle_number,
        'Direction': direction,
        'Small Vehicles': vehicle_count['Small'],
        'Two Wheelers': vehicle_count['Two Wheeler'],
        'Large Vehicles': vehicle_count['Large'],
        'Total Vehicles': total_vehicles,
        'Signal Time (s)': signal_time
    })

df_results = pd.DataFrame(results_list)

output_excel_path = '/content/drive/MyDrive/vehicle_signal_times_real_life.xlsx'

if not os.path.exists(output_excel_path):
    updated_df = df_results
else:
    existing_df = pd.read_excel(output_excel_path)
    updated_df = pd.concat([existing_df, df_results], ignore_index=True)

updated_df.to_excel(output_excel_path, index=False)

with open(cycle_file, 'w') as file:
    file.write(str(cycle_number + 1))

print("Results saved to", output_excel_path)
