import cv2
import numpy as np
import torch
from collections import deque
import time


class YOLOPeopleCounter:
    def __init__(self, video_path, output_path="yolo_output.mp4", model_name='yolov5s'):
        self.video_path = video_path
        self.output_path = output_path

        # 加载YOLO模型
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = 0.5  # 置信度阈值
        self.model.iou = 0.45  # IOU阈值

        self.people_count_history = deque(maxlen=30)
        self.current_count = 0
        self.total_people = 0

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("无法打开视频文件")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        people_data = []
        processing_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 每隔10帧检测一次（YOLO较慢）
            if frame_count % 10 == 0:
                start_time = time.time()

                # 使用YOLO检测
                results = self.model(frame)

                # 提取人物检测结果（类别0为人）
                people_detections = results.xyxy[0][results.xyxy[0][:, 5] == 0]
                self.current_count = len(people_detections)

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                self.people_count_history.append(self.current_count)
                avg_count = np.mean(self.people_count_history)
                self.total_people = max(self.total_people, int(avg_count))

                # 绘制检测结果
                frame = self.draw_detections(frame, people_detections)

                people_data.append({
                    'frame': frame_count,
                    'count': self.current_count,
                    'timestamp': frame_count / fps
                })

            out.write(frame)

            if frame_count % 50 == 0:
                print(f"已处理 {frame_count} 帧，当前人数: {self.current_count}")

        cap.release()
        out.release()

        self.generate_detailed_report(people_data, fps, processing_times)
        return people_data

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'Current: {self.current_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Max: {self.total_people}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def generate_detailed_report(self, people_data, fps, processing_times):
        counts = [data['count'] for data in people_data]

        print("\n" + "=" * 60)
        print("YOLO人物检测详细统计报告")
        print("=" * 60)
        print(f"视频总帧数: {len(people_data) * 10}")
        print(f"检测间隔: 每10帧检测一次")
        print(f"最大同时出现人数: {max(counts)}")
        print(f"平均人数: {np.mean(counts):.2f}")
        print(f"检测到人物的帧数: {sum(1 for c in counts if c > 0)}")
        print(f"平均处理时间: {np.mean(processing_times):.4f} 秒/帧")
        print(f"视频时长: {len(people_data) * 10 / fps:.2f} 秒")

        # 人数分布统计
        unique_counts = np.unique(counts)
        print(f"\n人数分布:")
        for count in unique_counts:
            percentage = (counts.count(count) / len(counts)) * 100
            print(f"  {count}人: {counts.count(count)}次 ({percentage:.1f}%)")
        print("=" * 60)


# 使用方法
if __name__ == "__main__":
    video_path = "a.mp4"

    # 使用YOLO检测（需要安装torch和torchvision）
    yolo_counter = YOLOPeopleCounter(video_path)
    results = yolo_counter.process_video()