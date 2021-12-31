import sys

sys.path.insert(0, './yolov5')

import os
from pathlib import Path
import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, \
    increment_path
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from reid.reid import ReID
from gooey import Gooey, GooeyParser


def track(arguments):
    # PARSED ARGS
    video_file = arguments.video_file
    reid_model = arguments.reid_model
    is_save = not arguments.save_results

    # HARDCODED ARGS
    yolo_model = 'yolov5/weights/yolov5m_pedestrian.pt'
    output_dir = 'output'
    image_size = [640, 640]
    is_export_yolo_stages = False

    device = select_device(0)

    if is_save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reid = ReID(reid_model)

    model = DetectMultiBackend(yolo_model, device=device, dnn=False)
    stride = model.stride
    pt = model.pt
    jit = model.jit

    half = device.type != 'cpu' and pt
    if pt:
        model.model.half() if half else model.model.float()

    is_display = check_imshow()
    image_size = check_img_size(image_size, s=stride)
    dataset = LoadImages(video_file, img_size=image_size, stride=stride, auto=pt and not jit)
    path_video = write_video = [None]

    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *image_size).to(device).type_as(next(model.model.parameters())))

    counter = 0
    for _, (path, image, image_0s, cap, info) in enumerate(dataset):
        image = torch.from_numpy(image).to(device)
        image = image.half() if half else image.float()
        image /= 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        time_start_det = time_sync()
        is_export_yolo_stages = increment_path(
            str(Path(output_dir)) + '/' + Path(path).stem,
            mkdir=True
        ) if is_export_yolo_stages else False
        predictions = model(image, augment=False, visualize=is_export_yolo_stages)
        time_end_det = time_sync()

        predictions = non_max_suppression(
            predictions,
            0.5, 0.7,  # CONF // IOU
            0,  # CLASSES 0 (PEDESTRIAN)
            False,  # AGNOSTIC
            1000  # MAX DETECTIONS
        )
        info = info.upper()
        for i, detection in enumerate(predictions):  # detections per image
            counter += 1
            ori_image = image_0s.copy()
            proc_info = '%gx%g ' % image.shape[2:]  # print string
            save_path = str(Path(output_dir) / Path(path).name)
            annotator = Annotator(ori_image, line_width=2, pil=not ascii)
            if detection is not None and len(detection):
                detection[:, :4] = scale_coords(image.shape[2:], detection[:, :4], ori_image.shape).round()
                bbox_xywh = xyxy2xywh(detection[:, 0:4])
                confidences = detection[:, 4]
                classes = detection[:, 5]
                for detected_class in detection[:, -1].unique():
                    amount = (detection[:, -1] == detected_class).sum()  # detections per class
                    proc_info += f"{amount} {names[int(detected_class)]}{'s' * (amount > 1)}, ".upper()  # add to string
                time_start_reid = time_sync()
                results = reid.forward(bbox_xywh.cpu(), confidences.cpu(), classes.cpu(), ori_image)
                time_end_reid = time_sync()
                if len(results) > 0:
                    for j, (result, conf) in enumerate(zip(results, confidences)):
                        class_num = result[5]
                        identifier = result[4]
                        bboxes = result[0:4]
                        detected_class = int(class_num)  # integer class
                        label = f'{identifier} {names[detected_class]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(int(identifier % 255), True))
            else:
                reid.increment_ages()
            LOGGER.info(
                f'[\n'
                f'\tTIME :: {time_end_det - time_start_det + time_end_reid - time_start_reid:.3f}s\n'
                f'\tDATA :: {info}.\n'
                f'\tPROC :: {proc_info}\n'
                f']'
            )
            ori_image = annotator.result()
            if is_save:
                if path_video != save_path:
                    path_video = save_path
                    if isinstance(write_video, cv2.VideoWriter):
                        write_video.release()
                    if cap:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                    else:
                        fps = 30
                        w, h = ori_image.shape[1], ori_image.shape[0]
                        save_path += '.mp4'
                    write_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                write_video.write(ori_image)
            if is_display:
                cv2.imshow(path, ori_image)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
    if is_save:
        LOGGER.info(f'[\n\t SAVED TO {os.getcwd() + os.sep + output_dir}  \n]')


@Gooey(
    program_name="Tracker",
    default_size=(800, 600),
)
def main():
    parser = GooeyParser(description="Tracking params")

    req_group = parser.add_argument_group(
        "Main",
        "Mandatory params",
        gooey_options={
            'columns': 1
        }
    )
    req_group.add_argument(
        '--video_file',
        metavar="File",
        required=True,
        help="Pick video file for tracking from your machine",
        widget="FileChooser",
        nargs="?",
        gooey_options={
            'wildcard':
                'MP4 (*.mp4)|*.mp4',
            'message': 'Choose video to track'
        }
    )
    req_group.add_argument(
        '--reid_model',
        metavar="ReID Model",
        required=True,
        help="Choose ReID net model from the list below",
        choices=[
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'resnet50_fc512', 'se_resnet50', 'se_resnet50_fc512', 'se_resnet101', 'se_resnext50_32x4d',
            'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512',
            'inceptionresnetv2', 'inceptionv4', 'xception', 'resnet50_ibn_a', 'resnet50_ibn_b', 'nasnsetmobile',
            'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'shufflenet', 'squeezenet1_0', 'squeezenet1_0_fc512',
            'squeezenet1_1', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'mudeep', 'resnet50mid', 'hacnn', 'pcb_p6', 'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5',
            'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5',
            'osnet_ain_x0_25'
        ]
    )

    opt_group = parser.add_argument_group("Optional params")
    opt_group.add_argument(
        '--save_results',
        metavar="Save",
        help="Save resulting video",
        action="store_false"
    )
    arguments = parser.parse_args()

    with torch.no_grad():
        track(arguments)


if __name__ == '__main__':
    main()
