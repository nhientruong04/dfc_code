from hailo_sdk_client import ClientRunner
from typing import List, Optional
from loguru import logger
import os

hw_arch = "hailo8l"
model_name = "EfficientAD_student"
onnx_path = '/home/nhien/aiot_model/student.onnx'


def main(save_har: bool = False, save_hef: bool = False):
    pass


def parse(onnx_path: str, model_name: str, net_input_shapes: List,
          input_names: Optional[str] = None, output_names: Optional[str] = None,
          hw_arch: str = 'hailo8l') -> str:
    assert os.path.exists(onnx_path), "Path not exist"

    har_path = f'{model_name}.har'
    runner = ClientRunner(hw_arch=hw_arch)

    hn, npz = runner.translate_onnx_model(
        onnx_path,
        model_name,
        start_node_names=input_names,
        end_node_names=output_names,
        net_input_shapes=net_input_shapes,
    )

    runner.save_har(f'{model_name}.har')

    logger.info(f'Saved parsed model from {onnx_path} to {har_path}')
    return har_path


if __name__ == "__main__":
    save_har = True
    save_hef = True
    main(save_har, save_hef)
