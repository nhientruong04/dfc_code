import os
from hailo_sdk_client import ClientRunner
from typing import Optional


def optimize(har_path: str, model_name: str, calib_dataset_path: str,
             output_dir: Optional[str] = None, **kwargs) -> None:
    assert os.path.isfile(har_path)

    if output_dir is None:
        output_dir = os.curdir()
    os.makedirs(output_dir, exist_ok=True)

    runner = ClientRunner(har=har_path)

    if 'model_script_path' in kwargs.keys():
        model_script_path = kwargs['model_script_path']
        assert os.path.isfile(model_script_path)

        runner.load_model_script(model_script_path)

    runner.optimize(calib_data=calib_dataset_path,
                    work_dir=os.path.join(output_dir, 'dump'))

    quantized_model_har_path = os.path.join(
        output_dir, f"{model_name}_quantized.har")
    runner.save_har(quantized_model_har_path)


if __name__ == "__main__":
    har_path = 'output/EfficientAD_student.har'
    model_name = 'EfficientAD_student'
    calib_dataset_path = 'test_bottle_dataset.npy'
    output_dir = 'output'
    optimize(har_path, model_name, calib_dataset_path, output_dir)
