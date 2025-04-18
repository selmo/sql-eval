import os
from pathlib import Path
import re


def detect_models_and_datasets(directory):
    """
    지정된 디렉토리에서 모델명과 데이터셋 유형을 자동으로 감지합니다.

    Parameters:
    -----------
    directory : str
        CSV 파일이 위치한 디렉토리 경로

    Returns:
    --------
    tuple
        (models, dataset_types) - 감지된 모델명 리스트와 데이터셋 유형 리스트
    """
    dir_path = Path(directory)

    # 디렉토리 내 모든 CSV 파일 찾기
    csv_files = list(dir_path.glob("*.csv"))

    if not csv_files:
        print(f"Warning: No CSV files found in {directory}")
        return [], []

    # CSV 파일명에서 모델명과 데이터셋 유형 추출
    models = set()
    dataset_types = set()

    # 파일명 패턴: modelname_datasettype.csv
    pattern = re.compile(r'([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)\.csv')

    for file in csv_files:
        match = pattern.match(file.name)
        if match:
            model_name, dataset_type = match.groups()
            models.add(model_name)
            dataset_types.add(dataset_type)

    return sorted(list(models)), sorted(list(dataset_types))


# 사용 예시:
if __name__ == "__main__":
    # 테스트 디렉토리 경로
    test_dir = "."  # 현재 디렉토리

    # 모델과 데이터셋 유형 감지
    models, dataset_types = detect_models_and_datasets(test_dir)

    print(f"Detected models: {models}")
    print(f"Detected dataset types: {dataset_types}")