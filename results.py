import pandas as pd
import os

def load_and_label_csv(filepath, model_name, scenario):
    """
    주어진 CSV 파일을 읽고, 추가적으로 model과 scenario 정보를 부착하여
    DataFrame을 반환합니다.
    """
    df = pd.read_csv(filepath)
    df['model'] = model_name
    df['scenario'] = scenario
    return df

def main():
    # CSV 파일 경로 설정 (여기서는 /mnt/data 디렉터리를 가정)
    base_dir = '.'

    # 불러올 CSV 파일과, 그에 대응하는 (모델명, 시나리오)를 매핑
    files_info = [
        ('hf_advanced.csv', 'hf', 'advanced'),
        ('hf_basic.csv', 'hf', 'basic'),
        ('hf_classic.csv', 'hf', 'classic'),
        ('gemma3_advanced.csv', 'gemma3', 'advanced'),
        ('gemma3_classic.csv', 'gemma3', 'classic'),
        ('sqlcoder_advanced.csv', 'sqlcoder', 'advanced'),
        ('sqlcoder_basic.csv', 'sqlcoder', 'basic'),
        ('sqlcoder_classic.csv', 'sqlcoder', 'classic'),
    ]

    # 개별 CSV를 하나씩 읽어들여 합치기
    dfs = []
    for filename, model_name, scenario in files_info:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = load_and_label_csv(filepath, model_name, scenario)
            dfs.append(df)
        else:
            print(f"경고: {filepath} 파일이 존재하지 않습니다. 건너뜁니다.")

    # 하나의 DataFrame으로 합치기
    if not dfs:
        print("읽어들인 파일이 없습니다. 스크립트를 종료합니다.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # 열 이름 예시
    # 아래 예시는 "exec_match"라는 열(실행 결과가 정답과 일치하는지 여부)과
    # "semantic_match"라는 열(의미적으로 정답과 일치하는지 여부)를 가정하였습니다.
    # 실제 CSV에 맞게 수정이 필요합니다.

    # 먼저, 데이터 확인(상위 5개 행) 출력
    print("==== Combined Data (Head) ====")
    print(combined_df.head())

    # 모델/시나리오별 통계 요약(예시)
    # exec_match, semantic_match 등 Boolean 혹은 0/1 컬럼이라 가정
    # CSV 구조에 따라 집계 로직을 알맞게 수정하세요.
    group_cols = ['model', 'scenario']
    agg_dict = {
        'exec_match': 'mean',       # 실행 결과 일치율(평균)
        'semantic_match': 'mean'    # 의미적 일치율(평균)
    }

    # 실제 컬럼명과 존재 여부 확인 필요
    summary_df = combined_df.groupby(group_cols).agg(agg_dict).reset_index()
    summary_df.rename(columns={
        'exec_match': 'exec_match_rate',
        'semantic_match': 'semantic_match_rate'
    }, inplace=True)

    print("\n==== 모델/시나리오별 평균 정확도 ====")
    print(summary_df)

    # 필요하다면 추가적인 통계(예: 샘플 수 등)도 계산
    summary_df_count = combined_df.groupby(group_cols).size().reset_index(name='count')
    print("\n==== 모델/시나리오별 평가 개수 ====")
    print(summary_df_count)

    # CSV로 결과 저장 예시
    output_path = os.path.join(base_dir, 'combined_summary.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"\n결과 요약이 {output_path} 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
