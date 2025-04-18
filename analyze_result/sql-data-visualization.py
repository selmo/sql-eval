import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SQL시각화')


class SQLVisualizationTool:
    """
    SQL 생성 모델의 성능 데이터를 시각화하는 도구
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.models = ["gemma3", "hf", "ollama"]
        self.dataset_types = ["advanced", "basic", "classic"]
        self.data = {}
        self.metrics = {}

    def load_data(self):
        """CSV 파일 로드"""
        logger.info(f"데이터 로드 시작: {self.input_dir}")

        for model in self.models:
            self.data[model] = {}
            for dataset_type in self.dataset_types:
                file_path = self.input_dir / f"{model}_{dataset_type}.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        self.data[model][dataset_type] = df
                        logger.info(f"로드 완료: {file_path.name} ({len(df)} 행)")
                    except Exception as e:
                        logger.error(f"파일 로드 오류: {file_path} - {e}")

        return self

    def calculate_metrics(self):
        """성능 지표 계산"""
        logger.info("성능 지표 계산 중...")

        for model in self.models:
            self.metrics[model] = {}
            for dataset_type in self.dataset_types:
                if dataset_type not in self.data[model]:
                    continue

                df = self.data[model][dataset_type]

                # 필수 열이 있는지 확인
                required_cols = ['correct', 'exact_match', 'error_query_gen', 'error_db_exec', 'latency_seconds']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"{model}_{dataset_type}: 누락된 열 - {missing_cols}")
                    continue

                # 지표 계산
                metrics = {}
                metrics['total_queries'] = len(df)
                metrics['accuracy'] = df['correct'].mean() if 'correct' in df else 0
                metrics['exact_match'] = df['exact_match'].mean() if 'exact_match' in df else 0

                # 오류율 계산
                metrics['error_query_gen'] = df['error_query_gen'].mean() if 'error_query_gen' in df else 0
                metrics['error_db_exec'] = df['error_db_exec'].mean() if 'error_db_exec' in df else 0
                metrics['error_rate'] = metrics['error_query_gen'] + metrics['error_db_exec']

                # 지연시간 계산
                if 'latency_seconds' in df:
                    latency_values = pd.to_numeric(df['latency_seconds'], errors='coerce')
                    metrics['latency'] = latency_values.mean()
                else:
                    metrics['latency'] = 0

                # 카테고리별 성능 분석
                if 'query_category' in df.columns:
                    category_metrics = {}
                    for category, group in df.groupby('query_category'):
                        if not category or pd.isna(category):
                            continue
                        category_metrics[category] = {
                            'total': len(group),
                            'accuracy': group['correct'].mean() if 'correct' in group else 0
                        }
                    metrics['category_performance'] = category_metrics

                self.metrics[model][dataset_type] = metrics

        logger.info("지표 계산 완료")
        return self

    def _model_avg_metrics(self):
        """모델별 평균 성능 지표 계산"""
        model_avg = {}

        for model in self.models:
            if not self.metrics.get(model):
                continue

            total_accuracy = 0
            total_exact_match = 0
            total_error_rate = 0
            total_latency = 0
            dataset_count = 0

            for dataset_type in self.dataset_types:
                if dataset_type not in self.metrics[model]:
                    continue

                metrics = self.metrics[model][dataset_type]
                total_accuracy += metrics['accuracy']
                total_exact_match += metrics['exact_match']
                total_error_rate += metrics['error_rate']
                total_latency += metrics['latency']
                dataset_count += 1

            if dataset_count > 0:
                model_avg[model] = {
                    'accuracy': total_accuracy / dataset_count,
                    'exact_match': total_exact_match / dataset_count,
                    'error_rate': total_error_rate / dataset_count,
                    'latency': total_latency / dataset_count
                }

        return model_avg

    def plot_overall_performance(self):
        """모델별 종합 성능 차트 생성"""
        logger.info("종합 성능 차트 생성 중...")

        model_avg = self._model_avg_metrics()

        if not model_avg:
            logger.warning("차트 생성을 위한 데이터가 없습니다")
            return None

        models = list(model_avg.keys())
        accuracies = [model_avg[m]['accuracy'] for m in models]
        exact_matches = [model_avg[m]['exact_match'] for m in models]
        error_rates = [model_avg[m]['error_rate'] for m in models]

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, accuracies, width, label='정확도', color='#4CAF50')
        ax.bar(x, exact_matches, width, label='정확 일치율', color='#2196F3')
        ax.bar(x + width, error_rates, width, label='오류율', color='#F44336')

        ax.set_ylabel('비율')
        ax.set_title('모델별 종합 성능 비교')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend()

        # Y축 형식 지정
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # 값 표시
        for i, v in enumerate(accuracies):
            ax.text(i - width, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(exact_matches):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(error_rates):
            ax.text(i + width, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 차트 저장
        chart_path = self.output_dir / "overall_performance.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()

        logger.info(f"지연시간 비교 차트 저장됨: {chart_path}")

        return chart_path

    def plot_accuracy_vs_latency(self):
        """정확도 vs 지연시간 산점도 생성"""
        logger.info("정확도 vs 지연시간 산점도 생성 중...")

        # 각 모델 및 데이터셋별 정확도와 지연시간 데이터 수집
        scatter_data = []

        for model in self.models:
            if not self.metrics.get(model):
                continue

            for dataset_type in self.dataset_types:
                if dataset_type not in self.metrics[model]:
                    continue

                accuracy = self.metrics[model][dataset_type]['accuracy']
                latency = self.metrics[model][dataset_type]['latency']

                scatter_data.append({
                    'model': model.upper(),
                    'dataset': dataset_type.upper(),
                    'accuracy': accuracy,
                    'latency': latency
                })

        if not scatter_data:
            logger.warning("산점도를 위한 데이터가 없습니다")
            return None

        # DataFrame으로 변환
        df = pd.DataFrame(scatter_data)

        # 그래프 생성
        plt.figure(figsize=(10, 8))

        # 색상 맵 설정
        colors = {'GEMMA3': '#1E88E5', 'HF': '#43A047', 'OLLAMA': '#E53935'}
        markers = {'ADVANCED': 'o', 'BASIC': 's', 'CLASSIC': '^'}

        # 각 데이터 포인트 플롯
        for i, row in df.iterrows():
            plt.scatter(
                row['latency'],
                row['accuracy'],
                color=colors.get(row['model'], '#000000'),
                marker=markers.get(row['dataset'], 'o'),
                s=100,
                alpha=0.7,
                label=f"{row['model']} - {row['dataset']}"
            )

        # 중복되는 레이블 제거
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        plt.title('정확도 vs 지연시간')
        plt.xlabel('지연시간 (초)')
        plt.ylabel('정확도')

        # Y축 형식 지정
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # 데이터 포인트에 레이블 추가
        for i, row in df.iterrows():
            plt.annotate(
                f"{row['model']}-{row['dataset']}",
                (row['latency'], row['accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 차트 저장
        chart_path = self.output_dir / "accuracy_vs_latency.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()

        logger.info(f"정확도 vs 지연시간 산점도 저장됨: {chart_path}")

        return chart_path

    def plot_all_charts(self):
        """모든 차트 생성"""
        logger.info("모든 차트 생성 시작...")

        # 병렬로 차트 생성
        with ThreadPoolExecutor(max_workers=4) as executor:
            chart_futures = [
                executor.submit(self.plot_overall_performance),
                executor.submit(self.plot_dataset_performance),
                executor.submit(self.plot_category_performance),
                executor.submit(self.plot_error_analysis),
                executor.submit(self.plot_latency_comparison),
                executor.submit(self.plot_accuracy_vs_latency)
            ]

            # 결과 수집
            for future in chart_futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"차트 생성 중 오류 발생: {e}")

        logger.info(f"모든 차트가 {self.output_dir}에 저장되었습니다")

    def create_index_html(self):
        """인덱스 HTML 페이지 생성"""
        logger.info("인덱스 HTML 페이지 생성 중...")

        # 생성된 모든 이미지 파일 찾기
        chart_files = list(self.output_dir.glob("*.png"))

        if not chart_files:
            logger.warning("생성된 차트 파일이 없습니다")
            return

        # HTML 내용 생성
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SQL 생성 모델 성능 시각화</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1, h2, h3 {{
            color: #2c3e50;
        }}

        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}

        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .chart {{
            max-width: 100%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 4px;
        }}

        .description {{
            margin: 15px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
        }}
    </style>
</head>
<body>
    <h1>SQL 생성 모델 성능 시각화</h1>
    <p>생성일: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>

    <div class="chart-container">
        <h2>종합 성능 비교</h2>
        <div class="description">
            <p>모든 데이터셋에 대한 각 모델의 평균 성능 지표를 비교합니다.</p>
        </div>
        <img src="overall_performance.png" alt="종합 성능 비교" class="chart">
    </div>

    <div class="chart-container">
        <h2>데이터셋별 성능</h2>
        <div class="description">
            <p>각 데이터셋 유형에 대한 모델별 성능 비교입니다.</p>
        </div>
        {"".join([f'<img src="dataset_{dataset_type}.png" alt="{dataset_type.upper()} 데이터셋 성능" class="chart"><br><br>' for dataset_type in self.dataset_types])}
    </div>

    <div class="chart-container">
        <h2>카테고리별 성능</h2>
        <div class="description">
            <p>각 카테고리에 대한 모델별 성능을 히트맵으로 표시합니다.</p>
        </div>
        <img src="category_heatmap.png" alt="카테고리별 성능" class="chart">
    </div>

    <div class="chart-container">
        <h2>오류 유형 분석</h2>
        <div class="description">
            <p>모델별 오류 유형(쿼리 생성 오류 vs DB 실행 오류)을 비교합니다.</p>
        </div>
        <img src="error_analysis.png" alt="오류 유형 분석" class="chart">
    </div>

    <div class="chart-container">
        <h2>지연시간 비교</h2>
        <div class="description">
            <p>모델 및 데이터셋별 평균 지연시간을 비교합니다.</p>
        </div>
        <img src="latency_comparison.png" alt="지연시간 비교" class="chart">
    </div>

    <div class="chart-container">
        <h2>정확도 vs 지연시간</h2>
        <div class="description">
            <p>정확도와 지연시간 간의 관계를 산점도로 표시합니다.</p>
        </div>
        <img src="accuracy_vs_latency.png" alt="정확도 vs 지연시간" class="chart">
    </div>

    <footer style="margin-top: 30px; text-align: center; color: #777; border-top: 1px solid #ddd; padding-top: 10px;">
        SQL 생성 모델 성능 시각화 보고서 | 자동 생성됨
    </footer>
</body>
</html>
        """

        # HTML 파일 저장
        html_path = self.output_dir / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"인덱스 HTML 페이지 생성 완료: {html_path}")

    def run(self):
        """시각화 도구 실행"""
        try:
            # 데이터 로드 및 분석
            self.load_data()
            self.calculate_metrics()

            # 모든 차트 생성
            self.plot_all_charts()

            # HTML 인덱스 페이지 생성
            self.create_index_html()

            logger.info("시각화 작업이 성공적으로 완료되었습니다.")
            return True
        except Exception as e:
            logger.error(f"시각화 중 오류 발생: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='SQL 생성 모델 성능 데이터 시각화 도구')
    parser.add_argument('--input', type=str, default='.', help='CSV 파일이 있는 디렉토리 경로')
    parser.add_argument('--output', type=str, default='./visualization', help='시각화 결과를 저장할 디렉토리 경로')

    args = parser.parse_args()

    visualizer = SQLVisualizationTool(args.input, args.output)
    success = visualizer.run()

    if success:
        print(f"시각화가 성공적으로 완료되었습니다. 결과는 {args.output} 디렉토리에서 확인할 수 있습니다.")
        print(f"브라우저에서 {os.path.join(args.output, 'index.html')}을 열어 결과를 확인하세요.")
    else:
        print("시각화 과정에서 오류가 발생했습니다. 로그를 확인하세요.")

    return 0 if success else 1


if __name__ == "__main__":
    main()
    logger.info(f"종합 성능 차트 저장됨: {chart_path}")

    return chart_path


def plot_dataset_performance(self):
    """데이터셋별 성능 차트 생성"""
    logger.info("데이터셋별 성능 차트 생성 중...")

    chart_paths = []

    for dataset_type in self.dataset_types:
        # 데이터 준비
        models = []
        accuracies = []
        exact_matches = []
        error_rates = []

        for model in self.models:
            if model in self.metrics and dataset_type in self.metrics[model]:
                models.append(model.upper())
                metrics = self.metrics[model][dataset_type]
                accuracies.append(metrics['accuracy'])
                exact_matches.append(metrics['exact_match'])
                error_rates.append(metrics['error_rate'])

        if not models:
            logger.warning(f"{dataset_type} 데이터셋 차트를 위한 데이터가 부족합니다")
            continue

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, accuracies, width, label='정확도', color='#4CAF50')
        ax.bar(x, exact_matches, width, label='정확 일치율', color='#2196F3')
        ax.bar(x + width, error_rates, width, label='오류율', color='#F44336')

        ax.set_ylabel('비율')
        ax.set_title(f'{dataset_type.upper()} 데이터셋 성능')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()

        # Y축 형식 지정
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # 값 표시
        for i, v in enumerate(accuracies):
            ax.text(i - width, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(exact_matches):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(error_rates):
            ax.text(i + width, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 차트 저장
        chart_path = self.output_dir / f"dataset_{dataset_type}.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()

        logger.info(f"{dataset_type} 데이터셋 차트 저장됨: {chart_path}")

        chart_paths.append(chart_path)

    return chart_paths


def plot_category_performance(self):
    """카테고리별 성능 히트맵 생성"""
    logger.info("카테고리별 성능 히트맵 생성 중...")

    # 모든 카테고리와 성능 데이터 수집
    all_categories = {}

    for model in self.models:
        if not self.metrics.get(model):
            continue

        for dataset_type in self.dataset_types:
            if dataset_type not in self.metrics[model]:
                continue

            if 'category_performance' not in self.metrics[model][dataset_type]:
                continue

            category_performance = self.metrics[model][dataset_type]['category_performance']

            for category, perf in category_performance.items():
                if category not in all_categories:
                    all_categories[category] = {m: {'total': 0, 'correct': 0} for m in self.models}

                all_categories[category][model]['total'] += perf['total']
                all_categories[category][model]['correct'] += perf['total'] * perf['accuracy']

    if not all_categories:
        logger.warning("카테고리 히트맵을 위한 데이터가 없습니다")
        return None

    # 각 카테고리의 모델별 정확도 계산
    category_accuracies = {}
    category_totals = {}

    for category, model_data in all_categories.items():
        category_totals[category] = sum(data['total'] for data in model_data.values())

        category_accuracies[category] = {}
        for model, data in model_data.items():
            if data['total'] > 0:
                category_accuracies[category][model] = data['correct'] / data['total']
            else:
                category_accuracies[category][model] = 0

    # 샘플 수가 많은 상위 카테고리 선택 (최대 20개)
    top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    top_category_names = [cat for cat, _ in top_categories]

    # 데이터 준비
    category_data = []
    for category in top_category_names:
        for model in self.models:
            if model in category_accuracies[category]:
                category_data.append({
                    'category': category,
                    'model': model.upper(),
                    'accuracy': category_accuracies[category][model]
                })

    # DataFrame으로 변환
    df = pd.DataFrame(category_data)

    if df.empty:
        logger.warning("카테고리 히트맵을 위한 충분한 데이터가 없습니다")
        return None

    # Seaborn 히트맵으로 표시
    pivot_table = df.pivot_table(
        index='category',
        columns='model',
        values='accuracy',
        aggfunc='mean'
    )

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2%',
        cmap='viridis',
        linewidths=.5,
        cbar_kws={'label': '정확도'}
    )

    plt.title('카테고리별 모델 성능')
    plt.tight_layout()

    # 차트 저장
    chart_path = self.output_dir / "category_heatmap.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    logger.info(f"카테고리 히트맵 저장됨: {chart_path}")

    return chart_path


def plot_error_analysis(self):
    """오류 유형 분석 차트 생성"""
    logger.info("오류 유형 분석 차트 생성 중...")

    model_avg = self._model_avg_metrics()

    if not model_avg:
        logger.warning("오류 분석 차트를 위한 데이터가 없습니다")
        return None

    # 데이터 준비
    models = list(model_avg.keys())

    # 모든 데이터셋의 평균 오류율 계산
    query_gen_errors = []
    db_exec_errors = []

    for model in models:
        # 각 데이터셋의 오류 유형별 비율 집계
        query_gen_total = 0
        db_exec_total = 0
        dataset_count = 0

        for dataset_type in self.dataset_types:
            if dataset_type in self.metrics[model]:
                query_gen_total += self.metrics[model][dataset_type]['error_query_gen']
                db_exec_total += self.metrics[model][dataset_type]['error_db_exec']
                dataset_count += 1

        if dataset_count > 0:
            query_gen_errors.append(query_gen_total / dataset_count)
            db_exec_errors.append(db_exec_total / dataset_count)
        else:
            query_gen_errors.append(0)
            db_exec_errors.append(0)

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, query_gen_errors, width, label='쿼리 생성 오류', color='#FF9800')
    ax.bar(x + width / 2, db_exec_errors, width, label='DB 실행 오류', color='#9C27B0')

    ax.set_ylabel('오류율')
    ax.set_title('모델별 오류 유형 분석')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()

    # Y축 형식 지정
    max_error = max(max(query_gen_errors), max(db_exec_errors))
    ax.set_ylim(0, max_error * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # 값 표시
    for i, v in enumerate(query_gen_errors):
        ax.text(i - width / 2, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(db_exec_errors):
        ax.text(i + width / 2, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 차트 저장
    chart_path = self.output_dir / "error_analysis.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    logger.info(f"오류 분석 차트 저장됨: {chart_path}")

    return chart_path


def plot_latency_comparison(self):
    """지연시간 비교 차트 생성"""
    logger.info("지연시간 비교 차트 생성 중...")

    # 각 모델 및 데이터셋별 지연시간 데이터 수집
    latency_data = []

    for model in self.models:
        if not self.metrics.get(model):
            continue

        for dataset_type in self.dataset_types:
            if dataset_type not in self.metrics[model]:
                continue

            latency = self.metrics[model][dataset_type]['latency']

            latency_data.append({
                'model': model.upper(),
                'dataset': dataset_type.upper(),
                'latency': latency
            })

    if not latency_data:
        logger.warning("지연시간 차트를 위한 데이터가 없습니다")
        return None

    # DataFrame으로 변환
    df = pd.DataFrame(latency_data)

    # 그래프 생성
    plt.figure(figsize=(12, 6))

    # 색상 맵 설정
    colors = {'ADVANCED': '#1976D2', 'BASIC': '#43A047', 'CLASSIC': '#E53935'}

    # 그룹화된 막대 그래프
    bar_plot = sns.barplot(
        x='model',
        y='latency',
        hue='dataset',
        data=df,
        palette=colors
    )

    plt.title('모델 및 데이터셋별 평균 지연시간')
    plt.ylabel('지연시간 (초)')
    plt.xlabel('모델')

    # 값 표시
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt='%.2f')

    plt.tight_layout()

    # 차트 저장
    chart_path = self.output_dir / "latency_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()