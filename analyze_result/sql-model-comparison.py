import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import logging
import sys
from tabulate import tabulate

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SQL모델비교')


class SQLModelComparison:
    """
    여러 SQL 생성 모델의 성능을 비교하는 간단한 명령줄 도구
    """

    def __init__(self):
        self.data = {}
        self.models = ["gemma3", "hf", "ollama"]
        self.dataset_types = ["advanced", "basic", "classic"]

    def load_data(self, directory="."):
        """지정된 디렉토리에서 모든 CSV 파일 로드"""
        base_path = Path(directory)

        for model in self.models:
            self.data[model] = {}
            for dataset_type in self.dataset_types:
                file_path = base_path / f"{model}_{dataset_type}.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        self.data[model][dataset_type] = df
                        logger.info(f"로드 완료: {file_path.name} ({len(df)} 행)")
                    except Exception as e:
                        logger.error(f"파일 로드 오류: {file_path} - {e}")
                else:
                    logger.warning(f"파일 없음: {file_path}")

        # 로드된 파일이 있는지 확인
        loaded_files = sum(1 for model in self.data.values() for _ in model.values())
        if loaded_files == 0:
            logger.error("로드된 파일이 없습니다.")
            return False

        return True

    def calculate_metrics(self):
        """각 모델과 데이터셋에 대한 성능 지표 계산"""
        metrics = {}

        for model in self.models:
            metrics[model] = {}
            for dataset_type in self.dataset_types:
                if dataset_type not in self.data.get(model, {}):
                    continue

                df = self.data[model][dataset_type]

                # 필수 열이 있는지 확인
                required_cols = ['correct', 'exact_match', 'error_query_gen', 'error_db_exec', 'latency_seconds']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"{model}_{dataset_type}: 필수 열 누락")
                    continue

                # 지표 계산
                result = {}
                result['total_queries'] = len(df)
                result['accuracy'] = df['correct'].mean() if 'correct' in df else 0
                result['exact_match'] = df['exact_match'].mean() if 'exact_match' in df else 0

                # 오류율 계산
                result['error_query_gen'] = df['error_query_gen'].mean() if 'error_query_gen' in df else 0
                result['error_db_exec'] = df['error_db_exec'].mean() if 'error_db_exec' in df else 0
                result['error_rate'] = result['error_query_gen'] + result['error_db_exec']

                # 지연시간 계산
                if 'latency_seconds' in df:
                    latency_values = pd.to_numeric(df['latency_seconds'], errors='coerce')
                    result['latency'] = latency_values.mean()
                else:
                    result['latency'] = 0

                metrics[model][dataset_type] = result

        return metrics

    def print_model_comparison(self, metrics):
        """모델 간 성능 비교 출력"""
        print("\n" + "=" * 60)
        print(" " * 20 + "모델 성능 비교")
        print("=" * 60)

        # 모델별 평균 성능 계산
        model_avg = {}
        for model in self.models:
            if model not in metrics:
                continue

            total_accuracy = 0
            total_exact_match = 0
            total_error_rate = 0
            total_latency = 0
            dataset_count = 0

            for dataset_type in self.dataset_types:
                if dataset_type not in metrics[model]:
                    continue

                total_accuracy += metrics[model][dataset_type]['accuracy']
                total_exact_match += metrics[model][dataset_type]['exact_match']
                total_error_rate += metrics[model][dataset_type]['error_rate']
                total_latency += metrics[model][dataset_type]['latency']
                dataset_count += 1

            if dataset_count > 0:
                model_avg[model] = {
                    'accuracy': total_accuracy / dataset_count,
                    'exact_match': total_exact_match / dataset_count,
                    'error_rate': total_error_rate / dataset_count,
                    'latency': total_latency / dataset_count,
                    'dataset_count': dataset_count
                }

        # 테이블 형식으로 출력
        table_data = []
        headers = ["모델", "정확도", "정확 일치율", "오류율", "지연시간(초)", "데이터셋 수"]

        for model, avg in model_avg.items():
            table_data.append([
                model.upper(),
                f"{avg['accuracy']:.2%}",
                f"{avg['exact_match']:.2%}",
                f"{avg['error_rate']:.2%}",
                f"{avg['latency']:.2f}",
                avg['dataset_count']
            ])

        # 정확도 기준으로 정렬
        table_data.sort(key=lambda x: float(x[1].strip('%')) / 100, reverse=True)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n")

    def print_dataset_comparison(self, metrics, dataset_type):
        """특정 데이터셋 유형에 대한 모델 간 성능 비교 출력"""
        print(f"\n{'-' * 60}")
        print(f" {dataset_type.upper()} 데이터셋 성능 비교")
        print(f"{'-' * 60}")

        table_data = []
        headers = ["모델", "정확도", "정확 일치율", "오류율", "지연시간(초)", "쿼리 수"]

        for model in self.models:
            if model in metrics and dataset_type in metrics[model]:
                result = metrics[model][dataset_type]
                table_data.append([
                    model.upper(),
                    f"{result['accuracy']:.2%}",
                    f"{result['exact_match']:.2%}",
                    f"{result['error_rate']:.2%}",
                    f"{result['latency']:.2f}",
                    result['total_queries']
                ])

        # 정확도 기준으로 정렬
        table_data.sort(key=lambda x: float(x[1].strip('%')) / 100, reverse=True)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n")

    def analyze_categories(self, model=None):
        """카테고리별 성능 분석"""
        # 카테고리 데이터 수집
        categories = {}

        for m in self.models if model is None else [model]:
            if m not in self.data:
                continue

            for dataset_type in self.dataset_types:
                if dataset_type not in self.data[m]:
                    continue

                df = self.data[m][dataset_type]

                if 'query_category' not in df.columns or 'correct' not in df.columns:
                    continue

                for category, group in df.groupby('query_category'):
                    if pd.isna(category) or not category:
                        continue

                    if category not in categories:
                        categories[category] = {}

                    if m not in categories[category]:
                        categories[category][m] = {
                            'total': 0,
                            'correct': 0
                        }

                    categories[category][m]['total'] += len(group)
                    categories[category][m]['correct'] += group['correct'].sum()

        if not categories:
            logger.warning("카테고리 분석에 사용할 데이터가 없습니다.")
            return

        # 각 카테고리의 정확도 계산
        category_accuracies = {}
        category_totals = {}

        for category, model_data in categories.items():
            category_totals[category] = sum(data['total'] for data in model_data.values())

            category_accuracies[category] = {}
            for m, data in model_data.items():
                if data['total'] > 0:
                    category_accuracies[category][m] = data['correct'] / data['total']
                else:
                    category_accuracies[category][m] = 0

        # 샘플 수가 많은 순으로 정렬
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)

        # 결과 출력
        print("\n" + "=" * 70)
        print(" " * 20 + "카테고리별 성능 분석")
        print("=" * 70)

        table_data = []
        headers = ["카테고리", "샘플 수"] + [m.upper() for m in self.models if m in self.data]

        for category, total in sorted_categories:
            row = [category, total]

            for m in self.models:
                if m in self.data and m in category_accuracies[category]:
                    row.append(f"{category_accuracies[category][m]:.2%}")
                else:
                    row.append("N/A")

            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n")

    def plot_model_comparison(self, metrics, output_file=None):
        """모델 간 성능 비교 차트 생성"""
        # 모델별 평균 성능 계산
        model_avg = {}
        for model in self.models:
            if model not in metrics:
                continue

            total_accuracy = 0
            total_exact_match = 0
            total_error_rate = 0
            dataset_count = 0

            for dataset_type in self.dataset_types:
                if dataset_type not in metrics[model]:
                    continue

                total_accuracy += metrics[model][dataset_type]['accuracy']
                total_exact_match += metrics[model][dataset_type]['exact_match']
                total_error_rate += metrics[model][dataset_type]['error_rate']
                dataset_count += 1

            if dataset_count > 0:
                model_avg[model] = {
                    'accuracy': total_accuracy / dataset_count,
                    'exact_match': total_exact_match / dataset_count,
                    'error_rate': total_error_rate / dataset_count
                }

        if not model_avg:
            logger.warning("차트 생성을 위한 데이터가 없습니다.")
            return

        # 데이터 준비
        models = list(model_avg.keys())
        accuracies = [model_avg[m]['accuracy'] for m in models]
        exact_matches = [model_avg[m]['exact_match'] for m in models]
        error_rates = [model_avg[m]['error_rate'] for m in models]

        # 그래프 생성
        plt.figure(figsize=(12, 8))

        x = np.arange(len(models))
        width = 0.25

        plt.bar(x - width, accuracies, width, label='정확도')
        plt.bar(x, exact_matches, width, label='정확 일치율')
        plt.bar(x + width, error_rates, width, label='오류율')

        plt.ylabel('비율')
        plt.title('모델별 평균 성능 비교')
        plt.xticks(x, [m.upper() for m in models])
        plt.legend()

        # Y축 형식 지정
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            logger.info(f"차트 저장됨: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_dataset_comparison(self, metrics, dataset_type, output_file=None):
        """특정 데이터셋 유형에 대한 모델 간 성능 비교 차트 생성"""
        # 데이터 준비
        models = []
        accuracies = []
        exact_matches = []
        error_rates = []

        for model in self.models:
            if model in metrics and dataset_type in metrics[model]:
                models.append(model.upper())
                result = metrics[model][dataset_type]
                accuracies.append(result['accuracy'])
                exact_matches.append(result['exact_match'])
                error_rates.append(result['error_rate'])

        if not models:
            logger.warning(f"{dataset_type} 데이터셋 차트를 위한 데이터가 없습니다.")
            return

        # 그래프 생성
        plt.figure(figsize=(12, 8))

        x = np.arange(len(models))
        width = 0.25

        plt.bar(x - width, accuracies, width, label='정확도')
        plt.bar(x, exact_matches, width, label='정확 일치율')
        plt.bar(x + width, error_rates, width, label='오류율')

        plt.ylabel('비율')
        plt.title(f'{dataset_type.upper()} 데이터셋 성능 비교')
        plt.xticks(x, models)
        plt.legend()

        # Y축 형식 지정
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            logger.info(f"차트 저장됨: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_category_heatmap(self, output_file=None):
        """카테고리별 모델 성능 히트맵 생성"""
        # 카테고리 데이터 수집
        categories = {}

        for model in self.models:
            if model not in self.data:
                continue

            for dataset_type in self.dataset_types:
                if dataset_type not in self.data[model]:
                    continue

                df = self.data[model][dataset_type]

                if 'query_category' not in df.columns or 'correct' not in df.columns:
                    continue

                for category, group in df.groupby('query_category'):
                    if pd.isna(category) or not category:
                        continue

                    if category not in categories:
                        categories[category] = {}

                    if model not in categories[category]:
                        categories[category][model] = {
                            'total': 0,
                            'correct': 0
                        }

                    categories[category][model]['total'] += len(group)
                    categories[category][model]['correct'] += group['correct'].sum()

        if not categories:
            logger.warning("히트맵 생성을 위한 카테고리 데이터가 없습니다.")
            return

        # 각 카테고리의 정확도 계산
        category_accuracies = {}
        category_totals = {}

        for category, model_data in categories.items():
            category_totals[category] = sum(data['total'] for data in model_data.values())

            category_accuracies[category] = {}
            for model, data in model_data.items():
                if data['total'] > 0:
                    category_accuracies[category][model] = data['correct'] / data['total']
                else:
                    category_accuracies[category][model] = 0

        # 샘플 수가 많은 상위 카테고리 선택 (최대 15개)
        top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:15]
        top_category_names = [cat for cat, _ in top_categories]

        # 데이터 준비
        category_data = []
        for category in top_category_names:
            for model in self.models:
                if model in category_accuracies[category]:
                    category_data.append({
                        'category': category,
                        'model': model,
                        'accuracy': category_accuracies[category][model]
                    })

        # DataFrame으로 변환
        df = pd.DataFrame(category_data)

        if df.empty:
            logger.warning("히트맵 생성을 위한 충분한 데이터가 없습니다.")
            return

        # 피벗 테이블 생성
        pivot_table = df.pivot_table(
            index='category',
            columns='model',
            values='accuracy',
            aggfunc='mean'
        )

        # 히트맵 생성
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.2%',
            cmap='viridis',
            linewidths=.5,
            cbar_kws={'label': '정확도'}
        )

        plt.title('카테고리별 모델 성능 히트맵')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            logger.info(f"히트맵 저장됨: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_all_charts(self, output_dir='.'):
        """모든 차트 생성 및 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 성능 지표 계산
        metrics = self.calculate_metrics()

        # 1. 모델 비교 차트
        self.plot_model_comparison(
            metrics,
            output_file=output_path / 'model_comparison.png'
        )

        # 2. 데이터셋별 차트
        for dataset_type in self.dataset_types:
            self.plot_dataset_comparison(
                metrics,
                dataset_type,
                output_file=output_path / f'dataset_{dataset_type}.png'
            )

        # 3. 카테고리 히트맵
        self.plot_category_heatmap(
            output_file=output_path / 'category_heatmap.png'
        )

        logger.info(f"모든 차트가 {output_path}에 저장되었습니다.")

    def run_comparison(self, args):
        """비교 도구 실행"""
        # 데이터 로드
        if not self.load_data(args.input):
            return

        # 성능 지표 계산
        metrics = self.calculate_metrics()

        # 전체 모델 비교 출력
        self.print_model_comparison(metrics)

        # 데이터셋별 비교 출력
        for dataset_type in self.dataset_types:
            self.print_dataset_comparison(metrics, dataset_type)

        # 카테고리 분석
        self.analyze_categories(args.model)

        # 차트 생성 (--charts 옵션이 있는 경우)
        if args.charts:
            self.plot_all_charts(args.output)


def main():
    parser = argparse.ArgumentParser(description='SQL 생성 모델 성능 비교 도구')
    parser.add_argument('--input', type=str, default='.', help='CSV 파일이 있는 디렉토리 경로')
    parser.add_argument('--output', type=str, default='./charts', help='차트를 저장할 디렉토리 경로')
    parser.add_argument('--model', type=str, help='특정 모델에 대한 분석 (옵션)')
    parser.add_argument('--charts', action='store_true', help='차트 생성 여부')

    args = parser.parse_args()

    tool = SQLModelComparison()
    tool.run_comparison(args)


if __name__ == "__main__":
    main()