import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

from utils import detect_models_and_datasets


class SQLPerformanceAnalyzer:
    """
    여러 모델의 SQL 생성 성능을 분석하고 시각화하는 도구
    """

    def __init__(self):
        self.data = {}
        self.models = []  # 동적으로 결정될 모델명 리스트
        self.dataset_types = []  # 동적으로 결정될 데이터셋 유형 리스트
        self.metrics = ["accuracy", "exact_match", "error_rate", "latency"]

    def load_data(self, directory="."):
        """
        지정된 디렉토리에서 모든 CSV 파일 로드
        """
        base_path = Path(directory)

        # 디렉토리에서 모델과 데이터셋 유형 감지
        self.models, self.dataset_types = detect_models_and_datasets(directory)

        if not self.models or not self.dataset_types:
            print("No valid CSV files found or naming pattern is incorrect.")
            return False

        print(f"Detected models: {self.models}")
        print(f"Detected dataset types: {self.dataset_types}")

        for model in self.models:
            self.data[model] = {}
            for dataset_type in self.dataset_types:
                file_path = base_path / f"{model}_{dataset_type}.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        self.data[model][dataset_type] = df
                        print(f"Loaded {file_path} - {len(df)} rows")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"File not found: {file_path}")

        return self

    def calculate_metrics(self):
        """
        각 모델과 데이터셋에 대한 성능 지표 계산
        """
        self.metrics_data = {}

        for model in self.models:
            self.metrics_data[model] = {}
            for dataset_type in self.dataset_types:
                if dataset_type not in self.data[model]:
                    continue

                df = self.data[model][dataset_type]

                # 필수 열이 있는지 확인
                required_cols = ['correct', 'exact_match', 'error_query_gen', 'error_db_exec', 'latency_seconds']
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {model}_{dataset_type}: Missing required columns")
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
                    # 지연시간이 문자열로 저장될 수 있으므로 숫자로 변환
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

                self.metrics_data[model][dataset_type] = metrics

        return self

    def generate_overview_report(self):
        """
        모델별 평균 성능 요약 보고서 생성
        """
        report = []
        report.append("=" * 50)
        report.append("SQL 생성 모델 성능 분석 보고서")
        report.append("=" * 50)
        report.append("")

        # 모델별 평균 성능
        report.append("-" * 50)
        report.append("모델별 평균 성능")
        report.append("-" * 50)

        for model in self.models:
            if not self.metrics_data.get(model):
                continue

            total_accuracy = 0
            total_exact_match = 0
            total_error_rate = 0
            total_latency = 0
            dataset_count = 0

            for dataset_type in self.dataset_types:
                if dataset_type not in self.metrics_data[model]:
                    continue

                metrics = self.metrics_data[model][dataset_type]
                total_accuracy += metrics['accuracy']
                total_exact_match += metrics['exact_match']
                total_error_rate += metrics['error_rate']
                total_latency += metrics['latency']
                dataset_count += 1

            if dataset_count > 0:
                avg_accuracy = total_accuracy / dataset_count
                avg_exact_match = total_exact_match / dataset_count
                avg_error_rate = total_error_rate / dataset_count
                avg_latency = total_latency / dataset_count

                report.append(f"\n모델: {model.upper()}")
                report.append(f"  - 평균 정확도: {avg_accuracy:.2%}")
                report.append(f"  - 평균 정확 일치율: {avg_exact_match:.2%}")
                report.append(f"  - 평균 오류율: {avg_error_rate:.2%}")
                report.append(f"  - 평균 지연시간: {avg_latency:.2f}초")

        # 데이터셋 유형별 비교
        for dataset_type in self.dataset_types:
            report.append(f"\n\n{'-' * 50}")
            report.append(f"{dataset_type.upper()} 데이터셋 성능 비교")
            report.append(f"{'-' * 50}")

            for model in self.models:
                if not self.metrics_data.get(model) or dataset_type not in self.metrics_data[model]:
                    continue

                metrics = self.metrics_data[model][dataset_type]
                report.append(f"\n모델: {model.upper()}")
                report.append(f"  - 정확도: {metrics['accuracy']:.2%}")
                report.append(f"  - 정확 일치율: {metrics['exact_match']:.2%}")
                report.append(f"  - 오류율: {metrics['error_rate']:.2%}")
                report.append(f"  - 지연시간: {metrics['latency']:.2f}초")

        return "\n".join(report)

    def plot_model_comparison(self, metric='accuracy', output_file=None):
        """
        모델 간 성능 비교 시각화
        """
        plt.figure(figsize=(12, 8))

        # 데이터 준비
        models = []
        advanced_values = []
        basic_values = []
        classic_values = []

        for model in self.models:
            if not self.metrics_data.get(model):
                continue

            models.append(model.upper())

            for dataset_type in self.dataset_types:
                if dataset_type not in self.metrics_data[model]:
                    continue

                value = self.metrics_data[model][dataset_type].get(metric, 0)

                if dataset_type == 'advanced':
                    advanced_values.append(value)
                elif dataset_type == 'basic':
                    basic_values.append(value)
                elif dataset_type == 'classic':
                    classic_values.append(value)

        # 막대 그래프 그리기
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        if advanced_values:
            ax.bar(x - width, advanced_values, width, label='Advanced')
        if basic_values:
            ax.bar(x, basic_values, width, label='Basic')
        if classic_values:
            ax.bar(x + width, classic_values, width, label='Classic')

        # 그래프 스타일 설정
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'모델별 {metric.replace("_", " ").title()} 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()

        # Y축 형식 지정
        if metric in ['accuracy', 'exact_match', 'error_rate']:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_category_performance(self, model='gemma3', output_file=None):
        """
        특정 모델의 카테고리별 성능 시각화
        """
        if not self.metrics_data.get(model):
            print(f"No data available for model: {model}")
            return

        # 모든 카테고리와 성능 데이터 수집
        categories = set()
        category_data = {}

        for dataset_type in self.dataset_types:
            if dataset_type not in self.metrics_data[model]:
                continue

            metrics = self.metrics_data[model][dataset_type]
            if 'category_performance' not in metrics:
                continue

            for category, perf in metrics['category_performance'].items():
                categories.add(category)
                if category not in category_data:
                    category_data[category] = []
                category_data[category].append((dataset_type, perf['accuracy'], perf['total']))

        if not categories:
            print(f"No category data available for model: {model}")
            return

        # 각 카테고리의 평균 정확도 계산
        category_avg = {}
        category_count = {}

        for category, data_points in category_data.items():
            total_weighted_accuracy = 0
            total_samples = 0

            for _, accuracy, count in data_points:
                total_weighted_accuracy += accuracy * count
                total_samples += count

            if total_samples > 0:
                category_avg[category] = total_weighted_accuracy / total_samples
                category_count[category] = total_samples

        # 그래프 그리기 (샘플 수가 많은 순으로 정렬)
        sorted_categories = sorted(categories, key=lambda x: category_count.get(x, 0), reverse=True)

        plt.figure(figsize=(12, 8))

        # 막대 그래프 그리기
        y_pos = np.arange(len(sorted_categories))
        accuracies = [category_avg.get(cat, 0) for cat in sorted_categories]

        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_categories)))

        bars = plt.barh(y_pos, accuracies, color=colors)

        # 그래프 스타일 설정
        plt.yticks(y_pos, sorted_categories)
        plt.xlabel('Accuracy')
        plt.title(f'{model.upper()} 모델의 카테고리별 성능')

        # 샘플 수 표시
        for i, v in enumerate(accuracies):
            plt.text(v + 0.01, i, f"{v:.2%} (n={category_count.get(sorted_categories[i], 0)})",
                     color='black', va='center')

        plt.xlim(0, 1)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_performance_heatmap(self, metric='accuracy', output_file=None):
        """
        모델과 데이터셋 유형에 따른 성능 히트맵 생성
        """
        # 히트맵 데이터 준비
        heatmap_data = np.zeros((len(self.models), len(self.dataset_types)))

        for i, model in enumerate(self.models):
            if not self.metrics_data.get(model):
                continue

            for j, dataset_type in enumerate(self.dataset_types):
                if dataset_type not in self.metrics_data[model]:
                    continue

                heatmap_data[i, j] = self.metrics_data[model][dataset_type].get(metric, 0)

        # 히트맵 그리기
        plt.figure(figsize=(10, 8))

        ax = sns.heatmap(heatmap_data,
                         annot=True,
                         fmt='.2%' if metric in ['accuracy', 'exact_match', 'error_rate'] else '.2f',
                         xticklabels=self.dataset_types,
                         yticklabels=[m.upper() for m in self.models],
                         cmap='viridis' if metric != 'error_rate' else 'coolwarm_r')

        plt.title(f'{metric.replace("_", " ").title()} 히트맵')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def run_analysis(self, output_dir='.'):
        """
        전체 분석 수행 및 결과 저장
        """
        # 결과 저장 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 보고서 생성
        report = self.generate_overview_report()
        with open(output_path / 'performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output_path / 'performance_report.txt'}")

        # 그래프 생성
        for metric in ['accuracy', 'exact_match', 'error_rate', 'latency']:
            self.plot_model_comparison(
                metric=metric,
                output_file=output_path / f'model_comparison_{metric}.png'
            )

            self.plot_performance_heatmap(
                metric=metric,
                output_file=output_path / f'heatmap_{metric}.png'
            )

        # 각 모델별 카테고리 성능
        for model in self.models:
            if self.metrics_data.get(model):
                self.plot_category_performance(
                    model=model,
                    output_file=output_path / f'category_performance_{model}.png'
                )

        print(f"Analysis completed. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SQL 생성 모델 성능 분석 도구')
    parser.add_argument('--dir', type=str, default='.', help='CSV 파일이 있는 디렉토리 경로')
    parser.add_argument('--output', type=str, default='./analysis_results', help='결과를 저장할 디렉토리 경로')

    args = parser.parse_args()

    analyzer = SQLPerformanceAnalyzer()
    analyzer.load_data(args.dir)
    analyzer.calculate_metrics()
    analyzer.run_analysis(args.output)

    print("분석이 완료되었습니다.")


if __name__ == "__main__":
    main()