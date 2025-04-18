import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from datetime import datetime
import jinja2
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SQL성능보고서')


class SQLPerformanceReportGenerator:
    """
    SQL 생성 모델의 성능 데이터를 분석하고 종합 보고서를 생성하는 도구
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.models = ["gemma3", "hf", "ollama"]
        self.dataset_types = ["advanced", "basic", "classic"]
        self.data = {}
        self.metrics = {}

        # 보고서 템플릿 초기화
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('.'),
            autoescape=jinja2.select_autoescape(['html'])
        )

        # 기본 템플릿 생성
        self._create_default_template()

    def _create_default_template(self):
        """기본 HTML 템플릿 생성"""
        template_path = Path('report_template.html')

        if not template_path.exists():
            template_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SQL 생성 모델 성능 분석 보고서</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }

        h1, h2, h3 {
            color: #2c3e50;
        }

        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        h2 {
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }

        th, td {
            text-align: left;
            padding: 12px;
            border: 1px solid #ddd;
        }

        th {
            background-color: #f2f2f2;
        }

        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .highlight {
            background-color: #e6f7ff;
            font-weight: bold;
        }

        .chart-container {
            margin: 20px 0;
            text-align: center;
        }

        .chart {
            max-width: 100%;
            height: auto;
        }

        .summary-box {
            background-color: #f0f7fb;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .grid-item {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }

        footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.8em;
            color: #777;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <h1>SQL 생성 모델 성능 분석 보고서</h1>
    <p><strong>생성일:</strong> {{ date }}</p>

    <div class="summary-box">
        <h2>종합 요약</h2>
        <p>{{ summary }}</p>
        <ul>
            <li><strong>최고 성능 모델:</strong> {{ best_model }}</li>
            <li><strong>분석된 총 쿼리 수:</strong> {{ total_queries }}</li>
            <li><strong>모델 평균 정확도:</strong> {{ avg_accuracy }}</li>
        </ul>
    </div>

    <h2>모델별 성능 비교</h2>
    <div class="chart-container">
        <img src="{{ model_comparison_chart }}" alt="모델별 성능 비교" class="chart">
    </div>

    <h2>모델별 평균 성능</h2>
    <table>
        <tr>
            <th>모델</th>
            <th>정확도</th>
            <th>정확 일치율</th>
            <th>오류율</th>
            <th>평균 지연시간(초)</th>
        </tr>
        {% for model in model_metrics %}
        <tr {% if model.is_best %}class="highlight"{% endif %}>
            <td>{{ model.name }}</td>
            <td>{{ model.accuracy }}</td>
            <td>{{ model.exact_match }}</td>
            <td>{{ model.error_rate }}</td>
            <td>{{ model.latency }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>데이터셋 유형별 성능</h2>
    <div class="grid-container">
        {% for dataset in dataset_charts %}
        <div class="grid-item">
            <h3>{{ dataset.name }}</h3>
            <div class="chart-container">
                <img src="{{ dataset.chart }}" alt="{{ dataset.name }} 성능" class="chart">
            </div>
        </div>
        {% endfor %}
    </div>

    <h2>카테고리별 성능 분석</h2>
    <div class="chart-container">
        <img src="{{ category_chart }}" alt="카테고리별 성능" class="chart">
    </div>
    <p>상위 5개 카테고리의 모델별 정확도:</p>
    <table>
        <tr>
            <th>카테고리</th>
            {% for model in models %}
            <th>{{ model }}</th>
            {% endfor %}
        </tr>
        {% for category in top_categories %}
        <tr>
            <td>{{ category.name }}</td>
            {% for acc in category.accuracies %}
            <td>{{ acc }}</td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table>

    <h2>오류 분석</h2>
    <div class="chart-container">
        <img src="{{ error_chart }}" alt="오류 유형 분석" class="chart">
    </div>

    <h2>지연시간 분석</h2>
    <div class="chart-container">
        <img src="{{ latency_chart }}" alt="지연시간 분석" class="chart">
    </div>

    <footer>
        자동 생성된 보고서 | {{ date }}
    </footer>
</body>
</html>
            """
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)

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

        self.metrics = {}
        total_queries = 0

        for model in self.models:
            self.metrics[model] = {}
            for dataset_type in self.dataset_types:
                if dataset_type not in self.data[model]:
                    continue

                df = self.data[model][dataset_type]
                total_queries += len(df)

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

        self.total_queries = total_queries
        logger.info(f"지표 계산 완료: 총 {total_queries}개 쿼리 분석됨")

        return self

    def _calculate_model_avg_metrics(self):
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

    def generate_model_comparison_chart(self):
        """모델 비교 차트 생성"""
        logger.info("모델 비교 차트 생성 중...")

        model_avg = self._calculate_model_avg_metrics()

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

        ax.bar(x - width, accuracies, width, label='정확도')
        ax.bar(x, exact_matches, width, label='정확 일치율')
        ax.bar(x + width, error_rates, width, label='오류율')

        ax.set_ylabel('비율')
        ax.set_title('모델별 성능 비교')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend()

        # Y축 형식 지정
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        # 차트 저장
        chart_path = self.output_dir / "model_comparison.png"
        plt.savefig(chart_path)
        plt.close()

        logger.info(f"모델 비교 차트 저장됨: {chart_path}")

        return chart_path.name

    def generate_dataset_charts(self):
        """데이터셋 유형별 차트 생성"""
        logger.info("데이터셋 유형별 차트 생성 중...")

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
                continue

            # 그래프 생성
            fig, ax = plt.subplots(figsize=(8, 5))

            x = np.arange(len(models))
            width = 0.25

            ax.bar(x - width, accuracies, width, label='정확도')
            ax.bar(x, exact_matches, width, label='정확 일치율')
            ax.bar(x + width, error_rates, width, label='오류율')

            ax.set_ylabel('비율')
            ax.set_title(f'{dataset_type.upper()} 데이터셋 성능')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()

            # Y축 형식 지정
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            plt.tight_layout()

            # 차트 저장
            chart_path = self.output_dir / f"dataset_{dataset_type}.png"
            plt.savefig(chart_path)
            plt.close()

            logger.info(f"{dataset_type} 데이터셋 차트 저장됨: {chart_path}")

            chart_paths.append({
                'name': f'{dataset_type.upper()} 데이터셋',
                'chart': chart_path.name
            })

        return chart_paths

    def generate_category_chart(self):
        """카테고리별 성능 차트 생성"""
        logger.info("카테고리별 성능 차트 생성 중...")

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
            logger.warning("카테고리 차트를 위한 데이터가 없습니다")
            return None, []

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

        # 샘플 수가 많은 상위 카테고리 선택 (최대 15개)
        top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:15]
        top_category_names = [cat for cat, _ in top_categories]

        # 그래프 생성
        plt.figure(figsize=(12, 10))

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
            logger.warning("카테고리 차트를 위한 충분한 데이터가 없습니다")
            return None, []

        # Seaborn 히트맵으로 표시
        pivot_table = df.pivot_table(
            index='category',
            columns='model',
            values='accuracy',
            aggfunc='mean'
        )

        plt.figure(figsize=(10, 8))
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
        chart_path = self.output_dir / "category_performance.png"
        plt.savefig(chart_path)
        plt.close()

        logger.info(f"카테고리 성능 차트 저장됨: {chart_path}")

        # 상위 5개 카테고리에 대한 데이터 준비
        top5_categories = []
        for i, (category, _) in enumerate(top_categories[:5]):
            accuracies = []
            for model in self.models:
                if model in category_accuracies[category]:
                    acc = category_accuracies[category][model]
                    accuracies.append(f"{acc:.2%}")
                else:
                    accuracies.append("N/A")

            top5_categories.append({
                'name': category,
                'accuracies': accuracies
            })

        return chart_path.name, top5_categories

    def generate_error_chart(self):
        """오류 유형 분석 차트 생성"""
        logger.info("오류 유형 분석 차트 생성 중...")

        model_avg = self._calculate_model_avg_metrics()

        if not model_avg:
            logger.warning("오류 차트를 위한 데이터가 없습니다")
            return None

        # 데이터 준비
        models = list(model_avg.keys())
        query_gen_errors = []
        db_exec_errors = []

        for model in models:
            # 모든 데이터셋의 오류율 평균 계산
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

        ax.bar(x - width / 2, query_gen_errors, width, label='쿼리 생성 오류')
        ax.bar(x + width / 2, db_exec_errors, width, label='DB 실행 오류')

        ax.set_ylabel('오류율')
        ax.set_title('모델별 오류 유형 분석')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend()

        # Y축 형식 지정
        ax.set_ylim(0, max(max(query_gen_errors), max(db_exec_errors)) * 1.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()

        # 차트 저장
        chart_path = self.output_dir / "error_analysis.png"
        plt.savefig(chart_path)
        plt.close()

        logger.info(f"오류 분석 차트 저장됨: {chart_path}")

        return chart_path.name

    def generate_latency_chart(self):
        """지연시간 분석 차트 생성"""
        logger.info("지연시간 분석 차트 생성 중...")

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
        plt.figure(figsize=(10, 6))

        bar_plot = sns.barplot(x='model', y='latency', hue='dataset', data=df)

        plt.title('모델 및 데이터셋별 평균 지연시간')
        plt.ylabel('지연시간 (초)')
        plt.xlabel('모델')

        # 값 표시
        for container in bar_plot.containers:
            bar_plot.bar_label(container, fmt='%.2f')

        plt.tight_layout()

        # 차트 저장
        chart_path = self.output_dir / "latency_analysis.png"
        plt.savefig(chart_path)
        plt.close()

        logger.info(f"지연시간 분석 차트 저장됨: {chart_path}")

        return chart_path.name

    def generate_report(self):
        """HTML 보고서 생성"""
        logger.info("HTML 보고서 생성 중...")

        # 필요한 차트와 데이터 생성
        model_comparison_chart = self.generate_model_comparison_chart()
        dataset_charts = self.generate_dataset_charts()
        category_chart, top_categories = self.generate_category_chart()
        error_chart = self.generate_error_chart()
        latency_chart = self.generate_latency_chart()

        model_avg = self._calculate_model_avg_metrics()

        # 모델별 평균 성능 데이터 준비
        if model_avg:
            # 최고 성능 모델 찾기
            best_model = max(model_avg.items(), key=lambda x: x[1]['accuracy'])
            best_model_name = best_model[0].upper()

            # 전체 평균 정확도
            all_accuracies = [m['accuracy'] for m in model_avg.values()]
            avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0

            # 모델 지표 데이터
            model_metrics = []
            for model, metrics in model_avg.items():
                model_metrics.append({
                    'name': model.upper(),
                    'accuracy': f"{metrics['accuracy']:.2%}",
                    'exact_match': f"{metrics['exact_match']:.2%}",
                    'error_rate': f"{metrics['error_rate']:.2%}",
                    'latency': f"{metrics['latency']:.2f}",
                    'is_best': model.upper() == best_model_name
                })

            # 종합 요약 작성
            summary = f"""
            이 보고서는 {len(self.models)}개의 SQL 생성 모델({', '.join([m.upper() for m in self.models])})에 대한 
            성능 분석 결과입니다. 총 {self.total_queries}개의 쿼리에 대한 분석이 수행되었으며, 
            {best_model_name} 모델이 {best_model[1]['accuracy']:.2%}의 정확도로 가장 높은 성능을 보여주었습니다.
            """
        else:
            best_model_name = "N/A"
            avg_accuracy = 0
            model_metrics = []
            summary = "분석을 위한 충분한 데이터가 없습니다."

        # HTML 템플릿 로딩 및 렌더링
        template = self.template_env.get_template('report_template.html')

        html_content = template.render(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            summary=summary,
            best_model=best_model_name,
            total_queries=self.total_queries,
            avg_accuracy=f"{avg_accuracy:.2%}",
            model_comparison_chart=model_comparison_chart,
            model_metrics=model_metrics,
            dataset_charts=dataset_charts,
            category_chart=category_chart,
            top_categories=top_categories,
            models=[m.upper() for m in self.models],
            error_chart=error_chart,
            latency_chart=latency_chart
        )

        # HTML 파일 저장
        report_path = self.output_dir / "performance_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML 보고서 생성 완료: {report_path}")

        return report_path


def main():
    parser = argparse.ArgumentParser(description='SQL 생성 모델 성능 보고서 생성기')
    parser.add_argument('--input', type=str, default='.', help='CSV 파일이 있는 디렉토리 경로')
    parser.add_argument('--output', type=str, default='./report', help='보고서를 저장할 디렉토리 경로')

    args = parser.parse_args()

    generator = SQLPerformanceReportGenerator(args.input, args.output)
    generator.load_data()
    generator.calculate_metrics()
    report_path = generator.generate_report()

    print(f"보고서 생성이 완료되었습니다: {report_path}")


if __name__ == "__main__":
    main()