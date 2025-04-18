import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import matplotlib

from utils import detect_models_and_datasets

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

import matplotlib.font_manager as fm

# main 함수 실행 전에 (파일 전역 범위에서) 설정
# 한글 폰트 경로 지정 - Windows 기준
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트
font_prop = fm.FontProperties(fname=font_path)

# matplotlib 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class SQLDataExplorer:
    """
    SQL 생성 모델의 성능 분석을 위한 GUI 도구
    """

    def __init__(self, root):
        self.root = root
        self.root.title("SQL 생성 모델 성능 분석기")
        self.root.geometry("1200x800")

        self.data = {}
        self.models = []  # 동적으로 결정될 모델명 리스트
        self.dataset_types = []  # 동적으로 결정될 데이터셋 유형 리스트
        self.metrics = {
            "accuracy": "정확도",
            "exact_match": "정확 일치율",
            "error_rate": "오류율",
            "latency": "지연시간(초)"
        }

        self.metrics_data = {}
        # 콤보박스 위젯에 대한 참조 저장
        self.model_dropdown = None

        self.create_widgets()

    def create_widgets(self):
        """GUI 위젯 생성"""
        # 상단 프레임 - 파일 로드 및 분석 제어
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="데이터 디렉토리:").grid(row=0, column=0, padx=5, pady=5)
        self.dir_entry = ttk.Entry(top_frame, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5)
        self.dir_entry.insert(0, ".")

        ttk.Button(top_frame, text="찾아보기", command=self.browse_directory).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(top_frame, text="데이터 로드", command=self.load_data).grid(row=0, column=3, padx=5, pady=5)

        # 탭 컨테이너
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 탭 1: 개요
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="개요")

        # 탭 2: 모델 비교
        self.comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_tab, text="모델 비교")

        # 비교 탭 내용
        comparison_control_frame = ttk.Frame(self.comparison_tab, padding="10")
        comparison_control_frame.pack(fill=tk.X)

        ttk.Label(comparison_control_frame, text="지표:").grid(row=0, column=0, padx=5, pady=5)
        self.metric_var = tk.StringVar(value="accuracy")
        metric_dropdown = ttk.Combobox(comparison_control_frame, textvariable=self.metric_var,
                                       values=list(self.metrics.keys()))
        metric_dropdown.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(comparison_control_frame, text="차트 업데이트",
                   command=lambda: self.update_comparison_chart()).grid(row=0, column=2, padx=5, pady=5)

        self.comparison_frame = ttk.Frame(self.comparison_tab)
        self.comparison_frame.pack(fill=tk.BOTH, expand=True)

        # 탭 3: 카테고리 분석
        self.category_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.category_tab, text="카테고리 분석")

        # 카테고리 탭 내용
        category_control_frame = ttk.Frame(self.category_tab, padding="10")
        category_control_frame.pack(fill=tk.X)

        ttk.Label(category_control_frame, text="모델:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="gemma3")
        # 콤보박스 위젯에 대한 참조 저장
        self.model_dropdown = ttk.Combobox(category_control_frame, textvariable=self.model_var,
                                     values=self.models)
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(category_control_frame, text="차트 업데이트",
                   command=lambda: self.update_category_chart()).grid(row=0, column=2, padx=5, pady=5)

        self.category_frame = ttk.Frame(self.category_tab)
        self.category_frame.pack(fill=tk.BOTH, expand=True)

        # 탭 4: 히트맵
        self.heatmap_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.heatmap_tab, text="히트맵")

        # 히트맵 탭 내용
        heatmap_control_frame = ttk.Frame(self.heatmap_tab, padding="10")
        heatmap_control_frame.pack(fill=tk.X)

        ttk.Label(heatmap_control_frame, text="지표:").grid(row=0, column=0, padx=5, pady=5)
        self.heatmap_metric_var = tk.StringVar(value="accuracy")
        heatmap_metric_dropdown = ttk.Combobox(heatmap_control_frame, textvariable=self.heatmap_metric_var,
                                               values=list(self.metrics.keys()))
        heatmap_metric_dropdown.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(heatmap_control_frame, text="차트 업데이트",
                   command=lambda: self.update_heatmap()).grid(row=0, column=2, padx=5, pady=5)

        self.heatmap_frame = ttk.Frame(self.heatmap_tab)
        self.heatmap_frame.pack(fill=tk.BOTH, expand=True)

        # 상태 표시줄
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var.set("준비")

    def browse_directory(self):
        """디렉토리 선택 다이얼로그"""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)

            # 선택한 디렉토리에서 모델과 데이터셋 유형 감지
            models, dataset_types = detect_models_and_datasets(directory)
            if models and dataset_types:
                self.models = models
                self.dataset_types = dataset_types
                self.status_var.set(f"감지된 모델: {self.models}, 데이터셋: {self.dataset_types}")

                # 모델 선택 드롭다운 업데이트 - 수정된 부분
                if self.model_dropdown is not None:
                    self.model_dropdown['values'] = self.models
                    if self.models:
                        self.model_var.set(self.models[0])

    def load_data(self):
        """선택한 디렉토리에서 CSV 파일 로드"""
        directory = self.dir_entry.get()
        base_path = Path(directory)

        self.status_var.set("데이터 로드 중...")
        self.root.update_idletasks()

        self.data = {}
        loaded_files = 0

        for model in self.models:
            self.data[model] = {}
            for dataset_type in self.dataset_types:
                file_path = base_path / f"{model}_{dataset_type}.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        self.data[model][dataset_type] = df
                        loaded_files += 1
                        self.status_var.set(f"{file_path.name} 로드 완료")
                        self.root.update_idletasks()
                    except Exception as e:
                        messagebox.showerror("오류", f"{file_path} 로드 중 오류 발생: {e}")

        if loaded_files > 0:
            self.calculate_metrics()
            self.update_overview()
            self.update_comparison_chart()
            self.update_category_chart()
            self.update_heatmap()
            self.status_var.set(f"{loaded_files}개 파일 로드 완료")
        else:
            self.status_var.set("로드할 파일이 없습니다")

    def calculate_metrics(self):
        """각 모델과 데이터셋에 대한 성능 지표 계산"""
        self.metrics_data = {}

        for model in self.models:
            self.metrics_data[model] = {}
            for dataset_type in self.dataset_types:
                if dataset_type not in self.data[model]:
                    continue

                df = self.data[model][dataset_type]

                # 필수 열이 있는지 확인
                required_cols = ['correct', 'exact_match', 'error_query_gen', 'error_db_exec', 'latency_seconds']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
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

    def update_overview(self):
        """개요 탭 업데이트"""
        # 기존 위젯 제거
        for widget in self.overview_tab.winfo_children():
            widget.destroy()

        # 모델별 평균 성능 표시
        ttk.Label(self.overview_tab, text="모델별 평균 성능", font=("Arial", 12, "bold")).pack(pady=10)

        tree = ttk.Treeview(self.overview_tab, columns=["model", "accuracy", "exact_match", "error_rate", "latency"],
                            show="headings")
        tree.heading("model", text="모델")
        tree.heading("accuracy", text="정확도")
        tree.heading("exact_match", text="정확 일치율")
        tree.heading("error_rate", text="오류율")
        tree.heading("latency", text="평균 지연시간(초)")

        tree.column("model", width=100, anchor="center")
        for column in ["accuracy", "exact_match", "error_rate"]:
            tree.column(column, width=130, anchor="center")
        tree.column("latency", width=150, anchor="center")

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 모델별 평균 성능 계산
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

                tree.insert("", "end", values=[
                    model.upper(),
                    f"{avg_accuracy:.2%}",
                    f"{avg_exact_match:.2%}",
                    f"{avg_error_rate:.2%}",
                    f"{avg_latency:.2f}"
                ])

        # 데이터셋 유형별 표 생성
        frame = ttk.Frame(self.overview_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 각 데이터셋 유형에 대한 라벨과 표
        for i, dataset_type in enumerate(self.dataset_types):
            dataset_frame = ttk.LabelFrame(frame, text=f"{dataset_type.upper()} 데이터셋 성능")
            dataset_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")

            dataset_tree = ttk.Treeview(dataset_frame,
                                        columns=["model", "accuracy", "exact_match", "error_rate", "latency"],
                                        show="headings", height=5)
            dataset_tree.heading("model", text="모델")
            dataset_tree.heading("accuracy", text="정확도")
            dataset_tree.heading("exact_match", text="정확 일치율")
            dataset_tree.heading("error_rate", text="오류율")
            dataset_tree.heading("latency", text="지연시간(초)")

            dataset_tree.column("model", width=80, anchor="center")
            for column in ["accuracy", "exact_match", "error_rate", "latency"]:
                dataset_tree.column(column, width=80, anchor="center")

            for model in self.models:
                if model in self.metrics_data and dataset_type in self.metrics_data[model]:
                    metrics = self.metrics_data[model][dataset_type]
                    dataset_tree.insert("", "end", values=[
                        model.upper(),
                        f"{metrics['accuracy']:.2%}",
                        f"{metrics['exact_match']:.2%}",
                        f"{metrics['error_rate']:.2%}",
                        f"{metrics['latency']:.2f}"
                    ])

            dataset_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            frame.columnconfigure(i, weight=1)

        frame.rowconfigure(0, weight=1)

    def update_comparison_chart(self):
        """모델 비교 차트 업데이트"""
        # 기존 위젯 제거
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        if not self.metrics_data:
            ttk.Label(self.comparison_frame, text="데이터가 로드되지 않았습니다").pack(expand=True)
            return

        metric = self.metric_var.get()
        metric_name = self.metrics[metric]

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

        if not models:
            ttk.Label(self.comparison_frame, text="비교할 데이터가 없습니다").pack(expand=True)
            return

        # 그래프 생성
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        x = np.arange(len(models))
        width = 0.25

        if advanced_values:
            ax.bar(x - width, advanced_values, width, label='Advanced')
        if basic_values:
            ax.bar(x, basic_values, width, label='Basic')
        if classic_values:
            ax.bar(x + width, classic_values, width, label='Classic')

        # 그래프 스타일 설정
        ax.set_ylabel(metric_name)
        ax.set_title(f'모델별 {metric_name} 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()

        # Y축 형식 지정
        if metric in ['accuracy', 'exact_match', 'error_rate']:
            ax.set_ylim(0, 1)
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # Tkinter에 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_category_chart(self):
        """카테고리 성능 차트 업데이트"""
        # 기존 위젯 제거
        for widget in self.category_frame.winfo_children():
            widget.destroy()

        model = self.model_var.get()

        if not self.metrics_data or model not in self.metrics_data:
            ttk.Label(self.category_frame, text="선택한 모델의 데이터가 없습니다").pack(expand=True)
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
            ttk.Label(self.category_frame, text="카테고리 데이터가 없습니다").pack(expand=True)
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

        # 샘플 수가 많은 순으로 정렬
        sorted_categories = sorted(categories, key=lambda x: category_count.get(x, 0), reverse=True)

        # 그래프 생성
        fig = Figure(figsize=(8, 10))  # 세로로 긴 그래프
        ax = fig.add_subplot(111)

        y_pos = np.arange(len(sorted_categories))
        accuracies = [category_avg.get(cat, 0) for cat in sorted_categories]

        # 색상 그라데이션
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_categories)))

        bars = ax.barh(y_pos, accuracies, color=colors)

        # 그래프 스타일 설정
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_categories)
        ax.set_xlabel('정확도')
        ax.set_title(f'{model.upper()} 모델의 카테고리별 성능')

        # 샘플 수 표시
        for i, v in enumerate(accuracies):
            ax.text(v + 0.01, i, f"{v:.2%} (n={category_count.get(sorted_categories[i], 0)})",
                    color='black', va='center')

        ax.set_xlim(0, 1)
        import matplotlib.ticker as mtick
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        fig.tight_layout()

        # Tkinter에 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=self.category_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_heatmap(self):
        """히트맵 업데이트"""
        # 기존 위젯 제거
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()

        if not self.metrics_data:
            ttk.Label(self.heatmap_frame, text="데이터가 로드되지 않았습니다").pack(expand=True)
            return

        metric = self.heatmap_metric_var.get()
        metric_name = self.metrics[metric]

        # 히트맵 데이터 준비
        heatmap_data = np.zeros((len(self.models), len(self.dataset_types)))

        for i, model in enumerate(self.models):
            if not self.metrics_data.get(model):
                continue

            for j, dataset_type in enumerate(self.dataset_types):
                if dataset_type not in self.metrics_data[model]:
                    continue

                heatmap_data[i, j] = self.metrics_data[model][dataset_type].get(metric, 0)

        # 그래프 생성
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # 히트맵 생성
        im = ax.imshow(heatmap_data, cmap='viridis' if metric != 'error_rate' else 'coolwarm_r')

        # 열과 행 라벨 설정
        ax.set_xticks(np.arange(len(self.dataset_types)))
        ax.set_yticks(np.arange(len(self.models)))
        ax.set_xticklabels(self.dataset_types)
        ax.set_yticklabels([m.upper() for m in self.models])

        # 격자 추가
        ax.set_xticks(np.arange(-.5, len(self.dataset_types), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(self.models), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

        # 색상바 추가
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric_name)

        # 값 표시
        for i in range(len(self.models)):
            for j in range(len(self.dataset_types)):
                value = heatmap_data[i, j]
                if metric in ['accuracy', 'exact_match', 'error_rate']:
                    text = f"{value:.2%}"
                else:
                    text = f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="white" if value > 0.5 else "black")

        ax.set_title(f'{metric_name} 히트맵')

        # Tkinter에 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()

    # Tkinter 앱 초기화 부분에 다음 코드 추가
    root.option_add("*Font", ("Malgun Gothic", 10))  # Windows에서 한글 지원 폰트

    app = SQLDataExplorer(root)

    root.mainloop()


if __name__ == "__main__":
    main()