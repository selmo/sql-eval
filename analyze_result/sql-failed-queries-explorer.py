import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import difflib
import re
import os
import platform
import matplotlib.font_manager as fm
import time  # 로그 타임스탬프용 추가

from utils import detect_models_and_datasets

matplotlib.use('TkAgg')


def setup_korean_font():
    """
    시스템에 맞는 한글 폰트를 설정합니다.
    """
    system = platform.system()

    # 운영체제별 기본 한글 폰트 경로
    if system == 'Windows':
        # 윈도우의 경우 맑은 고딕 사용
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
        font_family = 'Malgun Gothic'
    elif system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # Apple SD Gothic Neo
        font_family = 'AppleSDGothicNeo'
    else:  # Linux 등
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 나눔고딕
        font_family = 'NanumGothic'

    # 폰트 경로가 유효한지 확인
    try:
        fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_family
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        print(f"한글 폰트 설정 완료: {font_family}")
        return True
    except:
        print("기본 한글 폰트를 찾을 수 없습니다.")

        # 설치된 폰트 중에서 한글 폰트 찾기
        fonts = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = [f for f in fonts if any(keyword in f for keyword in
                                                ['Gothic', 'Malgun', '고딕', 'Nanum', '나눔', 'Batang', '바탕', 'Gulim',
                                                 '굴림'])]

        if korean_fonts:
            plt.rcParams['font.family'] = korean_fonts[0]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"대체 한글 폰트 설정: {korean_fonts[0]}")
            return True
        else:
            print("사용 가능한 한글 폰트를 찾을 수 없습니다.")
            print("한글이 깨질 수 있습니다. 필요시 나눔고딕 등의 한글 폰트를 설치하세요.")
            return False


# 한글 폰트 설정
setup_korean_font()


class SQLFailedQueriesExplorer:
    """
    SQL 생성 모델의 실패 사례를 분석하고 시각화하는 도구
    """

    def __init__(self, root):
        self.root = root
        self.root.title("SQL 생성 모델 실패 쿼리 분석기")
        self.root.geometry("1400x900")

        self.data = {}
        self.models = []  # 동적으로 결정될 모델명 리스트
        self.dataset_types = []  # 동적으로 결정될 데이터셋 유형 리스트

        # 선택된 항목을 저장하는 변수들
        self.selected_model = tk.StringVar()
        self.selected_dataset = tk.StringVar()
        self.selected_error_type = tk.StringVar(value="모든 오류")
        self.selected_category = tk.StringVar(value="모든 카테고리")

        # 실패 쿼리 데이터 저장
        self.failed_queries = {}
        self.filtered_queries = []
        self.error_examples = {}

        self.create_widgets()

        # 오류 분석 탭 초기화
        self.update_error_analysis_tab()

    def create_log_window(self):
        """로그 창 생성"""
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("로드 로그")
        self.log_window.geometry("800x400")
        self.log_window.protocol("WM_DELETE_WINDOW", lambda: self.log_window.withdraw())  # 닫기 버튼 누르면 숨기기

        # 툴바 프레임
        toolbar = ttk.Frame(self.log_window)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        # 툴바 버튼
        ttk.Button(toolbar, text="로그 지우기", command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="로그 저장", command=self.save_log).pack(side=tk.LEFT, padx=2)

        # 로그 텍스트 영역
        self.log_text = scrolledtext.ScrolledText(self.log_window, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 태그 설정 (오류 메시지는 빨간색, 경고는 주황색으로 표시)
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("success", foreground="green")

    def log_message(self, message, level="info"):
        """로그 메시지 추가"""
        if not hasattr(self, 'log_text'):
            return

        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, formatted_msg)

        # 레벨에 따라 태그 적용
        if level == "error":
            self.log_text.tag_add("error", f"end-{len(formatted_msg) + 1}c", "end-1c")
        elif level == "warning":
            self.log_text.tag_add("warning", f"end-{len(formatted_msg) + 1}c", "end-1c")
        elif level == "success":
            self.log_text.tag_add("success", f"end-{len(formatted_msg) + 1}c", "end-1c")

        # 스크롤을 가장 아래로 이동
        self.log_text.see(tk.END)

        # GUI 업데이트
        self.log_window.update_idletasks()

    def save_log(self):
        """로그 내용을 파일로 저장"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("로그 파일", "*.log"), ("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("저장 완료", f"로그가 {file_path}에 저장되었습니다.")
            except Exception as e:
                messagebox.showerror("저장 오류", f"로그 저장 중 오류 발생: {e}")

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
        ttk.Button(top_frame, text="로그 보기", command=self.show_log_window).grid(row=0, column=4, padx=5, pady=5)

        # 필터 프레임
        self.filter_frame = ttk.LabelFrame(self.root, text="필터", padding="10")
        self.filter_frame.pack(fill=tk.X, padx=10, pady=5)

        # 첫 번째 행: 모드 및 변형 선택
        ttk.Label(self.filter_frame, text="모드:").grid(row=0, column=0, padx=5, pady=5)
        self.selected_mode = tk.StringVar(value="모든 모드")
        self.mode_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_mode, state="readonly")
        self.mode_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.mode_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        ttk.Label(self.filter_frame, text="변형 타입:").grid(row=0, column=2, padx=5, pady=5)
        self.selected_variant = tk.StringVar(value="모든 변형")
        self.variant_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_variant, state="readonly")
        self.variant_combobox.grid(row=0, column=3, padx=5, pady=5)
        self.variant_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        # 두 번째 행: 모델 및 데이터셋 선택
        ttk.Label(self.filter_frame, text="모델:").grid(row=1, column=0, padx=5, pady=5)
        self.model_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_model, state="readonly")
        self.model_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.model_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        ttk.Label(self.filter_frame, text="데이터셋:").grid(row=1, column=2, padx=5, pady=5)
        self.dataset_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_dataset, state="readonly")
        self.dataset_combobox.grid(row=1, column=3, padx=5, pady=5)
        self.dataset_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        # 세 번째 행: 오류 유형 및 카테고리 선택
        ttk.Label(self.filter_frame, text="오류 유형:").grid(row=2, column=0, padx=5, pady=5)
        self.error_type_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_error_type,
                                                values=["모든 오류", "쿼리 생성 오류", "DB 실행 오류", "기타 오류"], state="readonly")
        self.error_type_combobox.grid(row=2, column=1, padx=5, pady=5)
        self.error_type_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        ttk.Label(self.filter_frame, text="쿼리 카테고리:").grid(row=2, column=2, padx=5, pady=5)
        self.category_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_category, width=30,
                                              state="readonly")
        self.category_combobox.grid(row=2, column=3, padx=5, pady=5)
        self.category_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        # 버튼은 세 번째 행 끝에 위치
        ttk.Button(self.filter_frame, text="필터 적용", command=self.apply_filters).grid(row=2, column=4, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="필터 초기화", command=self.reset_filters).grid(row=2, column=5, padx=5, pady=5)

        # 메인 컨텐츠 영역 - 노트북 위젯
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 탭 1: 실패 쿼리 리스트
        self.failed_queries_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.failed_queries_tab, text="실패 쿼리 목록")

        # 실패 쿼리 목록 표시
        queries_frame = ttk.Frame(self.failed_queries_tab)
        queries_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 트리뷰 열 정의 수정 (create_widgets 함수 내)
        self.queries_tree = ttk.Treeview(queries_frame,
                                         columns=["id", "mode", "model", "dataset", "variant", "category", "error_type",
                                                  "query_text"],
                                         show="headings", height=10)
        self.queries_tree.heading("id", text="ID")
        self.queries_tree.heading("mode", text="모드")  # 모드 컬럼 추가
        self.queries_tree.heading("model", text="모델")
        self.queries_tree.heading("dataset", text="데이터셋")
        self.queries_tree.heading("variant", text="변형")
        self.queries_tree.heading("category", text="카테고리")
        self.queries_tree.heading("error_type", text="오류 유형")
        self.queries_tree.heading("query_text", text="쿼리 텍스트")

        self.queries_tree.column("id", width=40, anchor="center")
        self.queries_tree.column("mode", width=70, anchor="center")  # 모드 컬럼 너비 설정
        self.queries_tree.column("model", width=80, anchor="center")
        self.queries_tree.column("dataset", width=80, anchor="center")
        self.queries_tree.column("variant", width=60, anchor="center")
        self.queries_tree.column("category", width=100, anchor="center")
        self.queries_tree.column("error_type", width=90, anchor="center")
        self.queries_tree.column("query_text", width=500)  # 너비 조정

        # 스크롤바 추가
        queries_scrollbar = ttk.Scrollbar(queries_frame, orient="vertical", command=self.queries_tree.yview)
        self.queries_tree.configure(yscrollcommand=queries_scrollbar.set)

        self.queries_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        queries_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 쿼리 선택 시 이벤트 처리
        self.queries_tree.bind("<<TreeviewSelect>>", self.on_query_selected)

        # 탭 2: 오류 분석
        self.error_analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.error_analysis_tab, text="오류 분석")

        # 탭 3: 쿼리 비교
        self.query_comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.query_comparison_tab, text="쿼리 비교")

        # 메타데이터 섹션 추가 (맨 위)
        metadata_frame = ttk.LabelFrame(self.query_comparison_tab, text="쿼리 메타데이터")
        metadata_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)

        # 메타데이터용 내부 탭 생성
        self.metadata_notebook = ttk.Notebook(metadata_frame)
        self.metadata_notebook.pack(fill=tk.X, expand=True, padx=5, pady=5)

        # 기본 메타데이터 탭
        self.basic_metadata_tab = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.basic_metadata_tab, text="기본 정보")

        # ▽ 기존 ScrolledText 삭제하고 Treeview + 스크롤바로 교체
        cols = ("항목", "값")
        self.basic_table = ttk.Treeview(
            self.basic_metadata_tab, columns=cols, show="headings", height=6
        )
        self.basic_table.heading("항목", text="항목")
        self.basic_table.heading("값", text="값")
        self.basic_table.column("항목", width=120, anchor="center")
        self.basic_table.column("값", width=400)

        scroll = ttk.Scrollbar(
            self.basic_metadata_tab, orient="vertical", command=self.basic_table.yview
        )
        self.basic_table.configure(yscrollcommand=scroll.set)

        self.basic_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 질문 탭
        self.question_tab = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.question_tab, text="질문")

        self.question_text = scrolledtext.ScrolledText(self.question_tab, wrap=tk.WORD, height=4)
        self.question_text.pack(fill=tk.BOTH, expand=True)

        # 지시사항 탭
        self.instructions_tab = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.instructions_tab, text="지시사항")

        self.instructions_text = scrolledtext.ScrolledText(self.instructions_tab, wrap=tk.WORD, height=4)
        self.instructions_text.pack(fill=tk.BOTH, expand=True)

        # 오류 메시지 탭
        self.error_msg_tab = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.error_msg_tab, text="오류 메시지")

        self.error_msg_text = scrolledtext.ScrolledText(self.error_msg_tab, wrap=tk.WORD, height=4)
        self.error_msg_text.pack(fill=tk.BOTH, expand=True)

        # 기타 메타데이터 탭
        self.other_metadata_tab = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.other_metadata_tab, text="기타 정보")

        self.other_metadata_text = scrolledtext.ScrolledText(self.other_metadata_tab, wrap=tk.WORD, height=4)
        self.other_metadata_text.pack(fill=tk.BOTH, expand=True)

        # 쿼리 비교 탭 구성
        comparison_frame = ttk.Frame(self.query_comparison_tab)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 왼쪽 패널 (생성된 SQL)
        left_panel = ttk.LabelFrame(comparison_frame, text="생성된 SQL")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.generated_sql_text = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, width=50, height=20)
        self.generated_sql_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 오른쪽 패널 (예상 SQL)
        right_panel = ttk.LabelFrame(comparison_frame, text="예상 SQL")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.expected_sql_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, width=50, height=20)
        self.expected_sql_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 하단 패널 (차이점)
        bottom_panel = ttk.LabelFrame(self.query_comparison_tab, text="차이점")
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.diff_text = scrolledtext.ScrolledText(bottom_panel, wrap=tk.WORD, height=10)
        self.diff_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 탭 4: 오류 패턴
        self.error_patterns_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.error_patterns_tab, text="오류 패턴")

        # 패턴 탭 구성
        patterns_frame = ttk.Frame(self.error_patterns_tab)
        patterns_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 왼쪽 패널 (패턴 목록) - width 옵션 제거
        left_patterns = ttk.LabelFrame(patterns_frame, text="오류 패턴 목록")
        left_patterns.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 패턴 트리뷰를 포함할 내부 프레임
        pattern_frame = ttk.Frame(left_patterns)
        pattern_frame.pack(fill=tk.BOTH, expand=True)

        self.patterns_tree = ttk.Treeview(pattern_frame, columns=["pattern", "count"], show="headings", height=20)
        self.patterns_tree.heading("pattern", text="오류 패턴")
        self.patterns_tree.heading("count", text="발생 횟수")

        self.patterns_tree.column("pattern", width=300)
        self.patterns_tree.column("count", width=80, anchor="center")

        patterns_scrollbar = ttk.Scrollbar(pattern_frame, orient="vertical", command=self.patterns_tree.yview)
        self.patterns_tree.configure(yscrollcommand=patterns_scrollbar.set)

        self.patterns_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        patterns_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 패턴 선택 시 이벤트 처리
        self.patterns_tree.bind("<<TreeviewSelect>>", self.on_pattern_selected)

        # 오른쪽 패널 (해당 패턴의 쿼리 예시)
        right_patterns = ttk.LabelFrame(patterns_frame, text="해당 패턴의 쿼리 예시")
        right_patterns.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pattern_examples_text = scrolledtext.ScrolledText(right_patterns, wrap=tk.WORD)
        self.pattern_examples_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 패턴 분석 버튼
        ttk.Button(left_patterns, text="패턴 분석 실행", command=self.analyze_error_patterns).pack(pady=10)

        # 상태 표시줄
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var.set("준비")

    def show_log_window(self):
        """로그 창 표시"""
        if not hasattr(self, 'log_window'):
            self.create_log_window()
        else:
            self.log_window.deiconify()  # 로그 창이 닫혀 있을 경우 표시

    def browse_directory(self):
        """디렉토리 선택 다이얼로그 및 파일 패턴 감지"""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)

            # 로그 창 생성 또는 초기화
            if not hasattr(self, 'log_window'):
                self.create_log_window()
            else:
                self.log_text.delete(1.0, tk.END)
                self.log_window.deiconify()  # 로그 창이 닫혀 있을 경우 표시

            self.log_message(f"디렉토리 선택: {directory}")

            # 디렉토리 내 모든 CSV 파일 목록 가져오기
            csv_files = list(Path(directory).glob("*.csv"))

            if not csv_files:
                self.log_message("CSV 파일을 찾을 수 없습니다.", level="error")
                messagebox.showerror("오류", "선택한 디렉토리에 CSV 파일이 없습니다.")
                return

            self.log_message(f"{len(csv_files)}개의 CSV 파일을 발견했습니다.", level="success")

            # 파일명 패턴 분석
            modes = set()
            models = set()
            variant_types = set()

            for csv_file in csv_files:
                filename = csv_file.stem  # 확장자 제외한 파일명
                self.log_message(f"파일 분석 중: {filename}")

                # 파일명 분석
                mode = "unknown"
                model = "unknown"
                variant = "unknown"

                # 파일 접두어에 따른 모드 식별
                if filename.startswith("hf-"):
                    mode = "huggingface"
                    rest = filename[3:]  # "hf-" 제거
                elif filename.startswith("o-"):
                    mode = "ollama"
                    rest = filename[2:]  # "o-" 제거
                elif filename.startswith("ollama") or filename.startswith("ollama2") or filename.startswith("ollama3"):
                    mode = "ollama"
                    if "-" in filename:
                        rest = filename[filename.find("-") + 1:]
                    else:
                        rest = filename
                else:
                    # 기타 케이스
                    rest = filename
                    if "_" in rest:
                        base_part = rest.split("_")[0]
                    else:
                        base_part = rest
                    self.log_message(f"  - 알 수 없는 모드: {base_part}", level="warning")

                # 모델명 추출 (underscore 앞부분)
                if "_" in rest:
                    model = rest.split("_")[0]
                    variant_part = rest.split("_")[1]

                    # 변형 유형 식별
                    if "advanced" in variant_part:
                        variant = "advanced"
                    elif "basic" in variant_part:
                        variant = "basic"
                    elif "classic" in variant_part:
                        variant = "classic"
                    else:
                        variant = variant_part
                else:
                    model = rest
                    variant = "default"

                self.log_message(f"  - 추출된 정보: 모드={mode}, 모델={model}, 변형={variant}")

                modes.add(mode)
                models.add(model)
                variant_types.add(variant)

            self.modes = sorted(list(modes))
            self.models = sorted(list(models))
            self.variant_types = sorted(list(variant_types))

            self.log_message(f"감지된 모드: {', '.join(self.modes)}", level="success")
            self.log_message(f"감지된 모델: {', '.join(self.models)}", level="success")
            self.log_message(f"감지된 변형: {', '.join(self.variant_types)}", level="success")

            self.status_var.set(f"감지: {len(self.modes)}개 모드, {len(self.models)}개 모델")

            # 콤보박스 업데이트
            self.update_comboboxes()

            # 추가 필터 콤보박스 (모드, 변형 타입) 추가
            if not hasattr(self, 'mode_combobox'):
                self.setup_additional_comboboxes()
            else:
                # 모드 콤보박스 업데이트
                mode_values = ["모든 모드"] + self.modes
                self.mode_combobox['values'] = mode_values
                self.selected_mode.set("모든 모드")

                # 변형 콤보박스 업데이트
                variant_values = ["모든 변형"] + self.variant_types
                self.variant_combobox['values'] = variant_values
                self.selected_variant.set("모든 변형")

    def setup_additional_comboboxes(self):
        """모드 및 변형 타입 선택 콤보박스 추가"""
        # 필터 프레임 재배치 - 모든 기존 위젯 그리드 위치 조정 필요

        # 모드 필터 (새로운 첫 번째 행)
        ttk.Label(self.filter_frame, text="모드:").grid(row=0, column=0, padx=5, pady=5)
        self.selected_mode = tk.StringVar(value="모든 모드")
        self.mode_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_mode, state="readonly")
        self.mode_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.mode_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        # 변형 필터 (새로운 첫 번째 행)
        ttk.Label(self.filter_frame, text="변형 타입:").grid(row=0, column=2, padx=5, pady=5)
        self.selected_variant = tk.StringVar(value="모든 변형")
        self.variant_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_variant, state="readonly")
        self.variant_combobox.grid(row=0, column=3, padx=5, pady=5)
        self.variant_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        # 모델과 데이터셋 필터는 두 번째 행으로 이동
        # create_widgets에서 필터 프레임 위젯 배치 순서 변경 필요

    def setup_variant_combobox(self):
        """변형 타입 선택 콤보박스 추가"""
        # filter_frame에 변형 콤보박스 추가
        ttk.Label(self.filter_frame, text="변형 타입:").grid(row=0, column=4, padx=5, pady=5)
        self.selected_variant = tk.StringVar(value="모든 변형")
        self.variant_combobox = ttk.Combobox(self.filter_frame, textvariable=self.selected_variant, state="readonly")
        self.variant_combobox.grid(row=0, column=5, padx=5, pady=5)
        self.variant_combobox.bind('<<ComboboxSelected>>', self.filter_changed)

        # 초기 값 설정
        variant_values = ["모든 변형"] + self.variant_types
        self.variant_combobox['values'] = variant_values

    def update_comboboxes(self):
        """콤보박스 옵션 업데이트"""
        # 모델 콤보박스 업데이트
        model_values = ["모든 모델"] + self.models
        self.model_combobox['values'] = model_values
        self.selected_model.set("모든 모델")

        # 데이터셋 콤보박스 업데이트
        dataset_values = ["모든 데이터셋"] + self.dataset_types
        self.dataset_combobox['values'] = dataset_values
        self.selected_dataset.set("모든 데이터셋")

        # 카테고리 리스트는 데이터 로드 후 업데이트됨

    def load_data(self):
        """선택한 디렉토리에서 CSV 파일 로드"""
        directory = self.dir_entry.get()
        base_path = Path(directory)

        # 로그 창 생성 또는 초기화
        if not hasattr(self, 'log_window'):
            self.create_log_window()
        else:
            self.log_text.delete(1.0, tk.END)
            self.log_window.deiconify()  # 로그 창이 닫혀 있을 경우 표시

        self.log_message("데이터 로드 시작...")
        self.status_var.set("데이터 로드 중...")
        self.root.update_idletasks()

        # 모드나 모델 목록이 없을 경우 확인
        if not hasattr(self, 'modes') or not hasattr(self, 'models') or not self.modes or not self.models:
            self.log_message("모드 또는 모델 정보가 없습니다. 디렉토리를 먼저 선택해주세요.", level="error")
            self.status_var.set("모드 또는 모델 정보 없음")
            messagebox.showerror("오류", "모드 또는 모델 정보가 없습니다.\n'찾아보기' 버튼을 통해 디렉토리를 먼저 선택해주세요.")
            return

        # 디렉토리 내 모든 CSV 파일 목록 가져오기
        all_csv_files = list(base_path.glob("*.csv"))
        self.log_message(f"디렉토리 내 총 CSV 파일: {len(all_csv_files)}개")

        if not all_csv_files:
            self.log_message("CSV 파일을 찾을 수 없습니다.", level="error")
            self.status_var.set("CSV 파일 없음")
            return

        # 처리할 파일 분류
        mode_filter = getattr(self, 'selected_mode', tk.StringVar(value="모든 모드")).get()
        model_filter = getattr(self, 'selected_model', tk.StringVar(value="모든 모델")).get()
        variant_filter = getattr(self, 'selected_variant', tk.StringVar(value="모든 변형")).get()

        # 파일 처리 정보 초기화
        loaded_files = 0
        failed_files = 0
        self.data = {}  # 데이터 초기화
        all_categories = set(["모든 카테고리"])  # 카테고리 수집

        # 각 파일 처리
        for file_path in all_csv_files:
            filename = file_path.stem

            # 파일 정보 파싱
            file_mode = "unknown"
            file_model = "unknown"
            file_variant = "unknown"

            # 모드 확인
            if filename.startswith("hf-"):
                file_mode = "huggingface"
                rest = filename[3:]
            elif filename.startswith("o-"):
                file_mode = "ollama"
                rest = filename[2:]
            elif filename.startswith("ollama"):
                file_mode = "ollama"
                if "-" in filename:
                    rest = filename[filename.find("-") + 1:]
                else:
                    rest = filename
            else:
                rest = filename

            # 모델명 및 변형 추출
            if "_" in rest:
                file_model = rest.split("_")[0]
                variant_part = rest.split("_")[1]

                if "advanced" in variant_part:
                    file_variant = "advanced"
                elif "basic" in variant_part:
                    file_variant = "basic"
                elif "classic" in variant_part:
                    file_variant = "classic"
                else:
                    file_variant = variant_part
            else:
                file_model = rest
                file_variant = "default"

            # 필터 적용 (파일 로드 시에는 모든 파일 로드)
            mode_match = mode_filter == "모든 모드" or file_mode == mode_filter
            model_match = model_filter == "모든 모델" or file_model == model_filter
            variant_match = variant_filter == "모든 변형" or file_variant == variant_filter

            self.log_message(f"파일 분석: {filename}")
            self.log_message(f"  - 모드: {file_mode}, 모델: {file_model}, 변형: {file_variant}")

            if mode_match and model_match and variant_match:
                self.log_message(f"파일 로드 중: {file_path.name}")
                try:
                    # CSV 파일 읽기
                    df = pd.read_csv(file_path)
                    self.log_message(f"  - CSV 읽기 완료: {len(df)}행, {len(df.columns)}열")

                    # 데이터 저장 구조 확인
                    if file_mode not in self.data:
                        self.data[file_mode] = {}

                    if file_model not in self.data[file_mode]:
                        self.data[file_mode][file_model] = {}

                    # 모델 + 변형 조합으로 저장
                    self.data[file_mode][file_model][file_variant] = df

                    loaded_files += 1

                    # 필수 열 확인
                    required_cols = ['correct', 'error_query_gen', 'error_db_exec']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        self.log_message(f"  - 경고: 필수 열 누락 - {', '.join(missing_cols)}", level="warning")
                        # 누락된 열 자동 추가 (기본값 0)
                        for col in missing_cols:
                            df[col] = 0
                        self.log_message(f"  - 누락된 열을 기본값(0)으로 자동 추가했습니다.", level="warning")
                    else:
                        self.log_message("  - 모든 필수 열 있음", level="success")

                    # 전체 열 목록 표시
                    self.log_message(f"  - 열 목록: {', '.join(df.columns)}")

                    # 실패 쿼리 수 로깅
                    if 'correct' in df.columns:
                        failed_count = df[df['correct'] == 0].shape[0]
                        self.log_message(f"  - 실패한 쿼리 수: {failed_count}")

                    # 카테고리 수집
                    if 'query_category' in df.columns:
                        categories = df['query_category'].dropna().unique()
                        self.log_message(f"  - 카테고리 수: {len(categories)}")
                        all_categories.update(categories)

                    self.status_var.set(f"{file_path.name} 로드 완료 ({loaded_files}/{len(all_csv_files)})")
                    self.log_message(f"로드 성공: {file_path.name}", level="success")
                except Exception as e:
                    failed_files += 1
                    error_msg = f"{file_path} 로드 중 오류 발생: {e}"
                    self.log_message(f"로드 실패: {error_msg}", level="error")
                    messagebox.showerror("오류", error_msg)
            else:
                self.log_message(f"  - 필터에 의해 스킵됨 (필터 조건과 일치하지 않음)")

        # 로드 결과 정리
        if loaded_files > 0:
            # 카테고리 콤보박스 업데이트
            self.category_combobox['values'] = sorted(list(all_categories))
            self.selected_category.set("모든 카테고리")

            # 실패 쿼리 분석
            self.log_message("실패 쿼리 분석 중...")
            self.identify_failed_queries()

            # 오류 분석 탭 업데이트
            self.log_message("오류 분석 탭 업데이트 중...")
            self.update_error_analysis_tab()

            success_msg = f"{loaded_files}개 파일 로드 완료, {failed_files}개 파일 로드 실패"
            self.status_var.set(success_msg)
            self.log_message(success_msg, level="success")
        else:
            error_msg = "로드된 파일이 없습니다. 필터 조건을 확인하세요."
            self.status_var.set(error_msg)
            self.log_message(error_msg, level="error")

    def identify_failed_queries(self):
        """실패한 쿼리 식별 및 분석"""
        self.failed_queries = {}
        total_failed = 0

        self.log_message("실패한 쿼리 식별 중...")

        # 새로운 계층적 데이터 구조 처리
        for mode in self.data.keys():
            self.failed_queries[mode] = {}

            for model in self.data[mode].keys():
                self.failed_queries[mode][model] = {}

                for variant in self.data[mode][model].keys():
                    df = self.data[mode][model][variant]

                    # 데이터셋 정보 로그
                    self.log_message(f"분석 중: {mode} - {model} - {variant} ({len(df)}행)")

                    # 필수 열이 있는지 확인
                    required_cols = ['correct', 'error_query_gen', 'error_db_exec']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        self.log_message(f"경고: {mode}/{model}/{variant}에서 필수 열이 누락되었습니다: {missing_cols}",
                                         level="warning")
                        # 누락된 열 자동 추가 (기본값 0)
                        for col in missing_cols:
                            df[col] = 0
                        self.log_message(f"누락된 열을 기본값(0)으로 자동 추가했습니다.", level="warning")

                    # 실패한 쿼리 필터링 (correct=0)
                    # 'correct' 열이 없는 경우 오류 처리
                    if 'correct' not in df.columns:
                        self.log_message(f"경고: {mode}/{model}/{variant}에 'correct' 열이 없습니다.", level="warning")
                        continue

                    failed_df = df[df['correct'] == 0].copy()

                    if len(failed_df) > 0:
                        self.log_message(f"  - 실패한 쿼리 발견: {len(failed_df)}개", level="success")

                        # 오류 유형 추가
                        # 기본값은 '기타 오류'로 설정
                        failed_df['error_type'] = '기타 오류'

                        # 우선순위에 따라 오류 유형 설정
                        # 쿼리 생성 오류가 있으면 '쿼리 생성 오류'로 설정
                        if 'error_query_gen' in failed_df.columns:
                            failed_df.loc[failed_df['error_query_gen'] == 1, 'error_type'] = '쿼리 생성 오류'
                            gen_errors = failed_df[failed_df['error_query_gen'] == 1].shape[0]
                            self.log_message(f"  - 쿼리 생성 오류: {gen_errors}개")

                        # DB 실행 오류가 있으면 'DB 실행 오류'로 설정
                        if 'error_db_exec' in failed_df.columns:
                            failed_df.loc[failed_df['error_db_exec'] == 1, 'error_type'] = 'DB 실행 오류'
                            db_errors = failed_df[failed_df['error_db_exec'] == 1].shape[0]
                            self.log_message(f"  - DB 실행 오류: {db_errors}개")

                        # 기타 오류 집계
                        other_errors = failed_df[failed_df['error_type'] == '기타 오류'].shape[0]
                        self.log_message(f"  - 기타 오류: {other_errors}개")

                        # 데이터셋 식별 (파일 이름에서 데이터셋 부분 추출)
                        if "sqlcoder" in model:
                            dataset = "sqlcoder"
                        elif "qwq" in model:
                            dataset = "qwq"
                        else:
                            dataset = "unknown"

                        # 데이터셋에 추가 (데이터셋+변형 조합으로 키 생성)
                        key = f"{dataset}_{variant}"
                        self.failed_queries[mode][model][key] = failed_df
                        total_failed += len(failed_df)
                    else:
                        self.log_message(f"  - 실패한 쿼리 없음")

        self.log_message(f"총 {total_failed}개의 실패한 쿼리를 찾았습니다", level="success" if total_failed > 0 else "warning")
        self.status_var.set(f"총 {total_failed}개의 실패한 쿼리를 찾았습니다")

        # 기본 필터 적용
        self.apply_filters()

    def apply_filters(self):
        """현재 선택된 필터를 적용하여 실패 쿼리 목록 갱신"""
        # 트리뷰 초기화
        for item in self.queries_tree.get_children():
            self.queries_tree.delete(item)

        mode_filter = self.selected_mode.get()
        model_filter = self.selected_model.get()
        dataset_filter = self.selected_dataset.get()
        error_type_filter = self.selected_error_type.get()
        category_filter = self.selected_category.get()
        variant_filter = self.selected_variant.get()

        self.log_message("필터 적용 중...")
        self.log_message(f"  - 모드: {mode_filter}")
        self.log_message(f"  - 모델: {model_filter}")
        self.log_message(f"  - 데이터셋: {dataset_filter}")
        self.log_message(f"  - 오류 유형: {error_type_filter}")
        self.log_message(f"  - 카테고리: {category_filter}")
        self.log_message(f"  - 변형: {variant_filter}")

        # 모델 선택
        models_to_show = self.models if model_filter == "모든 모델" else [model_filter]

        # 적용할 파일 목록 수집
        filtered_files = []

        # 디렉토리 내 모든 CSV 파일 목록 가져오기
        directory = self.dir_entry.get()
        base_path = Path(directory)
        all_csv_files = list(base_path.glob("*.csv"))

        for file_path in all_csv_files:
            filename = file_path.stem

            # 파일 정보 파싱
            file_mode = "unknown"
            file_model = "unknown"
            file_variant = "unknown"

            # 모드 확인
            if filename.startswith("hf-"):
                file_mode = "huggingface"
                rest = filename[3:]
            elif filename.startswith("o-"):
                file_mode = "ollama"
                rest = filename[2:]
            elif filename.startswith("ollama"):
                file_mode = "ollama"
                if "-" in filename:
                    rest = filename[filename.find("-") + 1:]
                else:
                    rest = filename
            else:
                rest = filename

            # 모델명 및 변형 추출
            if "_" in rest:
                file_model = rest.split("_")[0]
                variant_part = rest.split("_")[1]

                if "advanced" in variant_part:
                    file_variant = "advanced"
                elif "basic" in variant_part:
                    file_variant = "basic"
                elif "classic" in variant_part:
                    file_variant = "classic"
                else:
                    file_variant = variant_part
            else:
                file_model = rest
                file_variant = "default"

            # 필터 적용
            mode_match = mode_filter == "모든 모드" or file_mode == mode_filter
            model_match = model_filter == "모든 모델" or file_model == model_filter
            variant_match = variant_filter == "모든 변형" or file_variant == variant_filter

            if mode_match and model_match and variant_match:
                filtered_files.append((file_path, file_mode, file_model, file_variant))

        self.log_message(f"필터링된 파일 수: {len(filtered_files)}")

        # 필터링된 파일에서 쿼리 추출
        row_id = 0
        filtered_queries = []

        for file_path, file_mode, file_model, file_variant in filtered_files:
            try:
                # CSV 파일 읽기
                df = pd.read_csv(file_path)

                # 필요한 열이 없으면 추가
                for col in ['correct', 'error_query_gen', 'error_db_exec']:
                    if col not in df.columns:
                        df[col] = 0

                # 실패한 쿼리 필터링
                failed_df = df[df['correct'] == 0].copy()

                # 오류 유형 추가
                failed_df['error_type'] = '기타 오류'
                failed_df.loc[failed_df['error_query_gen'] == 1, 'error_type'] = '쿼리 생성 오류'
                failed_df.loc[failed_df['error_db_exec'] == 1, 'error_type'] = 'DB 실행 오류'

                # 오류 유형 필터
                if error_type_filter != "모든 오류":
                    failed_df = failed_df[failed_df['error_type'] == error_type_filter]

                # 카테고리 필터
                if category_filter != "모든 카테고리" and 'query_category' in failed_df.columns:
                    failed_df = failed_df[failed_df['query_category'] == category_filter]

                # 트리뷰에 추가
                for _, row in failed_df.iterrows():
                    # 쿼리 텍스트 준비
                    query_text = str(row.get('generated_query', '')) if pd.notna(
                        row.get('generated_query')) else "쿼리 생성 실패"

                    # 너무 긴 경우 잘라내기
                    if len(query_text) > 100:
                        query_text = query_text[:100] + "..."

                    category = str(row.get('query_category', '')) if pd.notna(row.get('query_category')) else "분류 없음"

                    # 트리뷰에 추가
                    self.queries_tree.insert("", "end", values=(
                        row_id,
                        file_mode.upper(),  # 모드 정보 추가
                        file_model.upper(),
                        dataset_filter if dataset_filter != "모든 데이터셋" else "기본",
                        file_variant,
                        category,
                        row['error_type'],
                        query_text
                    ))

                    # 상세 정보 저장
                    filtered_queries.append({
                        'id': row_id,
                        'model': file_model,
                        'mode': file_mode,
                        'dataset': dataset_filter if dataset_filter != "모든 데이터셋" else "기본",
                        'variant': file_variant,
                        'row_data': row.to_dict(),
                        'file_path': str(file_path)
                    })

                    row_id += 1

            except Exception as e:
                self.log_message(f"파일 처리 중 오류 발생: {file_path.name} - {str(e)}", level="error")

        # 필터링된 쿼리 저장
        self.filtered_queries = filtered_queries

        if filtered_queries:
            self.status_var.set(f"필터 적용 완료: {len(filtered_queries)}개의 쿼리가 표시됨")
            self.log_message(f"필터 적용 완료: {len(filtered_queries)}개의 쿼리가 표시됨", level="success")
        else:
            self.status_var.set("필터 조건에 맞는 실패 쿼리가 없습니다")
            self.log_message("필터 조건에 맞는 실패 쿼리가 없습니다", level="warning")

    def reset_filters(self):
        """필터 설정 초기화"""
        self.selected_mode.set("모든 모드")
        self.selected_model.set("모든 모델")
        self.selected_dataset.set("모든 데이터셋")
        self.selected_error_type.set("모든 오류")
        self.selected_category.set("모든 카테고리")
        self.selected_variant.set("모든 변형")
        self.apply_filters()

    def filter_changed(self, event=None):
        """필터 설정이 변경될 때 호출되는 함수"""
        # 자동으로 필터 적용 (선택적으로 사용)
        pass

    def on_query_selected(self, event=None):
        """쿼리가 선택되었을 때 호출되는 함수"""
        selected_items = self.queries_tree.selection()
        if not selected_items:
            return

        # 첫 번째 선택된 항목의 ID 가져오기
        selected_id = self.queries_tree.item(selected_items[0], "values")[0]

        # ID로 쿼리 데이터 찾기
        query_data = None
        for query in self.filtered_queries:
            if str(query['id']) == str(selected_id):
                query_data = query
                break

        if query_data:
            # 선택된 쿼리 정보로 UI 업데이트
            self.show_query_details(query_data)

    def show_query_details(self, query_data):
        """선택된 쿼리의 상세 정보 표시"""
        # 쿼리 비교 탭으로 전환
        self.notebook.select(self.query_comparison_tab)

        row_data = query_data['row_data']
        model = query_data['model']
        mode = query_data.get('mode', 'unknown')
        dataset = query_data.get('dataset', 'unknown')
        variant = query_data.get('variant', 'unknown')
        file_path = query_data.get('file_path', '')

        # 소스 파일 정보 구성
        if file_path:
            source_file = Path(file_path).name
        else:
            source_file = f"{mode}-{model}_{variant}.csv"

        # 쿼리 정보 가져오기
        generated_query = row_data.get('generated_query', '')
        expected_query = row_data.get('query', '')

        # 모든 메타데이터 텍스트 위젯 초기화
        # Treeview는 행을 모두 지우는 식으로 초기화
        for row in self.basic_table.get_children():
            self.basic_table.delete(row)
        self.question_text.delete(1.0, tk.END)
        self.instructions_text.delete(1.0, tk.END)
        self.error_msg_text.delete(1.0, tk.END)
        self.other_metadata_text.delete(1.0, tk.END)

        # 메타데이터 구성
        kv_pairs = [
            ("소스 파일", source_file),
            ("모드", mode.upper()),
            ("모델", model.upper()),
            ("데이터셋", dataset.upper()),
            ("변형", variant),
            ("카테고리", row_data.get("query_category", "N/A")
            if pd.notna(row_data.get("query_category")) else "N/A"),
            ("오류 유형", row_data.get("error_type", "N/A")),
        ]

        # 짧은 수치 데이터 추가
        for key in ["latency_seconds", "execution_time", "db_name", "db_type", "exact_match", "timeout", "tokens_used",
                    "cot_pregen"]:
            if key in row_data and pd.notna(row_data[key]):
                kv_pairs.append((key, row_data[key]))

        # Treeview에 삽입
        for k, v in kv_pairs:
            self.basic_table.insert("", tk.END, values=(k, v))

        # 질문 탭
        if 'question' in row_data and pd.notna(row_data['question']):
            self.question_text.insert(tk.END, row_data['question'])
            self.metadata_notebook.tab(1, state="normal")  # 질문 탭 활성화
        else:
            self.question_text.insert(tk.END, "질문 정보가 없습니다.")
            self.metadata_notebook.tab(1, state="disabled")  # 질문 탭 비활성화

        # 지시사항 탭
        if 'instructions' in row_data and pd.notna(row_data['instructions']):
            self.instructions_text.insert(tk.END, row_data['instructions'])
            self.metadata_notebook.tab(2, state="normal")  # 지시사항 탭 활성화
        else:
            self.instructions_text.insert(tk.END, "지시사항 정보가 없습니다.")
            self.metadata_notebook.tab(2, state="disabled")  # 지시사항 탭 비활성화

        # 오류 메시지 탭
        error_msg = ""
        # 여러 가능한 오류 메시지 필드 확인
        for error_field in ['error_msg', 'error_message', 'error_db_exec_message']:
            if error_field in row_data and pd.notna(row_data[error_field]):
                error_msg += f"{error_field}: {row_data[error_field]}\n\n"

        if error_msg:
            self.error_msg_text.insert(tk.END, error_msg)
            self.metadata_notebook.tab(3, state="normal")  # 오류 메시지 탭 활성화
        else:
            self.error_msg_text.insert(tk.END, "오류 메시지가 없습니다.")
            self.metadata_notebook.tab(3, state="disabled")  # 오류 메시지 탭 비활성화

        # 기타 메타데이터 수집
        other_metadata = ""
        excluded_keys = ['generated_query', 'query', 'query_category', 'error_type',
                         'error_msg', 'error_message', 'error_db_exec_message',
                         'execution_time', 'latency_seconds', 'correct',
                         'error_query_gen', 'error_db_exec', 'question', 'instructions',
                         'db_name', 'db_type', "exact_match", "timeout", "tokens_used", "cot_pregen"]

        for key, value in row_data.items():
            if key not in excluded_keys and pd.notna(value):
                other_metadata += f"{key}: {value}\n\n"

        if other_metadata:
            self.other_metadata_text.insert(tk.END, other_metadata)
            self.metadata_notebook.tab(4, state="normal")  # 기타 정보 탭 활성화
        else:
            self.other_metadata_text.insert(tk.END, "추가 메타데이터가 없습니다.")
            self.metadata_notebook.tab(4, state="disabled")  # 기타 정보 탭 비활성화

        # 기본 탭으로 초기화
        self.metadata_notebook.select(0)

        # 텍스트 위젯 초기화
        self.generated_sql_text.delete(1.0, tk.END)
        self.expected_sql_text.delete(1.0, tk.END)
        self.diff_text.delete(1.0, tk.END)

        # None 또는 NaN 처리
        has_expected_query = not pd.isna(expected_query) and expected_query != ''

        if pd.isna(generated_query):
            generated_query = "쿼리 생성 실패"

        # SQL 서식화
        try:
            generated_query_formatted = self.format_sql(str(generated_query))

            # 생성된 SQL 표시
            self.generated_sql_text.insert(tk.END, generated_query_formatted)

            # 예상 SQL이 있는 경우에만 처리
            if has_expected_query:
                expected_query_formatted = self.format_sql(str(expected_query))
                self.expected_sql_text.insert(tk.END, expected_query_formatted)

                # 차이점 계산 및 표시 (예상 SQL이 있는 경우에만)
                diff = self.compute_diff(str(generated_query), str(expected_query))
                self.diff_text.insert(tk.END, diff)
            else:
                # 예상 SQL이 없는 경우 메시지 표시
                self.expected_sql_text.insert(tk.END, "예상 쿼리 정보 없음")
                self.diff_text.insert(tk.END, "예상 쿼리가 없어 차이점을 표시할 수 없습니다.")
        except Exception as e:
            # 오류 발생시 원본 내용 표시
            self.generated_sql_text.insert(tk.END, str(generated_query))
            if has_expected_query:
                self.expected_sql_text.insert(tk.END, str(expected_query))
            else:
                self.expected_sql_text.insert(tk.END, "예상 쿼리 정보 없음")
            self.diff_text.insert(tk.END, f"서식화 중 오류 발생: {str(e)}")

        # 상태 업데이트
        self.status_var.set(f"{mode.upper()} 모드의 {model.upper()} 모델 ({variant} 변형) 쿼리 분석 중")
        self.log_message(f"쿼리 상세 정보 표시: {source_file}")

    def format_sql(self, sql):
        """SQL 쿼리 서식화 (간단한 구현)"""
        # 기본 키워드 리스트
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
                    'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
                    'LIMIT', 'OFFSET', 'UNION', 'INSERT', 'UPDATE', 'DELETE']

        # 모든 키워드를 대문자로 변환
        formatted = sql
        for keyword in keywords:
            pattern = re.compile(r'\b' + keyword.replace(' ', r'\s+') + r'\b', re.IGNORECASE)
            formatted = pattern.sub(keyword, formatted)

        # 줄바꿈 추가
        for keyword in keywords:
            if keyword in formatted:
                formatted = formatted.replace(keyword, f"\n{keyword}")

        return formatted.strip()

    def compute_diff(self, str1, str2):
        """두 문자열 간의 차이점 계산"""
        diff = difflib.ndiff(str1.splitlines(keepends=True),
                             str2.splitlines(keepends=True))
        return ''.join(diff)

    def analyze_error_patterns(self):
        """실패 쿼리에서 오류 패턴 분석"""
        self.status_var.set("오류 패턴 분석 중...")
        self.root.update_idletasks()

        # 오류 메시지 및 패턴 수집
        error_patterns = {}
        examples = {}

        for model in self.models:
            for dataset_type in self.dataset_types:
                if dataset_type not in self.failed_queries.get(model, {}):
                    continue

                df = self.failed_queries[model][dataset_type]

                # DB 실행 오류 메시지 분석
                if 'error_message' in df.columns:
                    for i, row in df.iterrows():
                        if pd.notna(row.get('error_message')):
                            msg = str(row['error_message'])

                            # 오류 메시지 정규화 (소문자 변환 및 숫자/변수명 일반화)
                            pattern = self.normalize_error_message(msg)

                            if pattern not in error_patterns:
                                error_patterns[pattern] = 1
                                examples[pattern] = [(model, dataset_type, msg, row.get('generated_query', ''))]
                            else:
                                error_patterns[pattern] += 1
                                if len(examples[pattern]) < 5:  # 최대 5개 예시만 저장
                                    examples[pattern].append((model, dataset_type, msg, row.get('generated_query', '')))

        # 패턴별로 정렬
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

        # 트리뷰 업데이트
        for item in self.patterns_tree.get_children():
            self.patterns_tree.delete(item)

        for pattern, count in sorted_patterns:
            self.patterns_tree.insert("", "end", values=(pattern, count))

        if sorted_patterns:
            self.status_var.set(f"{len(sorted_patterns)}개의 오류 패턴이 분석되었습니다")

            # 패턴 예시 정보 저장
            self.error_examples = examples
        else:
            self.status_var.set("오류 패턴을 찾을 수 없습니다")
            self.error_examples = {}

    def on_pattern_selected(self, event=None):
        """오류 패턴이 선택되었을 때 호출되는 함수"""
        selected_items = self.patterns_tree.selection()
        if not selected_items:
            return

        # 선택된 패턴 가져오기
        selected_pattern = self.patterns_tree.item(selected_items[0], "values")[0]

        # 패턴에 대한 예시 표시
        self.show_pattern_examples(selected_pattern)

    def show_pattern_examples(self, pattern):
        """선택된 오류 패턴의 예시 표시"""
        self.pattern_examples_text.delete(1.0, tk.END)

        if pattern not in self.error_examples:
            self.pattern_examples_text.insert(tk.END, "이 패턴에 대한 예시를 찾을 수 없습니다.")
            return

        examples = self.error_examples[pattern]

        self.pattern_examples_text.insert(tk.END, f"패턴: {pattern}\n\n")

        for i, (model, dataset, error_msg, query) in enumerate(examples, 1):
            self.pattern_examples_text.insert(tk.END, f"예시 {i} ({model.upper()} - {dataset}):\n")
            self.pattern_examples_text.insert(tk.END, f"오류: {error_msg}\n")
            self.pattern_examples_text.insert(tk.END, f"쿼리: {query}\n")
            self.pattern_examples_text.insert(tk.END, "-" * 50 + "\n\n")

    def normalize_error_message(self, message):
        """오류 메시지를 정규화하여 패턴 추출"""
        # 1. 소문자로 변환
        normalized = message.lower()

        # 2. 테이블명, 필드명 등의 구체적인 이름 일반화
        normalized = re.sub(r'\'[^\']+\'', "'IDENTIFIER'", normalized)

        # 3. 숫자 일반화
        normalized = re.sub(r'\d+', "NUMBER", normalized)

        # 4. 구체적인 SQL 구문 일반화
        normalized = re.sub(r'select[^;]+from', "SELECT ... FROM", normalized)

        # 5. 중복 공백 제거
        normalized = re.sub(r'\s+', " ", normalized).strip()

        return normalized

    def update_error_analysis_tab(self):
        """오류 분석 탭 업데이트"""
        # 기존 위젯 제거
        for widget in self.error_analysis_tab.winfo_children():
            widget.destroy()

        # 오류 유형 분포 분석
        error_types = {'쿼리 생성 오류': 0, 'DB 실행 오류': 0, '기타 오류': 0}
        model_errors = {}
        category_errors = {}

        for model in self.models:
            model_errors[model] = {'쿼리 생성 오류': 0, 'DB 실행 오류': 0, '기타 오류': 0}

            for dataset_type in self.dataset_types:
                if dataset_type not in self.failed_queries.get(model, {}):
                    continue

                df = self.failed_queries[model][dataset_type]

                # 오류 유형 집계
                error_types['쿼리 생성 오류'] += df[df['error_type'] == '쿼리 생성 오류'].shape[0]
                error_types['DB 실행 오류'] += df[df['error_type'] == 'DB 실행 오류'].shape[0]
                error_types['기타 오류'] += df[(df['error_type'] != '쿼리 생성 오류') &
                                           (df['error_type'] != 'DB 실행 오류')].shape[0]

                # 모델별 오류 집계
                model_errors[model]['쿼리 생성 오류'] += df[df['error_type'] == '쿼리 생성 오류'].shape[0]
                model_errors[model]['DB 실행 오류'] += df[df['error_type'] == 'DB 실행 오류'].shape[0]
                model_errors[model]['기타 오류'] += df[(df['error_type'] != '쿼리 생성 오류') &
                                                   (df['error_type'] != 'DB 실행 오류')].shape[0]

                # 카테고리별 오류 집계
                if 'query_category' in df.columns:
                    for category, group in df.groupby('query_category'):
                        if pd.isna(category):
                            continue

                        if category not in category_errors:
                            category_errors[category] = 0
                        category_errors[category] += len(group)

        # 시각화 영역 생성
        visualization_frame = ttk.Frame(self.error_analysis_tab)
        visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 왼쪽 패널 - 오류 유형 파이 차트
        left_panel = ttk.LabelFrame(visualization_frame, text="오류 유형 분포")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig1 = Figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)

        # 데이터가 있는 경우에만 파이 차트 생성
        if sum(error_types.values()) > 0:
            wedges, texts, autotexts = ax1.pie(
                error_types.values(),
                labels=error_types.keys(),
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff', '#99ff99']
            )
            ax1.set_title('오류 유형 분포')
        else:
            ax1.text(0.5, 0.5, '데이터 없음', ha='center', va='center')

        # 파이 차트를 Tkinter 캔버스에 추가
        canvas1 = FigureCanvasTkAgg(fig1, master=left_panel)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 오른쪽 패널 - 모델별 오류 막대 그래프
        right_panel = ttk.LabelFrame(visualization_frame, text="모델별 오류 건수")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig2 = Figure(figsize=(8, 4))
        ax2 = fig2.add_subplot(111)

        # 모델별 오류 데이터 준비
        models = []
        query_gen_errors = []
        db_exec_errors = []
        other_errors = []

        for model, errors in model_errors.items():
            if sum(errors.values()) > 0:  # 오류가 있는 모델만 포함
                models.append(model.upper())
                query_gen_errors.append(errors['쿼리 생성 오류'])
                db_exec_errors.append(errors['DB 실행 오류'])
                other_errors.append(errors['기타 오류'])

        if models:
            x = np.arange(len(models))
            width = 0.25

            ax2.bar(x - width, query_gen_errors, width, label='쿼리 생성 오류')
            ax2.bar(x, db_exec_errors, width, label='DB 실행 오류')
            ax2.bar(x + width, other_errors, width, label='기타 오류')

            ax2.set_ylabel('오류 건수')
            ax2.set_title('모델별 오류 유형 분포')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '데이터 없음', ha='center', va='center')

        fig2.tight_layout()

        # 막대 그래프를 Tkinter 캔버스에 추가
        canvas2 = FigureCanvasTkAgg(fig2, master=right_panel)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 하단 패널 - 카테고리별 오류 막대 그래프
        bottom_panel = ttk.LabelFrame(self.error_analysis_tab, text="쿼리 카테고리별 오류 건수")
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        fig3 = Figure(figsize=(10, 4))
        ax3 = fig3.add_subplot(111)

        # 카테고리 오류 데이터 준비 (상위 10개)
        categories = []
        error_counts = []

        for category, count in sorted(category_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            categories.append(category)
            error_counts.append(count)

        if categories:
            y_pos = np.arange(len(categories))

            # 가로 막대 그래프
            ax3.barh(y_pos, error_counts, align='center')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(categories)
            ax3.invert_yaxis()  # 가장 많은 것이 상단에 오도록
            ax3.set_xlabel('오류 건수')
            ax3.set_title('카테고리별 오류 건수 (상위 10개)')

            # 막대 끝에 값 표시
            for i, v in enumerate(error_counts):
                ax3.text(v + 0.5, i, str(v), color='black', va='center')
        else:
            ax3.text(0.5, 0.5, '카테고리 데이터 없음', ha='center', va='center')

        fig3.tight_layout()

        # 막대 그래프를 Tkinter 캔버스에 추가
        canvas3 = FigureCanvasTkAgg(fig3, master=bottom_panel)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    root.title("SQL 생성 모델 실패 쿼리 분석기")
    app = SQLFailedQueriesExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()