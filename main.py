from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Matrix = List[List[float]]
EPSILON = 1e-9
BENCHMARK_REPEATS = 10
JSON_SIZES = (5, 13, 25)
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")


@dataclass
class JsonCase:
    id: str
    size: int
    pattern: Matrix
    expected_label: str


@dataclass
class CaseResult:
    id: str
    score_cross: Optional[float]
    score_x: Optional[float]
    predicted: str
    expected: Optional[str]
    status: str
    reason: str


@dataclass
class PerformanceRow:
    size: int
    average_ms: Optional[float]
    operation_count: int


def print_banner() -> None:
    print("=== Mini NPU Simulator ===")


def print_section(index: int, title: str) -> None:
    print()
    print("#---------------------------------------")
    print(f"# [{index}] {title}")
    print("#---------------------------------------")


def prompt_mode() -> str:
    while True:
        print()
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("입력 오류: 1 또는 2를 입력하세요.")


def format_cell(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def print_matrix(matrix: Matrix) -> None:
    for row in matrix:
        print(" ".join(format_cell(value) for value in row))


def format_score(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return repr(value)


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def normalize_label(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValueError("라벨은 문자열이어야 합니다.")

    normalized = raw.strip().lower()
    if normalized in {"+", "cross"}:
        return "Cross"
    if normalized == "x":
        return "X"
    raise ValueError(f"지원하지 않는 라벨입니다: {raw}")


def validate_matrix(matrix_data: Any, expected_size: int) -> Tuple[Optional[Matrix], str]:
    if not isinstance(matrix_data, list):
        return None, "schema_error: 2차원 배열(list)이 아닙니다."
    if len(matrix_data) != expected_size:
        return None, (
            f"size_mismatch: 행 수가 {expected_size}가 아닙니다. "
            f"(현재 {len(matrix_data)})"
        )

    validated: Matrix = []
    for row_index, row in enumerate(matrix_data, start=1):
        if not isinstance(row, list):
            return None, f"schema_error: {row_index}번째 행이 배열이 아닙니다."
        if len(row) != expected_size:
            return None, (
                f"size_mismatch: {row_index}번째 행의 열 수가 {expected_size}가 아닙니다. "
                f"(현재 {len(row)})"
            )

        converted_row: List[float] = []
        for col_index, value in enumerate(row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, (
                    f"schema_error: {row_index}행 {col_index}열 값이 숫자가 아닙니다."
                )
            converted_row.append(float(value))
        validated.append(converted_row)

    return validated, ""


def parse_console_row(raw: str, expected_size: int) -> Tuple[Optional[List[float]], str]:
    parts = raw.strip().split()
    if len(parts) != expected_size:
        return None, (
            f"입력 형식 오류: 각 줄에 {expected_size}개의 숫자를 "
            "공백으로 구분해 입력하세요."
        )

    values: List[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return None, (
                f"입력 형식 오류: 각 줄에 {expected_size}개의 숫자를 "
                "공백으로 구분해 입력하세요."
            )
    return values, ""


def prompt_matrix(name: str, size: int) -> Matrix:
    print(f"{name} ({size}줄 입력, 공백 구분)")
    rows: Matrix = []
    for row_index in range(1, size + 1):
        while True:
            raw = input(f"{row_index}행: ")
            row, error = parse_console_row(raw, size)
            if row is not None:
                rows.append(row)
                break
            print(error)
    return rows


def compute_mac(pattern: Matrix, filter_matrix: Matrix) -> float:
    total = 0.0
    for row_index in range(len(pattern)):
        for col_index in range(len(pattern[row_index])):
            total += pattern[row_index][col_index] * filter_matrix[row_index][col_index]
    return total


def classify_scores(score_cross: float, score_x: float, epsilon: float = EPSILON) -> str:
    if abs(score_cross - score_x) < epsilon:
        return "UNDECIDED"
    if score_cross > score_x:
        return "Cross"
    return "X"


def benchmark_mac(pattern: Matrix, filter_matrix: Matrix, repeats: int = BENCHMARK_REPEATS) -> float:
    total_seconds = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        compute_mac(pattern, filter_matrix)
        total_seconds += time.perf_counter() - start
    return (total_seconds / repeats) * 1000.0


def print_performance_table(rows: List[PerformanceRow]) -> None:
    print("크기(N×N)   평균 시간(ms)    연산 횟수(N²)")
    print("-------------------------------------------")
    for row in rows:
        size_label = f"{row.size}×{row.size}"
        avg_label = format_ms(row.average_ms)
        print(f"{size_label:<12}{avg_label:>14}{row.operation_count:>16}")


def default_cross_3x3() -> Matrix:
    return [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]


def load_json_root(json_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"JSON 로드 오류: {json_path.name} 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError as error:
        print(f"JSON 로드 오류: JSON 파싱에 실패했습니다. ({error})")
        return None
    except OSError as error:
        print(f"JSON 로드 오류: 파일을 읽는 중 문제가 발생했습니다. ({error})")
        return None

    if not isinstance(data, dict):
        print("JSON 로드 오류: 최상위 구조는 객체(dict)여야 합니다.")
        return None

    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        print("JSON 로드 오류: top-level에 filters 와 patterns 객체가 필요합니다.")
        return None

    return data


def load_filters(
    filters_data: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Matrix]], List[str], Dict[int, List[str]]]:
    filters_by_size: Dict[int, Dict[str, Matrix]] = {}
    messages: List[str] = []
    issues_by_size: Dict[int, List[str]] = {}

    for size in JSON_SIZES:
        size_key = f"size_{size}"
        raw_bucket = filters_data.get(size_key)
        issues: List[str] = []
        valid_filters: Dict[str, Matrix] = {}

        if not isinstance(raw_bucket, dict):
            issues.append("필터 버킷이 없거나 객체가 아닙니다.")
        else:
            for raw_key, matrix_data in raw_bucket.items():
                try:
                    label = normalize_label(raw_key)
                except ValueError:
                    continue

                if label in valid_filters:
                    issues.append(f"{label} 필터가 중복되었습니다.")
                    continue

                validated_matrix, error = validate_matrix(matrix_data, size)
                if validated_matrix is None:
                    issues.append(f"{label} 필터 오류: {error}")
                    continue
                valid_filters[label] = validated_matrix

        for required_label in ("Cross", "X"):
            if required_label not in valid_filters:
                issues.append(f"{required_label} 필터가 없습니다.")

        if valid_filters:
            filters_by_size[size] = valid_filters
        if issues:
            issues_by_size[size] = issues.copy()

        if issues:
            messages.append(f"✗ {size_key} 필터 로드 실패: {'; '.join(issues)}")
        else:
            messages.append(f"✓ {size_key:<7} 필터 로드 완료 (Cross, X)")

    return filters_by_size, messages, issues_by_size


def parse_pattern_size(case_id: str) -> Optional[int]:
    match = PATTERN_KEY_RE.match(case_id)
    if match is None:
        return None
    return int(match.group(1))


def build_failure_result(
    case_id: str,
    reason: str,
    expected: Optional[str] = None,
) -> CaseResult:
    return CaseResult(
        id=case_id,
        score_cross=None,
        score_x=None,
        predicted="UNDECIDED",
        expected=expected,
        status="FAIL",
        reason=reason,
    )


def parse_json_case(case_id: str, case_data: Any) -> Tuple[Optional[JsonCase], Optional[CaseResult]]:
    size = parse_pattern_size(case_id)
    if size is None:
        return None, build_failure_result(
            case_id,
            "schema_error: 케이스 키는 size_{N}_{idx} 형식이어야 합니다.",
        )

    if not isinstance(case_data, dict):
        return None, build_failure_result(
            case_id,
            "schema_error: 패턴 항목이 객체(dict)가 아닙니다.",
        )

    if "input" not in case_data:
        return None, build_failure_result(
            case_id,
            "schema_error: input 필드가 없습니다.",
        )
    if "expected" not in case_data:
        return None, build_failure_result(
            case_id,
            "schema_error: expected 필드가 없습니다.",
        )

    pattern, matrix_error = validate_matrix(case_data["input"], size)
    if pattern is None:
        return None, build_failure_result(case_id, matrix_error)

    try:
        expected_label = normalize_label(case_data["expected"])
    except ValueError as error:
        return None, build_failure_result(
            case_id,
            f"invalid_label: {error}",
        )

    return JsonCase(case_id, size, pattern, expected_label), None


def evaluate_case(
    case: JsonCase,
    filters_by_size: Dict[int, Dict[str, Matrix]],
    filter_issues_by_size: Dict[int, List[str]],
) -> CaseResult:
    size_filters = filters_by_size.get(case.size, {})
    cross_filter = size_filters.get("Cross")
    x_filter = size_filters.get("X")

    if cross_filter is None or x_filter is None:
        issue_messages = filter_issues_by_size.get(case.size, [])
        if any("size_mismatch" in issue for issue in issue_messages):
            reason = (
                f"size_mismatch: size_{case.size} 필터의 크기가 "
                "패턴 크기와 일치하지 않습니다."
            )
        else:
            reason = (
                f"missing_filter: size_{case.size}의 Cross/X 필터를 "
                "모두 찾을 수 없습니다."
            )
        return build_failure_result(
            case.id,
            reason,
            expected=case.expected_label,
        )

    score_cross = compute_mac(case.pattern, cross_filter)
    score_x = compute_mac(case.pattern, x_filter)
    predicted = classify_scores(score_cross, score_x)

    if predicted == case.expected_label:
        status = "PASS"
        reason = "matched_expected"
    elif predicted == "UNDECIDED":
        status = "FAIL"
        reason = f"undecided_by_epsilon: |Cross-X| < {EPSILON}"
    else:
        status = "FAIL"
        reason = (
            f"prediction_mismatch: predicted {predicted}, "
            f"expected {case.expected_label}"
        )

    return CaseResult(
        id=case.id,
        score_cross=score_cross,
        score_x=score_x,
        predicted=predicted,
        expected=case.expected_label,
        status=status,
        reason=reason,
    )


def print_case_result(result: CaseResult) -> None:
    print(f"--- {result.id} ---")
    print(f"Cross 점수: {format_score(result.score_cross)}")
    print(f"X 점수: {format_score(result.score_x)}")
    expected = result.expected if result.expected is not None else "N/A"
    print(f"판정: {result.predicted} | expected: {expected} | {result.status}")
    if result.status == "FAIL":
        print(f"사유: {result.reason}")


def build_mode1_performance(pattern: Matrix, filter_a: Matrix) -> List[PerformanceRow]:
    average_ms = benchmark_mac(pattern, filter_a)
    return [PerformanceRow(size=3, average_ms=average_ms, operation_count=9)]


def build_mode2_performance(
    filters_by_size: Dict[int, Dict[str, Matrix]],
    benchmark_patterns: Dict[int, Matrix],
) -> List[PerformanceRow]:
    rows: List[PerformanceRow] = []

    benchmark_3x3 = default_cross_3x3()
    rows.append(
        PerformanceRow(
            size=3,
            average_ms=benchmark_mac(benchmark_3x3, benchmark_3x3),
            operation_count=9,
        )
    )

    for size in JSON_SIZES:
        cross_filter = filters_by_size.get(size, {}).get("Cross")
        if cross_filter is None:
            rows.append(
                PerformanceRow(size=size, average_ms=None, operation_count=size * size)
            )
            continue

        pattern = benchmark_patterns.get(size, cross_filter)
        rows.append(
            PerformanceRow(
                size=size,
                average_ms=benchmark_mac(pattern, cross_filter),
                operation_count=size * size,
            )
        )

    return rows


def run_user_input_mode() -> None:
    print_section(1, "필터 입력")
    filter_a = prompt_matrix("필터 A", 3)
    print()
    filter_b = prompt_matrix("필터 B", 3)

    print_section(2, "저장 확인")
    print("필터 A")
    print_matrix(filter_a)
    print()
    print("필터 B")
    print_matrix(filter_b)

    print_section(3, "패턴 입력")
    pattern = prompt_matrix("패턴", 3)

    print_section(4, "MAC 결과")
    score_a = compute_mac(pattern, filter_a)
    score_b = compute_mac(pattern, filter_b)
    performance_row = build_mode1_performance(pattern, filter_a)[0]

    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"연산 시간(평균/{BENCHMARK_REPEATS}회): {format_ms(performance_row.average_ms)} ms")

    if abs(score_a - score_b) < EPSILON:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    elif score_a > score_b:
        print("판정: A")
    else:
        print("판정: B")

    print_section(5, "성능 분석")
    print_performance_table([performance_row])


def run_json_mode(project_root: Path) -> None:
    json_path = project_root / "data.json"
    data = load_json_root(json_path)
    if data is None:
        return

    print_section(1, "필터 로드")
    filters_by_size, filter_messages, filter_issues_by_size = load_filters(data["filters"])
    for message in filter_messages:
        print(message)

    print_section(2, "패턴 분석 (라벨 정규화 적용)")
    results: List[CaseResult] = []
    benchmark_patterns: Dict[int, Matrix] = {}

    for case_id, case_data in data["patterns"].items():
        parsed_case, failure = parse_json_case(case_id, case_data)
        if failure is not None:
            results.append(failure)
            print_case_result(failure)
            continue

        result = evaluate_case(parsed_case, filters_by_size, filter_issues_by_size)
        results.append(result)
        print_case_result(result)

        if parsed_case.size not in benchmark_patterns and result.score_cross is not None:
            benchmark_patterns[parsed_case.size] = parsed_case.pattern

    print_section(3, f"성능 분석 (평균/{BENCHMARK_REPEATS}회)")
    performance_rows = build_mode2_performance(filters_by_size, benchmark_patterns)
    print_performance_table(performance_rows)

    print_section(4, "결과 요약")
    total_count = len(results)
    pass_count = sum(1 for result in results if result.status == "PASS")
    fail_results = [result for result in results if result.status == "FAIL"]

    print(f"총 테스트: {total_count}개")
    print(f"통과: {pass_count}개")
    print(f"실패: {len(fail_results)}개")

    if fail_results:
        print()
        print("실패 케이스:")
        for result in fail_results:
            print(f"- {result.id}: {result.reason}")


def main() -> None:
    print_banner()
    mode = prompt_mode()
    if mode == "1":
        run_user_input_mode()
    else:
        run_json_mode(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
