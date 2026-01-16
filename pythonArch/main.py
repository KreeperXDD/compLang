import argparse
import bz2
import os
import shutil
import sys
import tarfile
import tempfile
import time

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

CHUNK_SIZE = 64 * 1024

@dataclass
class BenchmarkResult:
    method: str
    operation: str
    time_taken: float
    size_before: Optional[int] = None
    size_after: Optional[int] = None
    compression_ratio: Optional[float] = None

def human_size(num: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"

def print_progress(prefix: str, processed: int, total: Optional[int]) -> None:
    if total and total > 0:
        pct = processed / total * 100
        bar_len = 30
        filled = int(bar_len * processed // total)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r{prefix} [{bar}] {pct:6.2f}% ({human_size(processed)}/{human_size(total)})"
        )
    else:
        sys.stdout.write(
            f"\r{prefix} {human_size(processed)}"
        )
    sys.stdout.flush()

def get_archive_extension(source: Path, method: str) -> str:
    if method == "bz2":
        if source.is_dir():
            return ".tar.bz2"
        else:
            return ".bz2"
    elif method == "zstd":
        if source.is_dir():
            return ".tar.zst"
        else:
            return ".zst"
    else:
        raise ValueError(f"Неподдерживаемый метод сжатия: {method}")


def detect_method_by_extension(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".zst") or fn.endswith(".zstd") or fn.endswith(".tar.zstd"):
        return "zstd"
    if fn.endswith(".bz2") or fn.endswith(".tar.bz2"):
        return "bz2"
    raise ValueError("Не удалось определить метод сжатия по расширению. Укажите --method=zstd|bz2")

class Archiver:
    def __init__(self, show_progress: bool = False):
        self.show_progress = show_progress

    # ---------------- TAR ----------------
    def create_tar(self, source: Path, tar_path: Path) -> None:
        with tarfile.open(tar_path, "w") as tf:
            tf.add(source, arcname=source.name)

    def extract_tar(self, tar_path: Path, dest_dir: Path) -> None:
        with tarfile.open(tar_path, "r:*") as tf:
            members = list(tf.getmembers())
            total = len(members)
            for i, member in enumerate(members):
                tf.extract(member, dest_dir)
                extracted_path = dest_dir / member.name
                if member.isfile():
                    extracted_path.chmod(member.mode)
                elif member.isdir():
                    extracted_path.chmod(member.mode)

                if self.show_progress:
                    print_progress("Извлечение TAR", i + 1, total)

        if self.show_progress:
            sys.stdout.write("\n")

    # ---------------- BZ2 ----------------
    def compress_bz2(self, input_path: Path, output_path: Path) -> None:
        total = input_path.stat().st_size
        processed = 0
        with open(input_path, "rb") as fin, bz2.open(output_path, "wb") as fout:
            while True:
                chunk = fin.read(CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)
                processed += len(chunk)
                if self.show_progress:
                    print_progress("Сжатие bz2", processed, total)
        if self.show_progress:
            sys.stdout.write("\n")

    def decompress_bz2(self, input_path: Path, output_path: Path) -> None:
        processed = 0
        total = input_path.stat().st_size
        with bz2.open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            while True:
                chunk = fin.read(CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)
                processed += len(chunk)
                if self.show_progress:
                    print_progress("Распаковка bz2", processed, total)
        if self.show_progress:
            sys.stdout.write("\n")

    # ---------------- ZSTD ----------------
    def compress_zstd_file(self, input_path: Path, output_path: Path, level: int = 3) -> None:
        total = input_path.stat().st_size
        processed = 0

        with tarfile.open(str(output_path), "w:zst", level=level) as tf:
            tf.add(str(input_path), arcname=input_path.name)

            if self.show_progress:
                print_progress("Сжатие zstd", processed, total)

        if self.show_progress:
            sys.stdout.write("\n")

    def compress_zstd_dir(self, input_path: Path, output_path: Path, level: int = 3) -> None:
        with tarfile.open(str(output_path), f"w:zst", level=level) as tf:
            tf.add(str(input_path), arcname=input_path.name)

        if self.show_progress:
            sys.stdout.write("\n")

    def decompress_zstd(self, input_path: Path, output_path: Path) -> None:
        try:
            with tarfile.open(input_path, "r:zst") as tf:
                members = list(tf.getmembers())
                total = len(members)

                for i, member in enumerate(members):
                    try:
                        tf.extract(member, str(output_path))
                        try:
                            extracted_path = output_path / member.name
                            if extracted_path.exists():
                                extracted_path.chmod(member.mode)
                        except (PermissionError, FileNotFoundError):
                            pass
                    except PermissionError as e:
                        print(f"Не удалось извлечь файлы, ошибка:{e}")
                        continue
                    if self.show_progress:
                        print_progress("Распаковка zstd", i + 1, total)
            if self.show_progress:
                sys.stdout.write("\n")

        except tarfile.ReadError as e:
            raise ValueError(f"Архив {input_path} не является корректным архивом. Ошибка: {e}")

    # ---------------- HIGH LEVEL ----------------
    def compress(self, source: Path, destination: Path, method: Optional[str] = None, zstd_level: int = 3) -> Tuple[
        float, int, int]:
        start = time.time()
        original_size = source.stat().st_size if source.is_file() else self._get_directory_size(source)

        if method is None:
            method = detect_method_by_extension(destination.name)

        if method not in ("bz2", "zstd"):
            raise ValueError("Метод должен быть 'bz2' или 'zstd'")

        if not destination.suffix:
            ext = get_archive_extension(source, method)
            destination = destination.with_suffix(ext)
        else:
            expected_ext = get_archive_extension(source, method)

            if not destination.name.lower().endswith(expected_ext.lower()):
                destination = destination.with_suffix(expected_ext)
                print(f"Заменяем расширение на {expected_ext}")

        print(f"Создаём архив: {destination.name}")

        if source.is_dir():
            with tempfile.NamedTemporaryFile(prefix="archutil_", suffix=".tar", delete=False) as tmp:
                tmp_tar = Path(tmp.name)
            try:
                print(f"Создаём TAR из директории: {source}")
                self.create_tar(source, tmp_tar)

                print(f"Сжимаем TAR методом {method} → {destination}")

                if method == "bz2":
                    self.compress_bz2(tmp_tar, destination)
                else:
                    self.compress_zstd_dir(source, destination, level=zstd_level)
            finally:
                if tmp_tar.exists():
                    tmp_tar.unlink()
        else:
            if method == "bz2":
                self.compress_bz2(source, destination)
            else:
                self.compress_zstd_file(source, destination, level=zstd_level)

        elapsed = time.time() - start
        compressed_size = destination.stat().st_size

        print(f"Сжатие завершено за {elapsed:.2f} сек")
        print(f"Исходный размер: {human_size(original_size)}")
        print(f"Размер архива: {human_size(compressed_size)}")
        print(f"Коэффициент сжатия: {original_size / compressed_size:.2f}x")

        return elapsed, original_size, compressed_size

    def decompress(self, archive: Path, destination: Path, method: Optional[str] = None) -> float:
        start = time.time()

        if method is None:
            method = detect_method_by_extension(archive.name)

        if method not in ("bz2", "zstd"):
            raise ValueError("Метод должен быть 'bz2' или 'zstd'")

        destination.mkdir(parents=True, exist_ok=True)

        if method == "bz2":
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(prefix="archutil_bz2_", delete=False) as t:
                    tmp = Path(t.name)
                self.decompress_bz2(archive, tmp)

                if tarfile.is_tarfile(tmp):
                    self.extract_tar(tmp, destination)
                else:
                    if archive.name.lower().endswith(".bz2"):
                        original_name = archive.name[:-4]
                        target = destination / original_name
                    else:
                        target = destination / archive.name

                    shutil.copy2(tmp, target)

                    try:
                        original_stat = archive.stat()
                        target.chmod(original_stat.st_mode)
                    except PermissionError:
                        pass
                    tmp = None
            finally:
                if tmp and tmp.exists():
                    tmp.unlink()
        else:
            self.decompress_zstd(archive, destination)

        elapsed = time.time() - start
        print(f"Распаковка завершена за {elapsed:.2f} сек")
        return elapsed

    def _get_directory_size(self, directory: Path) -> int:
        total = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                total += (Path(root) / file).stat().st_size
        return total

    def benchmark_all(self, source: Path, base_destination: Path, zstd_level: int = 3) -> None:
        print("=" * 80)
        print("ЗАПУСК ПОЛНОГО БЕНЧМАРКА")
        print("=" * 80)
        print(f"Источник: {source}")
        print(f"Тип: {'директория' if source.is_dir() else 'файл'}")
        print(f"Размер: {human_size(source.stat().st_size if source.is_file() else self._get_directory_size(source))}")
        print("-" * 80)

        results = []
        methods = ["bz2", "zstd"]

        for method in methods:
            print(f"\n ТЕСТИРУЕМ МЕТОД: {method.upper()}")
            print("-" * 40)

            archive_name = base_destination.with_suffix(get_archive_extension(source, method))
            print(f"Создаём архив: {archive_name}")

            start = time.time()
            original_size = source.stat().st_size if source.is_file() else self._get_directory_size(source)

            if source.is_dir():
                with tempfile.NamedTemporaryFile(prefix="archutil_", suffix=".tar", delete=False) as tmp:
                    tmp_tar = Path(tmp.name)
                try:
                    self.create_tar(source, tmp_tar)
                    if method == "bz2":
                        self.compress_bz2(tmp_tar, archive_name)
                    else:
                        self.compress_zstd_dir(source, archive_name, level=zstd_level)
                finally:
                    if tmp_tar.exists():
                        tmp_tar.unlink()
            else:
                if method == "bz2":
                    self.compress_bz2(source, archive_name)
                else:
                    self.compress_zstd_file(source, archive_name, level=zstd_level)

            compress_time = time.time() - start
            compressed_size = archive_name.stat().st_size
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

            compress_result = BenchmarkResult(
                method=method,
                operation="compress",
                time_taken=compress_time,
                size_before=original_size,
                size_after=compressed_size,
                compression_ratio=compression_ratio
            )
            results.append(compress_result)

            print(f"Время сжатия: {compress_time:.3f} сек")
            print(f"Коэффициент сжатия: {compression_ratio:.2f}x")

            extract_dir = base_destination.parent / f"extracted_{method}"
            print(f"Распаковываем в: {extract_dir}")

            start = time.time()
            if method == "bz2":
                self.decompress(archive_name, extract_dir, method)
            else:
                self.decompress_zstd(archive_name, extract_dir)

            decompress_time = time.time() - start

            decompress_result = BenchmarkResult(
                method=method,
                operation="decompress",
                time_taken=decompress_time
            )
            results.append(decompress_result)

            print(f"Время распаковки: {decompress_time:.3f} сек")

            if archive_name.exists():
                archive_name.unlink()
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

        self._print_benchmark_table(results)

    def _print_benchmark_table(self, results: List[BenchmarkResult]) -> None:
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
        print("=" * 80)

        grouped = {}
        for result in results:
            if result.method not in grouped:
                grouped[result.method] = {}
            grouped[result.method][result.operation] = result

        print(
            f"{'Метод':<10} {'Операция':<12} {'Время, сек':<12} {'Размер до':<12} {'Размер после':<12} {'Коэф. сжатия':<12}")
        print("-" * 80)

        for method in ["bz2", "zstd"]:
            if method in grouped:
                compress = grouped[method].get("compress")
                decompress = grouped[method].get("decompress")

                if compress:
                    print(f"{method:<10} {'Сжатие':<12} {compress.time_taken:<12.3f} "
                          f"{human_size(compress.size_before) if compress.size_before else 'N/A':<12} "
                          f"{human_size(compress.size_after) if compress.size_after else 'N/A':<12} "
                          f"{compress.compression_ratio:<12.2f}")

                if decompress:
                    print(f"{method:<10} {'Распаковка':<12} {decompress.time_taken:<12.3f} "
                          f"{'N/A':<12} {'N/A':<12} {'N/A':<12}")

        print("-" * 80)

        bz2_compress = grouped.get("bz2", {}).get("compress")
        zstd_compress = grouped.get("zstd", {}).get("compress")
        bz2_decompress = grouped.get("bz2", {}).get("decompress")
        zstd_decompress = grouped.get("zstd", {}).get("decompress")

        if bz2_compress and zstd_compress:
            print("\n ПОБЕДИТЕЛИ:")
            print("-" * 40)


            if bz2_compress.compression_ratio and zstd_compress.compression_ratio:
                if bz2_compress.compression_ratio > zstd_compress.compression_ratio:
                    print(
                        f"• Лучшее сжатие: BZ2 ({bz2_compress.compression_ratio:.2f}x против {zstd_compress.compression_ratio:.2f}x)")
                else:
                    print(
                        f"• Лучшее сжатие: ZSTD ({zstd_compress.compression_ratio:.2f}x против {bz2_compress.compression_ratio:.2f}x)")

            if bz2_compress.time_taken < zstd_compress.time_taken:
                print(
                    f"• Быстрее сжатие: BZ2 ({bz2_compress.time_taken:.3f} сек против {zstd_compress.time_taken:.3f} сек)")
            else:
                print(
                    f"• Быстрее сжатие: ZSTD ({zstd_compress.time_taken:.3f} сек против {bz2_compress.time_taken:.3f} сек)")

            if bz2_decompress and zstd_decompress:
                if bz2_decompress.time_taken < zstd_decompress.time_taken:
                    print(
                        f"• Быстрее распаковка: BZ2 ({bz2_decompress.time_taken:.3f} сек против {zstd_decompress.time_taken:.3f} сек)")
                else:
                    print(
                        f"• Быстрее распаковка: ZSTD ({zstd_decompress.time_taken:.3f} сек против {bz2_decompress.time_taken:.3f} сек)")

        print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Archutil — утилита сжатия (bz2, zstd)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress", help="Сжать файл/директорию")
    pc.add_argument("source", type=Path, help="Исходный файл или директория")
    pc.add_argument("destination", type=Path, help="Путь для сохранения архива")
    pc.add_argument("--method", choices=["bz2", "zstd"], help="Метод сжатия")
    pc.add_argument("--level", type=int, default=3, help="Уровень сжатия для zstd (1-22)")
    pc.add_argument("--progress", action="store_true", help="Показывать прогресс")

    pd = sub.add_parser("decompress", help="Распаковать архив")
    pd.add_argument("archive", type=Path, help="Архив для распаковки")
    pd.add_argument("destination", type=Path, help="Целевая директория")
    pd.add_argument("--method", choices=["bz2", "zstd"], help="Метод сжатия")
    pd.add_argument("--progress", action="store_true", help="Показывать прогресс")

    pba = sub.add_parser("benchmark-all", help="Запустить полный бенчмарк")
    pba.add_argument("source", type=Path, help="Исходный файл или директория")
    pba.add_argument("destination", type=Path, help="Базовое имя для тестовых архивов")
    pba.add_argument("--level", type=int, default=3, help="Уровень сжатия для zstd (1-22)")
    pba.add_argument("--progress", action="store_true", help="Показывать прогресс")

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        arch = Archiver(show_progress=getattr(args, "progress", False))

        if args.cmd == "compress":
            arch.compress(args.source, args.destination, method=args.method, zstd_level=args.level)

        elif args.cmd == "decompress":
            arch.decompress(args.archive, args.destination, method=args.method)

        elif args.cmd == "benchmark-all":
            arch.benchmark_all(args.source, args.destination, zstd_level=args.level)

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()