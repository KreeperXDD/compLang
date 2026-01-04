1. Основные командные строки
	а) python main.py compress document.txt backup 
	б) python main.py compress myfolder backup
	в) python main.py decompress backup.tar.zst extracted
	г) python main.py benchmark-all data.bin test_results
2. Дополнительные опции
	а) для выбора метода сжатия
	--method bz2   (для BZ2 сжатия)
	--method zstd  (для ZSTD сжатия, используется по умолчанию)
	б)что бы показать прогресс бар
	--progress
	в)для изменения уровня сжатия ZSTD
	--level 5
3. Примеры работы программы
	python main.py compress photo.jpg archive --method bz2
	python main.py decompress data.tar.bz2 output --progress
	python main.py benchmark-all project/ test --progress --level 5

	