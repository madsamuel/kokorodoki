from src.utils import parse_srt_file
entries = parse_srt_file('example_subtitles.srt')
for e in entries:
    print(f'{e.index}: {e.start_time:.1f}-{e.end_time:.1f}s ({e.end_time-e.start_time:.1f}s) "{e.text}"')