import base64
import bisect
import json
import os
import re
import shutil
import subprocess
import threading
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from agno.exceptions import PathSecurityError
from agno.tools import Toolkit
from agno.tools._local_file_utils import DEFAULT_EXCLUDE_PATTERNS, path_matches_exclude
from agno.utils.log import log_debug, log_error
from agno.utils.path_safety import safe_join_relative_path

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".sql",
    ".sh",
    ".toml",
    ".cfg",
    ".ini",
    ".log",
    ".rst",
}

GREP_MAX_LIMIT = 1000
GREP_TIMEOUT_SECONDS = 30
GREP_MAX_LINE_LENGTH = 500
GREP_BINARY_SNIFF_BYTES = 8192


def _format_size(size: int) -> str:
    """Format a file size in bytes to a human-readable string."""
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
        size /= 1024  # type: ignore
    return f"{size:.1f}GB"


def _extract_snippet(content: str, query: str, context_chars: int = 200) -> str:
    """Extract a snippet of content around the first occurrence of query."""
    lower_content = content.lower()
    lower_query = query.lower()
    idx = lower_content.find(lower_query)
    if idx == -1:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(content), idx + len(query) + context_chars)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


def _glob_to_regex(glob: str) -> "re.Pattern[str]":
    """Translate a path glob (``*``, ``?``, ``**``) into a regex matched against a posix path."""
    out: List[str] = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out))


def _glob_matches(glob: str, rel_path: str) -> bool:
    """Match ``rel_path`` (posix, relative to the base directory) against an include glob.

    A glob without ``/`` matches the file name at any depth; a glob with ``/`` is anchored
    at the base directory.
    """
    if "/" not in glob:
        return fnmatch(rel_path.rsplit("/", 1)[-1], glob)
    return _glob_to_regex(glob).fullmatch(rel_path) is not None


def _clip_line(text: str) -> str:
    if len(text) > GREP_MAX_LINE_LENGTH:
        return text[:GREP_MAX_LINE_LENGTH] + "..."
    return text


class FileTools(Toolkit):
    """Toolkit for read/write access to a local directory tree.

    ``exclude_patterns`` here is a **listing filter, not an access boundary**: results
    from ``list_files``, ``search_files``, ``search_content`` and ``grep`` skip matching paths,
    but ``read_file``, ``read_file_chunk``, ``save_file``, ``replace_file_chunk`` and
    ``delete_file`` do not consult it — an agent that names an excluded path directly
    still reads or writes it. The shared default list includes credential files, so an
    agent given ``FileTools`` over a directory holding secrets can still read them by
    name. Use ``agno.tools.workspace.Workspace`` where exclusion has to be enforced:
    there the same patterns refuse every path the agent names.

    By default, results from ``list_files``, ``search_files``, ``search_content`` and ``grep``
    skip common noise directories (``.venv``, ``.venvs``, ``.context``,
    ``.git``, ``__pycache__``, ``node_modules``, etc.). See
    ``DEFAULT_EXCLUDE_PATTERNS`` for the full list.

    To customize:
    - Pass ``exclude_patterns=[...]`` with your own list of fnmatch-style patterns.
    - Pass ``exclude_patterns=[]`` to disable exclusion entirely (prior behavior).

    Each pattern is matched with ``fnmatch`` against *any path component* of the
    file's path relative to ``base_dir``. A file is excluded if any component
    matches any pattern, so ``.git`` will exclude both ``.git/`` at the root
    and ``vendor/thing/.git/`` nested deep. An entry prefixed with ``!`` is an
    exemption: a component matching it is never excluded, whatever else it
    matches. The defaults use this to keep committed env templates
    (``.env.example`` and friends) listed while real env files stay out of
    listings.

    Note: ``exclude_patterns`` does not parse ``.gitignore`` files — it only
    applies the literal patterns provided.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        enable_save_file: bool = True,
        enable_read_file: bool = True,
        enable_delete_file: bool = False,
        enable_list_files: bool = True,
        enable_search_files: bool = True,
        enable_read_file_chunk: bool = True,
        enable_replace_file_chunk: bool = True,
        enable_search_content: bool = True,
        enable_grep: bool = True,
        expose_base_directory: bool = False,
        max_file_length: int = 10000000,
        max_file_lines: int = 100000,
        line_separator: str = "\n",
        exclude_patterns: Optional[List[str]] = None,
        all: bool = False,
        **kwargs,
    ):
        self.base_dir: Path = (base_dir or Path.cwd()).resolve()

        tools: List[Any] = []
        self.max_file_length = max_file_length
        self.max_file_lines = max_file_lines
        self.line_separator = line_separator
        self.expose_base_directory = expose_base_directory
        self.exclude_patterns: List[str] = (
            exclude_patterns if exclude_patterns is not None else list(DEFAULT_EXCLUDE_PATTERNS)
        )
        # Resolved once; grep falls back to Python ``re`` when ripgrep is not installed.
        self._rg_path: Optional[str] = shutil.which("rg")
        if all or enable_save_file:
            tools.append(self.save_file)
        if all or enable_read_file:
            tools.append(self.read_file)
        if all or enable_list_files:
            tools.append(self.list_files)
        if all or enable_search_files:
            tools.append(self.search_files)
        if all or enable_delete_file:
            tools.append(self.delete_file)
        if all or enable_read_file_chunk:
            tools.append(self.read_file_chunk)
        if all or enable_replace_file_chunk:
            tools.append(self.replace_file_chunk)
        if all or enable_search_content:
            tools.append(self.search_content)
        if all or enable_grep:
            tools.append(self.grep)

        super().__init__(name="file_tools", tools=tools, **kwargs)

    def _is_excluded(self, path: Path) -> bool:
        """Return True if any component of ``path`` (relative to ``base_dir``) matches an exclude pattern."""
        return path_matches_exclude(path, self.base_dir, self.exclude_patterns)

    def save_file(self, contents: str, file_name: str, overwrite: bool = True, encoding: str = "utf-8") -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        :param contents: The contents to save.
        :param file_name: The name of the file to save to.
        :param overwrite: Overwrite the file if it already exists.
        :return: The file name if successful, otherwise returns an error message.
        """
        try:
            safe, file_path = self._check_path(file_name, self.base_dir)
            if not (safe):
                log_error(f"Attempted to save file: {file_name}")
                return "Error saving file"
            log_debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(contents, encoding=encoding)
            log_debug(f"Saved: {file_path}")
            return str(file_name)
        except Exception as e:
            log_error(f"Error saving to file: {str(e)}")
            return f"Error saving to file: {e}"

    def read_file_chunk(self, file_name: str, start_line: int, end_line: int, encoding: str = "utf-8") -> str:
        """Reads the contents of the file `file_name` and returns lines from start_line to end_line.

        :param file_name: The name of the file to read.
        :param start_line: Number of first line in the returned chunk
        :param end_line: Number of the last line in the returned chunk
        :param encoding: Encoding to use, default - utf-8

        :return: The contents of the selected chunk
        """
        try:
            log_debug(f"Reading file: {file_name}")
            safe, file_path = self._check_path(file_name, self.base_dir)
            if not (safe):
                log_error(f"Attempted to read file: {file_name}")
                return "Error reading file"
            contents = file_path.read_text(encoding=encoding)
            lines = contents.split(self.line_separator)
            return self.line_separator.join(lines[start_line : end_line + 1])
        except Exception as e:
            log_error(f"Error reading file: {str(e)}")
            return f"Error reading file: {e}"

    def replace_file_chunk(
        self, file_name: str, start_line: int, end_line: int, chunk: str, encoding: str = "utf-8"
    ) -> str:
        """Reads the contents of the file, replaces lines
        between start_line and end_line with chunk and writes the file

        :param file_name: The name of the file to process.
        :param start_line: Number of first line in the replaced chunk
        :param end_line: Number of the last line in the replaced chunk
        :param chunk: String to be inserted instead of lines from start_line to end_line. Can have multiple lines.
        :param encoding: Encoding to use, default - utf-8

        :return: file name if successful, error message otherwise
        """
        try:
            log_debug(f"Patching file: {file_name}")
            if start_line < 0 or end_line < start_line:
                return "Error patching file: require 0 <= start_line <= end_line"
            safe, file_path = self._check_path(file_name, self.base_dir)
            if not (safe):
                log_error(f"Attempted to read file: {file_name}")
                return "Error reading file"
            contents = file_path.read_text(encoding=encoding)
            lines = contents.split(self.line_separator)
            # A trailing separator terminates the last line rather than adding a line.
            # Keep its empty split element for reconstruction so the terminator survives.
            line_count = len(lines) - 1 if lines[-1] == "" else len(lines)
            if end_line >= line_count:
                return f"Error patching file: end_line must be less than {line_count}"
            start = lines[0:start_line]
            end = lines[end_line + 1 :]
            return self.save_file(
                file_name=file_name, contents=self.line_separator.join(start + [chunk] + end), encoding=encoding
            )
        except Exception as e:
            log_error(f"Error patching file: {str(e)}")
            return f"Error patching file: {e}"

    def read_file(self, file_name: str, encoding: str = "utf-8") -> str:
        """Reads the contents of the file `file_name` and returns the contents if successful.

        :param file_name: The name of the file to read.
        :param encoding: Encoding to use, default - utf-8
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            log_debug(f"Reading file: {file_name}")
            safe, file_path = self._check_path(file_name, self.base_dir)
            if not (safe):
                log_error(f"Attempted to read file: {file_name}")
                return "Error reading file"
            contents = file_path.read_text(encoding=encoding)
            if len(contents) > self.max_file_length:
                return "Error reading file: file too long. Use read_file_chunk instead"
            if len(contents.split(self.line_separator)) > self.max_file_lines:
                return "Error reading file: file too long. Use read_file_chunk instead"

            return str(contents)
        except Exception as e:
            log_error(f"Error reading file: {str(e)}")
            return f"Error reading file: {e}"

    def delete_file(self, file_name: str) -> str:
        """Deletes a file
        :param file_name: Name of the file to delete

        :return: Empty string, if operation succeeded, otherwise returns an error message
        """
        safe, path = self._check_path(file_name, self.base_dir)
        try:
            if safe:
                if path.is_dir():
                    path.rmdir()
                    return ""
                path.unlink()
                return ""
            else:
                log_error(f"Attempt to delete file outside {self.base_dir}: {file_name}")
                return "Incorrect file_name"
        except Exception as e:
            log_error(f"Error removing {file_name}: {str(e)}")
            return f"Error removing file: {e}"

    def list_files(self, directory: str = ".") -> str:
        """Returns a list of files in directory
        :param directory: (Optional) name of directory to list.

        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            d = self.base_dir
            if directory:
                safe, d = self._check_path(directory, self.base_dir)
                if not safe:
                    return "{}"
            log_debug(f"Reading files in : {d}")
            files = []
            for file_path in d.iterdir():
                rel_path = file_path.relative_to(self.base_dir).as_posix()
                try:
                    safe_join_relative_path(self.base_dir, rel_path)
                except PathSecurityError:
                    continue
                if self._is_excluded(file_path):
                    continue
                files.append(rel_path)
            return json.dumps(files, indent=4)
        except Exception as e:
            log_error(f"Error reading files: {str(e)}")
            return f"Error reading files: {e}"

    def search_files(self, pattern: str) -> str:
        """Searches for files in the base directory that match the pattern

        :param pattern: The pattern to search for, e.g. "*.txt", "file*.csv", "**/*.py".
        :return: JSON formatted list of matching file paths, or error message.
        """
        try:
            if not pattern or not pattern.strip():
                return "Error: Pattern cannot be empty"

            log_debug(f"Searching files in {self.base_dir} with pattern {pattern}")
            matching_files = []
            for p in self.base_dir.glob(pattern):
                try:
                    safe_join_relative_path(self.base_dir, p.relative_to(self.base_dir).as_posix())
                except PathSecurityError:
                    continue
                if not self._is_excluded(p):
                    matching_files.append(p)
            result = None
            if self.expose_base_directory:
                file_paths = [str(file_path) for file_path in matching_files]
                result = {
                    "pattern": pattern,
                    "matches_found": len(file_paths),
                    "base_directory": str(self.base_dir),
                    "files": file_paths,
                }
            else:
                file_paths = [file_path.relative_to(self.base_dir).as_posix() for file_path in matching_files]

                result = {
                    "pattern": pattern,
                    "matches_found": len(file_paths),
                    "files": file_paths,
                }
            log_debug(f"Found {len(file_paths)} files matching pattern {pattern}")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error searching files with pattern '{pattern}': {e}"
            log_error(error_msg)
            return error_msg

    def search_content(self, query: str, directory: Optional[str] = None, limit: int = 10) -> str:
        """Search file contents within the base directory for a query string (case-insensitive).

        Only text files (by extension) under 500KB are searched.

        :param query: The text to search for in file contents.
        :param directory: Optional subdirectory to scope the search to.
        :param limit: Maximum number of matching files to return (default 10).
        :return: JSON formatted results with file paths, sizes, and content snippets.
        """
        try:
            if not query or not query.strip():
                return "Error: Query cannot be empty"

            search_dir = self.base_dir
            if directory:
                safe, search_dir = self._check_path(directory, self.base_dir)
                if not safe:
                    log_error(f"Attempted to search outside base directory: {directory}")
                    return "Error: Directory is outside the allowed base directory"

            if not search_dir.is_dir():
                return f"Error: '{directory}' is not a directory"

            log_debug(f"Searching file contents in {search_dir} for '{query}'")
            lower_query = query.lower()
            matches: List[dict] = []
            max_file_size = 500 * 1024  # 500KB

            walk_done = False
            for dirpath, dirnames, filenames in os.walk(search_dir):
                if walk_done:
                    break
                # Prune excluded directories in place so os.walk doesn't descend into them.
                dirnames[:] = [d for d in dirnames if not self._is_excluded(Path(dirpath) / d)]
                for filename in filenames:
                    if len(matches) >= limit:
                        walk_done = True
                        break
                    file_path = Path(dirpath) / filename
                    try:
                        safe_join_relative_path(self.base_dir, file_path.relative_to(self.base_dir).as_posix())
                    except PathSecurityError:
                        continue
                    if self._is_excluded(file_path):
                        continue
                    if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                        continue
                    try:
                        if file_path.stat().st_size > max_file_size:
                            continue
                    except OSError:
                        continue

                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue

                    if lower_query in content.lower():
                        rel_path = file_path.relative_to(self.base_dir).as_posix()
                        snippet = _extract_snippet(content, query)
                        matches.append(
                            {
                                "file": rel_path,
                                "size": _format_size(file_path.stat().st_size),
                                "snippet": snippet,
                            }
                        )

            result = {
                "query": query,
                "matches_found": len(matches),
                "files": matches,
            }
            log_debug(f"Found {len(matches)} files containing '{query}'")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error searching content for '{query}': {e}"
            log_error(error_msg)
            return error_msg

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        include: Optional[str] = None,
        ignore_case: bool = False,
        context: int = 0,
        limit: int = 250,
        multiline: bool = False,
    ) -> str:
        """Search file contents with a regular expression.

        Uses ripgrep when it is installed and Python's ``re`` otherwise; the result shape is the same.
        Binary files are skipped. If the result has "truncated": true, narrow the search
        (a more specific pattern, ``path`` or ``include``) instead of raising ``limit``.

        :param pattern: Regular expression to search for, e.g. "def \\w+" or "TODO.*priority".
        :param path: Optional file or directory, relative to the base directory, to scope the search to.
        :param output_mode: "files_with_matches" lists matching file paths (default), "content" returns
            matching lines with line numbers, "count" returns the number of matches per file.
        :param include: Optional glob to filter files, e.g. "*.py" (any depth) or "src/**/*.tsx" (from the base directory).
        :param ignore_case: Case-insensitive search (default: False).
        :param context: Lines of context around each match, only used with output_mode "content" (default: 0).
        :param limit: Maximum number of files (or matching lines in "content" mode) to return (default: 250, maximum: 1000).
        :param multiline: Allow the pattern to match across line boundaries, with "." matching newlines (default: False).
        :return: JSON with "results" and "truncated", or an error message.
        """
        try:
            if not pattern:
                return "Error: Pattern cannot be empty"
            if output_mode not in ("files_with_matches", "content", "count"):
                return "Error: output_mode must be one of 'files_with_matches', 'content' or 'count'"

            search_path = self.base_dir
            if path:
                safe, search_path = self._check_path(path, self.base_dir)
                if not safe:
                    log_error(f"Attempted to grep outside base directory: {path}")
                    return "Error: Path is outside the allowed base directory"
            if not search_path.exists():
                return f"Error: Path not found: {path}"

            limit = max(1, min(int(limit), GREP_MAX_LIMIT))
            context = max(0, int(context)) if output_mode == "content" else 0

            log_debug(f"Grep for '{pattern}' in {search_path}")
            if self._rg_path:
                hits, truncated = self._grep_ripgrep(
                    self._rg_path, pattern, search_path, output_mode, include, ignore_case, context, limit, multiline
                )
            else:
                try:
                    regex = re.compile(
                        pattern,
                        (re.IGNORECASE if ignore_case else 0) | (re.DOTALL | re.MULTILINE if multiline else 0),
                    )
                except re.error as e:
                    return f"Error: Invalid regular expression: {e}"
                hits, truncated = self._grep_python(regex, search_path, output_mode, include, context, limit, multiline)

            result: Dict[str, Any] = {
                "pattern": pattern,
                "output_mode": output_mode,
                "results": hits,
                "truncated": truncated,
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error running grep for '{pattern}': {e}"
            log_error(error_msg)
            return error_msg

    def _grep_visible(self, file_path: Path, include: Optional[str], cache: Dict[Path, Optional[str]]) -> Optional[str]:
        """Return the base_dir-relative posix path if ``file_path`` may appear in grep results, else None."""
        if file_path in cache:
            return cache[file_path]
        rel: Optional[str] = None
        try:
            rel = file_path.relative_to(self.base_dir).as_posix()
            safe_join_relative_path(self.base_dir, rel)
            if self._is_excluded(file_path) or (include and not _glob_matches(include, rel)):
                rel = None
        except (ValueError, PathSecurityError, OSError):
            rel = None
        cache[file_path] = rel
        return rel

    def _grep_python(
        self,
        regex: "re.Pattern[str]",
        search_path: Path,
        output_mode: str,
        include: Optional[str],
        context: int,
        limit: int,
        multiline: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        hits: List[Dict[str, Any]] = []
        cache: Dict[Path, Optional[str]] = {}
        match_total = 0

        def candidate_files() -> Iterator[Path]:
            if search_path.is_file():
                yield search_path
                return
            for dirpath, dirnames, filenames in os.walk(search_path):
                dirnames[:] = sorted(d for d in dirnames if not self._is_excluded(Path(dirpath) / d))
                for filename in sorted(filenames):
                    yield Path(dirpath) / filename

        for file_path in candidate_files():
            rel = self._grep_visible(file_path, include, cache)
            if rel is None:
                continue
            try:
                if file_path.stat().st_size > self.max_file_length:
                    continue
                raw = file_path.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:GREP_BINARY_SNIFF_BYTES]:
                continue
            text = raw.decode("utf-8", errors="replace")
            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]

            matches: List[Tuple[int, int]] = []  # (first line, last line), 1-based
            if multiline:
                starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
                seen = set()
                for m in regex.finditer(text):
                    first = bisect.bisect_right(starts, m.start())
                    last = bisect.bisect_right(starts, max(m.end() - 1, m.start()))
                    if first not in seen:
                        seen.add(first)
                        matches.append((first, max(first, last)))
            else:
                matches = [(i, i) for i, ln in enumerate(lines, 1) if regex.search(ln)]
            if not matches:
                continue

            if output_mode == "files_with_matches":
                if len(hits) >= limit:
                    return hits, True
                hits.append({"file": rel})
            elif output_mode == "count":
                if len(hits) >= limit:
                    return hits, True
                hits.append({"file": rel, "count": len(matches)})
            else:
                emit: Dict[int, bool] = {}
                truncated = False
                for first, last in matches:
                    if match_total >= limit:
                        truncated = True
                        break
                    match_total += 1
                    for ln in range(max(1, first - context), min(len(lines), last + context) + 1):
                        emit[ln] = emit.get(ln, False) or first <= ln <= last
                for ln in sorted(emit):
                    hits.append({"file": rel, "line": ln, "text": _clip_line(lines[ln - 1]), "is_match": emit[ln]})
                if truncated:
                    return hits, True
        return hits, False

    def _grep_exclude_globs(self) -> List[str]:
        """Exclude patterns that can be handed to ripgrep as globs; the rest are filtered afterwards."""
        exempt = [p[1:] for p in self.exclude_patterns if p.startswith("!")]
        return [
            p
            for p in self.exclude_patterns
            if not p.startswith("!") and "/" not in p and not any(fnmatch(name, p) for name in exempt)
        ]

    def _grep_ripgrep(
        self,
        rg_path: str,
        pattern: str,
        search_path: Path,
        output_mode: str,
        include: Optional[str],
        ignore_case: bool,
        context: int,
        limit: int,
        multiline: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        # Hidden files and .gitignore are handled by exclude_patterns, as in the Python backend.
        cmd = [rg_path, "--json", "--hidden", "--no-ignore", "--no-messages", "--sort=path"]
        cmd.append(f"--max-filesize={self.max_file_length}")
        if ignore_case:
            cmd.append("--ignore-case")
        if multiline:
            cmd.extend(["--multiline", "--multiline-dotall"])
        if output_mode == "files_with_matches":
            cmd.append("--max-count=1")
        if context > 0:
            cmd.append(f"--context={context}")
        if include:
            cmd.extend(["--glob", include])
        for exclude in self._grep_exclude_globs():
            cmd.extend(["--glob", f"!{exclude}"])
        cmd.extend(["-e", pattern, "--", str(search_path)])

        hits: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        cache: Dict[Path, Optional[str]] = {}
        match_total = 0
        last_match: Tuple[str, int] = ("", 0)
        truncated = False
        timed_out = False

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(self.base_dir), text=True, errors="replace"
        )

        def on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            proc.kill()

        timer = threading.Timer(GREP_TIMEOUT_SECONDS, on_timeout)
        timer.start()
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                try:
                    message = json.loads(raw_line)
                except ValueError:
                    continue
                kind = message.get("type")
                if kind not in ("match", "context"):
                    continue
                data = message["data"]
                path_text = data.get("path", {}).get("text")
                if path_text is None:
                    continue
                rel = self._grep_visible(Path(path_text), include, cache)
                if rel is None:
                    continue

                if output_mode == "files_with_matches":
                    if kind == "match" and not (hits and hits[-1]["file"] == rel):
                        if len(hits) >= limit:
                            truncated = True
                            break
                        hits.append({"file": rel})
                elif output_mode == "count":
                    if kind == "match":
                        if rel not in counts and len(counts) >= limit:
                            truncated = True
                            break
                        counts[rel] = counts.get(rel, 0) + 1
                else:
                    if kind == "match":
                        if match_total >= limit:
                            truncated = True
                            break
                        match_total += 1
                    lines_field = data.get("lines", {})
                    if "text" in lines_field:
                        text = lines_field["text"]
                    else:
                        text = base64.b64decode(lines_field.get("bytes", "")).decode("utf-8", errors="replace")
                    line_number = data["line_number"]
                    entries = text.rstrip("\n").split("\n") if kind == "match" else [text.rstrip("\n")]
                    for offset, entry in enumerate(entries):
                        hits.append(
                            {
                                "file": rel,
                                "line": line_number + offset,
                                "text": _clip_line(entry.rstrip("\r")),
                                "is_match": kind == "match",
                            }
                        )
                    if kind == "match":
                        last_match = (rel, line_number + len(entries) - 1)
            if truncated:
                proc.kill()
            proc.wait()
            stderr = proc.stderr.read().strip() if proc.stderr else ""
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        if timed_out:
            raise TimeoutError(f"grep timed out after {GREP_TIMEOUT_SECONDS} seconds")
        if proc.returncode not in (0, 1) and not truncated and not hits and not counts and stderr:
            raise RuntimeError(stderr)

        if output_mode == "count":
            hits = [{"file": file, "count": count} for file, count in counts.items()]
        elif truncated and output_mode == "content":
            # Drop context that belongs to the first match that was cut off.
            while hits and not hits[-1]["is_match"]:
                if hits[-1]["file"] == last_match[0] and hits[-1]["line"] <= last_match[1] + context:
                    break
                hits.pop()
        return hits, truncated
