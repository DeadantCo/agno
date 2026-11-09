import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from agno.tools.file import FileTools
from agno.tools.file.file import GREP_MAX_LIMIT


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (0, -1), (3, 1), (4, 4), (1, 8)])
def test_replace_file_chunk_rejects_invalid_ranges_without_writing(tmp_path, start, end):
    file_path = tmp_path / "sample.txt"
    original = b"a\nb\nc\nd"
    file_path.write_bytes(original)
    tools = FileTools(base_dir=tmp_path)

    result = tools.replace_file_chunk("sample.txt", start, end, "replacement")

    assert result.startswith("Error patching file:")
    assert file_path.read_bytes() == original


def test_replace_file_chunk_replaces_inclusive_valid_range(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("a\nb\nc\nd", encoding="utf-8")
    tools = FileTools(base_dir=tmp_path)

    assert tools.replace_file_chunk("sample.txt", 1, 2, "replacement") == "sample.txt"
    assert file_path.read_text(encoding="utf-8") == "a\nreplacement\nd"


@pytest.mark.parametrize(
    ("original", "separator", "line_count"),
    [
        ("a\nb\nc\n", "\n", 3),
        ("", "\n", 0),
        ("\n", "\n", 1),
        ("a\n\n", "\n", 2),
        ("a||b||", "||", 2),
        ("||", "||", 1),
        ("a|||", "||", 2),
        ("aaa", "aa", 2),
        ("a\nb\nc", "\n", 3),
        ("你好\n世界\n", "\n", 2),
    ],
)
def test_replace_file_chunk_rejects_ranges_past_real_lines(tmp_path, original, separator, line_count):
    file_path = tmp_path / "sample.txt"
    original_bytes = original.encode("utf-8")
    file_path.write_bytes(original_bytes)
    tools = FileTools(base_dir=tmp_path, line_separator=separator)

    result = tools.replace_file_chunk("sample.txt", 0, line_count, "replacement")

    assert result == f"Error patching file: end_line must be less than {line_count}"
    assert file_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("original", "separator", "start", "end", "expected"),
    [
        ("a\nb\nc\n", "\n", 1, 2, "a\nreplacement\n"),
        ("a\nb\nc\n", "\n", 0, 2, "replacement\n"),
        ("\n", "\n", 0, 0, "replacement\n"),
        ("a\n\n", "\n", 1, 1, "a\nreplacement\n"),
        ("a\n\n\n", "\n", 0, 0, "replacement\n\n\n"),
        ("a||b||", "||", 1, 1, "a||replacement||"),
        ("a||b||||", "||", 1, 1, "a||replacement||||"),
        ("a\nb\nc", "\n", 1, 2, "a\nreplacement"),
        ("你好\n世界\n", "\n", 1, 1, "你好\nreplacement\n"),
        ("a\nb||c||", "||", 1, 1, "a\nb||replacement||"),
        ("a|||", "||", 1, 1, "a||replacement"),
        ("aaa", "aa", 1, 1, "aareplacement"),
    ],
)
def test_replace_file_chunk_preserves_real_line_boundaries(tmp_path, original, separator, start, end, expected):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(original.encode("utf-8"))
    tools = FileTools(base_dir=tmp_path, line_separator=separator)

    assert tools.replace_file_chunk("sample.txt", start, end, "replacement") == "sample.txt"
    assert file_path.read_bytes() == expected.encode("utf-8")


def test_save_and_read_file():
    """Test saving and reading a file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        # Save a file
        content = "Hello, World!"
        result = file_tools.save_file(contents=content, file_name="test.txt")
        assert result == "test.txt"

        # Read it back
        read_content = file_tools.read_file(file_name="test.txt")
        assert read_content == content


def test_list_files_returns_relative_paths():
    """Test that list_files returns relative paths, not absolute paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        # Create some test files
        (base_dir / "file1.txt").write_text("content1")
        (base_dir / "file2.txt").write_text("content2")
        (base_dir / "file3.md").write_text("content3")

        # List files
        result = file_tools.list_files()
        files = json.loads(result)

        # Verify we have 3 files
        assert len(files) == 3

        # Verify all paths are relative (not absolute)
        for file_path in files:
            assert not file_path.startswith("/")
            assert not file_path.startswith(tmp_dir)
            assert file_path in ["file1.txt", "file2.txt", "file3.md"]


def test_list_files_schema_exposes_directory():
    """Test that list_files exposes its optional directory argument."""
    file_tools = FileTools()
    function = file_tools.functions["list_files"]
    function.process_entrypoint()

    properties = function.parameters["properties"]
    assert properties["directory"]["type"] == "string"
    assert "kwargs" not in properties
    assert "directory" not in function.parameters.get("required", [])


def test_list_files_empty_directory_falls_back_to_base_dir():
    """Test that an empty directory string lists the base directory instead of returning nothing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        (base_dir / "file1.txt").write_text("content1")

        assert json.loads(file_tools.list_files(directory="")) == ["file1.txt"]
        assert file_tools.list_files(directory="") == file_tools.list_files()


def test_search_files_returns_relative_paths():
    """Test that search_files returns relative paths in JSON structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        # Create test files in nested directories
        (base_dir / "file1.txt").write_text("content1")
        (base_dir / "file2.md").write_text("content2")
        subdir = base_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        # Search for .txt files
        result = file_tools.search_files(pattern="*.txt")
        data = json.loads(result)

        # Verify JSON structure
        assert "pattern" in data
        assert "matches_found" in data
        assert "files" in data

        assert data["pattern"] == "*.txt"
        assert data["matches_found"] == 1
        assert len(data["files"]) == 1

        # Verify paths are relative (not absolute)
        for file_path in data["files"]:
            assert not file_path.startswith("/")
            assert not file_path.startswith(tmp_dir)
            assert file_path == "file1.txt"

        # Search with recursive pattern
        result = file_tools.search_files(pattern="**/*.txt")
        data = json.loads(result)

        assert data["matches_found"] == 2
        assert len(data["files"]) == 2

        # Verify all paths are relative
        for file_path in data["files"]:
            assert not file_path.startswith("/")
            assert not file_path.startswith(tmp_dir)

        assert "file1.txt" in data["files"]
        assert "subdir/file3.txt" in data["files"]


def test_search_files_does_not_return_traversal_matches_outside_base():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        base_dir = root / "base"
        outside_dir = root / "outside"
        base_dir.mkdir()
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("outside secret")
        file_tools = FileTools(base_dir=base_dir)

        result = json.loads(file_tools.search_files(pattern="../outside/*.txt"))

        assert result["matches_found"] == 0
        assert result["files"] == []


def test_save_and_delete_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        f = FileTools(base_dir=Path(tmpdirname), enable_delete_file=True)
        res = f.save_file(contents="contents", file_name="file.txt")
        assert res == "file.txt"
        contents = f.read_file(file_name="file.txt")
        assert contents == "contents"
        result = f.delete_file(file_name="file.txt")
        assert result == ""
        contents = f.read_file(file_name="file.txt")
        assert contents != "contents"


def test_read_file_chunk():
    """Test chunked file read"""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="line0\nline1\nline2\nline3\n", file_name="file1.txt")
        res = f.read_file_chunk(file_name="file1.txt", start_line=0, end_line=2)
        assert res == "line0\nline1\nline2"
        res = f.read_file_chunk(file_name="file1.txt", start_line=2, end_line=4)
        assert res == "line2\nline3\n"


def test_replace_file_chunk():
    """Test replace file chunk"""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="line0\nline1\nline2\nline3\n", file_name="file1.txt")
        res = f.replace_file_chunk(file_name="file1.txt", start_line=1, end_line=2, chunk="some\nstuff")
        assert res == "file1.txt"
        new_contents = f.read_file(file_name="file1.txt")
        assert new_contents == "line0\nsome\nstuff\nline3\n"


def test_check_path():
    """Test the path-containment check used by all file operations"""
    with tempfile.TemporaryDirectory() as tempdirname:
        base_dir = Path(tempdirname)
        f = FileTools(base_dir=base_dir)
        flag, path = f._check_path(".", base_dir)
        assert flag
        assert path.resolve() == base_dir.resolve()
        flag, path = f._check_path("..", base_dir)
        assert not (flag)
        flag, path = f._check_path("a/b/..", base_dir)
        assert flag
        assert path.resolve() == base_dir.joinpath(Path("a")).resolve()
        flag, path = f._check_path("a/b/../../..", base_dir)
        assert not (flag)


def test_search_content_finds_matches():
    """Test that search_content finds files containing the query string."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        (base_dir / "hello.txt").write_text("Hello World, this is a test file")
        (base_dir / "other.py").write_text("def greet():\n    print('hello')")
        (base_dir / "nope.txt").write_text("nothing relevant here")

        result = file_tools.search_content(query="hello")
        data = json.loads(result)

        assert data["query"] == "hello"
        assert data["matches_found"] == 2
        file_names = [m["file"] for m in data["files"]]
        assert "hello.txt" in file_names
        assert "other.py" in file_names


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks require admin on Windows")
def test_search_content_skips_symlink_targets_outside_base():
    """Test that search_content skips symlink targets outside base_dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        base_dir = root / "base"
        outside_dir = root / "outside"
        base_dir.mkdir()
        outside_dir.mkdir()
        secret = outside_dir / "secret.txt"
        secret.write_text("outside needle")
        try:
            (base_dir / "linked-secret.txt").symlink_to(secret)
        except OSError:
            pytest.skip("Symlink creation not permitted on this platform")
        file_tools = FileTools(base_dir=base_dir)

        result = json.loads(file_tools.search_content(query="needle"))

        assert result["matches_found"] == 0
        assert result["files"] == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks require admin on Windows")
def test_list_files_skips_symlink_targets_outside_base():
    """Test that list_files skips symlink targets outside base_dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        base_dir = root / "base"
        outside_dir = root / "outside"
        base_dir.mkdir()
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("outside secret")
        (base_dir / "inside.txt").write_text("inside content")
        try:
            (base_dir / "linked-secret.txt").symlink_to(outside_dir / "secret.txt")
        except OSError:
            pytest.skip("Symlink creation not permitted on this platform")
        file_tools = FileTools(base_dir=base_dir)

        files = json.loads(file_tools.list_files())

        assert "inside.txt" in files
        assert "linked-secret.txt" not in files


def test_search_content_directory_scoping():
    """Test that search_content respects directory scoping."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        (base_dir / "root.txt").write_text("target text here")
        subdir = base_dir / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("target text also here")

        result = file_tools.search_content(query="target", directory="sub")
        data = json.loads(result)

        assert data["matches_found"] == 1
        assert data["files"][0]["file"] == "sub/nested.txt"


def test_search_content_no_matches():
    """Test that search_content returns zero matches when query is not found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        (base_dir / "file.txt").write_text("some content")

        result = file_tools.search_content(query="nonexistent_string_xyz")
        data = json.loads(result)

        assert data["matches_found"] == 0
        assert data["files"] == []


def test_search_content_limit():
    """Test that search_content respects the limit parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        for i in range(5):
            (base_dir / f"file{i}.txt").write_text("common string in all files")

        result = file_tools.search_content(query="common string", limit=2)
        data = json.loads(result)

        assert data["matches_found"] == 2


def test_default_exclude_patterns_hide_junk():
    """By default, .venv and similar noise dirs are excluded from results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        tmp_sub = base_dir / "tmp"
        tmp_sub.mkdir()
        (tmp_sub / "foo.txt").write_text("foo")
        venv_pkg = tmp_sub / ".venv" / "lib" / "site-packages"
        venv_pkg.mkdir(parents=True)
        (venv_pkg / "x.py").write_text("print('x')")

        search_result = json.loads(file_tools.search_files(pattern="**/*.py"))
        assert search_result["matches_found"] == 0

        listed = json.loads(file_tools.list_files(directory="tmp"))
        assert ".venv" not in [Path(p).name for p in listed]
        assert "tmp/foo.txt" in listed


def test_empty_exclude_patterns_opts_out():
    """exclude_patterns=[] restores prior behavior: no exclusions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir, exclude_patterns=[])

        venv_pkg = base_dir / ".venv" / "lib" / "site-packages"
        venv_pkg.mkdir(parents=True)
        (venv_pkg / "x.py").write_text("print('x')")

        search_result = json.loads(file_tools.search_files(pattern="**/*.py"))
        assert search_result["matches_found"] == 1
        assert any(".venv" in p for p in search_result["files"])


def test_custom_exclude_patterns():
    """Custom patterns replace the default list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir, exclude_patterns=["node_modules"])

        (base_dir / "node_modules").mkdir()
        (base_dir / "node_modules" / "a.txt").write_text("a")
        (base_dir / ".venv").mkdir()
        (base_dir / ".venv" / "b.txt").write_text("b")

        result = json.loads(file_tools.search_files(pattern="**/*.txt"))
        files = result["files"]
        assert not any("node_modules" in p for p in files)
        assert any(".venv" in p for p in files)


def test_exclude_patterns_match_nested_components():
    """Patterns match on any path component, not just top-level dirs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir, exclude_patterns=[".git"])

        nested = base_dir / "vendor" / "thing" / ".git"
        nested.mkdir(parents=True)
        (nested / "config").write_text("[core]")
        (base_dir / "vendor" / "thing" / "readme.txt").write_text("hi")

        result = json.loads(file_tools.search_files(pattern="**/*"))
        files = result["files"]
        assert not any(".git" in Path(p).parts for p in files)
        assert "vendor/thing/readme.txt" in files


def test_exclude_patterns_support_globs():
    """Glob patterns like '*.egg-info' are honored."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir, exclude_patterns=["*.egg-info"])

        egg = base_dir / "foo.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text("Name: foo")
        (base_dir / "keep.txt").write_text("keep")

        result = json.loads(file_tools.search_files(pattern="**/*"))
        files = result["files"]
        assert not any("egg-info" in p for p in files)
        assert "keep.txt" in files


def test_default_excludes_env_family():
    """The '.env*' and '*.env' default patterns together hide both prefix
    (.env.local) and suffix (local.env) naming conventions for env files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        # Prefix convention (Next.js / Vite / most modern web frameworks)
        (base_dir / ".env").write_text("SECRET=1")
        (base_dir / ".env.local").write_text("SECRET=2")
        (base_dir / ".env.production").write_text("SECRET=3")
        (base_dir / ".envrc").write_text("export FOO=bar")

        # Suffix convention (Docker Compose --env-file, shell scripts)
        (base_dir / "local.env").write_text("SECRET=4")
        (base_dir / "prod.env").write_text("SECRET=5")

        # Real code that happens to contain "env" - must stay visible
        (base_dir / "environment.py").write_text("import os")
        (base_dir / "env.yaml").write_text("key: value")
        (base_dir / "keep.txt").write_text("visible")

        listed = sorted(json.loads(file_tools.list_files()))
        assert listed == ["env.yaml", "environment.py", "keep.txt"]


def test_default_excludes_credential_files():
    """The default list is shared with Workspace, where it is an access boundary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        for name in ("id_rsa", "server.pem", "credentials.json", "secrets.yaml", ".npmrc", "terraform.tfvars"):
            (base_dir / name).write_text("SECRET=1")
        # Ordinary source with a credential-shaped name stays visible: the patterns are
        # deliberately not `credentials.*` / `secrets.*`.
        for name in ("credentials.py", "secrets.py", "key.py", "keep.txt"):
            (base_dir / name).write_text("import os")

        listed = sorted(json.loads(file_tools.list_files()))
        assert listed == ["credentials.py", "keep.txt", "key.py", "secrets.py"]


def test_default_excludes_keep_committed_env_templates_visible():
    """A '!' entry exempts a component from every exclude pattern."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        (base_dir / ".env").write_text("SECRET=1")
        (base_dir / ".env.production").write_text("SECRET=2")
        (base_dir / ".env.example").write_text("OPENAI_API_KEY=")
        (base_dir / ".env.sample").write_text("OPENAI_API_KEY=")
        (base_dir / "example.env").write_text("OPENAI_API_KEY=")

        listed = sorted(json.loads(file_tools.list_files()))
        assert listed == [".env.example", ".env.sample", "example.env"]


def test_default_excludes_agent_scratch_and_plural_venvs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        (base_dir / "real.py").write_text("# TODO: real work")
        scratch = base_dir / ".context"
        scratch.mkdir()
        (scratch / "notes.py").write_text("# TODO: scratch")
        venvs_pkg = base_dir / ".venvs" / "demo" / "lib"
        venvs_pkg.mkdir(parents=True)
        (venvs_pkg / "installed.py").write_text("# TODO: dependency")

        result = json.loads(file_tools.search_content(query="TODO", limit=10))
        file_names = [m["file"] for m in result["files"]]
        assert result["matches_found"] == 1
        assert file_names == ["real.py"]


def test_search_content_honors_exclusions():
    """search_content should not return matches inside excluded directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        file_tools = FileTools(base_dir=base_dir)

        venv_pkg = base_dir / ".venv" / "lib"
        venv_pkg.mkdir(parents=True)
        (venv_pkg / "hit.py").write_text("# TODO: something")
        (base_dir / "real.py").write_text("# TODO: real work")

        result = json.loads(file_tools.search_content(query="TODO"))
        file_names = [m["file"] for m in result["files"]]
        assert result["matches_found"] == 1
        assert "real.py" in file_names
        assert not any(".venv" in f for f in file_names)


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


@pytest.fixture(params=["python", "ripgrep"])
def grep_tools(request, tmp_path):
    if request.param == "ripgrep" and shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    (tmp_path / "a.py").write_text("def alpha():\n    pass\n\nclass Beta:\n    # TODO high priority\n    pass\n")
    (tmp_path / "b.txt").write_text("nothing here\nDEF upper\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("def gamma():\n    return 1\n")
    (tmp_path / "sub" / "d.proto").write_text("message Foo {\n  string def_name = 1;\n}\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "hidden.py").write_text("def hidden():\n    pass\n")
    (tmp_path / ".env").write_text("def secret\n")
    (tmp_path / ".env.example").write_text("def placeholder\n")
    (tmp_path / "blob.bin").write_bytes(b"def\x00binary")
    tools = FileTools(base_dir=tmp_path)
    if request.param == "python":
        tools._rg_path = None
    else:
        assert tools._rg_path
    return tools


def _grep(tools, pattern, **kwargs):
    return json.loads(tools.grep(pattern, **kwargs))


def test_grep_registered_by_default_and_flag(tmp_path):
    assert "grep" in FileTools(base_dir=tmp_path).functions
    assert "grep" not in FileTools(base_dir=tmp_path, enable_grep=False).functions
    assert "grep" in FileTools(base_dir=tmp_path, enable_grep=False, all=True).functions


def test_grep_files_with_matches_is_default_and_regex(grep_tools):
    result = _grep(grep_tools, r"def \w+\(")
    assert result["output_mode"] == "files_with_matches"
    assert result["truncated"] is False
    assert [h["file"] for h in result["results"]] == ["a.py", "sub/c.py"]


def test_grep_skips_excluded_binary_and_keeps_exempt_files(grep_tools):
    files = [h["file"] for h in _grep(grep_tools, "def")["results"]]
    assert ".venv/hidden.py" not in files
    assert ".env" not in files
    assert "blob.bin" not in files
    assert ".env.example" in files
    assert "sub/d.proto" in files


def test_grep_ignore_case(grep_tools):
    assert [h["file"] for h in _grep(grep_tools, "def upper")["results"]] == []
    result = _grep(grep_tools, "def upper", ignore_case=True)
    assert [h["file"] for h in result["results"]] == ["b.txt"]


def test_grep_include_glob(grep_tools):
    assert [h["file"] for h in _grep(grep_tools, "def", include="*.proto")["results"]] == ["sub/d.proto"]
    assert [h["file"] for h in _grep(grep_tools, "def", include="**/*.py")["results"]] == ["a.py", "sub/c.py"]
    assert [h["file"] for h in _grep(grep_tools, "def", include="sub/*.py")["results"]] == ["sub/c.py"]


def test_grep_path_scoping_file_and_directory(grep_tools):
    assert [h["file"] for h in _grep(grep_tools, "def", path="sub", include="*.py")["results"]] == ["sub/c.py"]
    assert [h["file"] for h in _grep(grep_tools, "def", path="a.py")["results"]] == ["a.py"]


def test_grep_path_outside_base_dir_is_rejected(grep_tools):
    assert grep_tools.grep("def", path="../").startswith("Error")
    assert grep_tools.grep("def", path="/etc").startswith("Error")


def test_grep_content_mode(grep_tools):
    result = _grep(grep_tools, r"class [A-Z]", output_mode="content")
    assert result["results"] == [{"file": "a.py", "line": 4, "text": "class Beta:", "is_match": True}]


def test_grep_content_mode_with_context(grep_tools):
    result = _grep(grep_tools, "TODO", path="a.py", output_mode="content", context=1)
    assert [(h["line"], h["is_match"]) for h in result["results"]] == [(4, False), (5, True), (6, False)]


def test_grep_count_mode(grep_tools):
    result = _grep(grep_tools, "def", output_mode="count", ignore_case=True)
    counts = {h["file"]: h["count"] for h in result["results"]}
    assert counts["a.py"] == 1
    assert counts["b.txt"] == 1
    assert counts["sub/d.proto"] == 1


def test_grep_multiline(grep_tools):
    assert _grep(grep_tools, r"alpha\(\):.pass")["results"] == []
    result = _grep(grep_tools, r"alpha\(\):.*?pass", output_mode="content", multiline=True)
    assert [(h["line"], h["text"]) for h in result["results"]] == [(1, "def alpha():"), (2, "    pass")]


def test_grep_limit_truncates(grep_tools):
    result = _grep(grep_tools, "def", limit=1)
    assert len(result["results"]) == 1
    assert result["truncated"] is True

    result = _grep(grep_tools, "def", output_mode="content", limit=1)
    assert len([h for h in result["results"] if h["is_match"]]) == 1
    assert result["truncated"] is True

    result = _grep(grep_tools, "def", output_mode="count", limit=1)
    assert len(result["results"]) == 1
    assert result["truncated"] is True


def test_grep_limit_is_capped(tmp_path):
    for i in range(GREP_MAX_LIMIT + 5):
        (tmp_path / f"f{i:04d}.txt").write_text("hit\n")
    tools = FileTools(base_dir=tmp_path)
    tools._rg_path = None
    result = _grep(tools, "hit", limit=10**6)
    assert len(result["results"]) == GREP_MAX_LIMIT
    assert result["truncated"] is True


def test_grep_no_match_and_errors(grep_tools):
    assert _grep(grep_tools, "zzz_not_present") == {
        "pattern": "zzz_not_present",
        "output_mode": "files_with_matches",
        "results": [],
        "truncated": False,
    }
    assert grep_tools.grep("").startswith("Error")
    assert grep_tools.grep("(unclosed").startswith("Error")
    assert grep_tools.grep("def", path="missing").startswith("Error")
    assert grep_tools.grep("def", output_mode="bogus").startswith("Error")


def test_grep_pattern_starting_with_dash(grep_tools):
    (grep_tools.base_dir / "flags.txt").write_text("--verbose\n")
    assert [h["file"] for h in _grep(grep_tools, "--verbose")["results"]] == ["flags.txt"]


def test_grep_backends_return_identical_results(tmp_path):
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    (tmp_path / "a.py").write_text("one\ntwo target\nthree\nfour\ntarget five\nsix\n")
    (tmp_path / "b.py").write_text("target\n")
    rg_tools = FileTools(base_dir=tmp_path)
    py_tools = FileTools(base_dir=tmp_path)
    py_tools._rg_path = None
    for kwargs in (
        {},
        {"output_mode": "count"},
        {"output_mode": "content"},
        {"output_mode": "content", "context": 1},
        {"output_mode": "content", "context": 2, "limit": 2},
        {"output_mode": "content", "limit": 1},
    ):
        assert rg_tools.grep("target", **kwargs) == py_tools.grep("target", **kwargs), kwargs
