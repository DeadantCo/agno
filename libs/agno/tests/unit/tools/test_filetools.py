import json
import sys
import tempfile
from pathlib import Path

import pytest

from agno.tools.file import FileTools


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


def test_edit_file_requires_read_before_edit():
    """edit_file refuses to touch a file that has never been fully read."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="existing", file_name="file.txt")
        # Bypass the save_file tracking to simulate a file the agent never observed.
        resolved = (Path(tempdirname) / "file.txt").resolve()
        f._observed_mtimes.pop(resolved, None)

        result = f.edit_file(file_name="file.txt", old_text="existing", new_text="new")

        assert "has not been read yet" in result
        assert f.read_file(file_name="file.txt") == "existing"


def test_edit_file_rejects_partial_read():
    """A read_file_chunk alone does not satisfy the read-before-edit guard."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="line0\nline1\n", file_name="file.txt")
        resolved = (Path(tempdirname) / "file.txt").resolve()
        f._observed_mtimes.pop(resolved, None)
        f.read_file_chunk(file_name="file.txt", start_line=0, end_line=0)

        result = f.edit_file(file_name="file.txt", old_text="line0", new_text="lineX")

        assert "has not been read yet" in result


def test_edit_file_detects_staleness():
    """edit_file refuses an edit if the file's mtime changed since the last read."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="original", file_name="file.txt")
        f.read_file(file_name="file.txt")

        # Simulate an external modification after the read observed the file.
        file_path = Path(tempdirname) / "file.txt"
        file_path.write_text("changed externally")
        import os

        os.utime(file_path, (os.path.getmtime(file_path) + 10, os.path.getmtime(file_path) + 10))

        result = f.edit_file(file_name="file.txt", old_text="original", new_text="new")

        assert "changed on disk" in result


def test_edit_file_unique_match_succeeds():
    """edit_file applies the change when old_text matches exactly once."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="def foo():\n    return 1\n", file_name="file.py")
        f.read_file(file_name="file.py")

        result = f.edit_file(file_name="file.py", old_text="return 1", new_text="return 2")

        assert result == "file.py"
        assert f.read_file(file_name="file.py") == "def foo():\n    return 2\n"


def test_edit_file_rejects_ambiguous_match():
    """edit_file refuses when old_text matches more than once and replace_all is False."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="x = 1\nx = 1\n", file_name="file.txt")
        f.read_file(file_name="file.txt")

        result = f.edit_file(file_name="file.txt", old_text="x = 1", new_text="x = 2")

        assert "matches 2 locations" in result
        assert f.read_file(file_name="file.txt") == "x = 1\nx = 1\n"


def test_edit_file_replace_all():
    """edit_file replaces every occurrence when replace_all=True."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="x = 1\nx = 1\n", file_name="file.txt")
        f.read_file(file_name="file.txt")

        result = f.edit_file(file_name="file.txt", old_text="x = 1", new_text="x = 2", replace_all=True)

        assert result == "file.txt"
        assert f.read_file(file_name="file.txt") == "x = 2\nx = 2\n"


def test_edit_file_creates_missing_file_with_empty_old_text():
    """edit_file with old_text="" creates a new file, matching Claude Code's Edit semantics."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))

        result = f.edit_file(file_name="new_file.txt", old_text="", new_text="hello world")

        assert result == "new_file.txt"
        assert f.read_file(file_name="new_file.txt") == "hello world"


def test_edit_file_rejects_identical_old_and_new_text():
    """edit_file returns a no-op error instead of silently writing when old_text == new_text."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))
        f.save_file(contents="same", file_name="file.txt")
        f.read_file(file_name="file.txt")

        result = f.edit_file(file_name="file.txt", old_text="same", new_text="same")

        assert "No changes needed" in result


def test_edit_file_missing_file_without_empty_old_text():
    """edit_file on a nonexistent file with non-empty old_text returns an error, not a creation."""
    with tempfile.TemporaryDirectory() as tempdirname:
        f = FileTools(base_dir=Path(tempdirname))

        result = f.edit_file(file_name="missing.txt", old_text="something", new_text="else")

        assert "does not exist" in result


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
