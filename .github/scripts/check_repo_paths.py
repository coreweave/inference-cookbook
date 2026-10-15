# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Fail when a file moves out from under something that names its path.

- Every path in .github/externally-linked-paths.txt exists. Pages outside the repository link
  them, and GitHub does not redirect a moved path.
- Every relative link in a Markdown file, and every link to this repository's main branch,
  resolves to a file or directory in the repository.
- The filled-in copy of every template (*.env.example -> *.env, *.example.json -> *.run.json) is
  git-ignored in every directory that holds a template, and git ignores no tracked file.
- Every path in a REUSE.toml annotation matches a file, and no patch (*.patch, *.diff) falls back
  to the default annotation. A patch carries context lines from the project it modifies, so its
  annotation also names that project's copyright holders. A file that moves away from its
  annotation silently gets the default one.

The checks see the files that git tracks or would add (untracked but not ignored) as they are on
disk, so they also work before a commit.
"""

from __future__ import annotations

import os
import posixpath
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

LINKED = ".github/externally-linked-paths.txt"
# The name of a template's filled-in copy, by template suffix. The READMEs tell users to create
# run.env, deployment.run.json and bucket-policy.run.json, and that git ignores them.
FILLED_IN = ((".env.example", ".env"), (".example.json", ".run.json"))
PATCHES = (".patch", ".diff")

FENCE = re.compile(r"\s*(`{3,}|~{3,})(.*)")
CODE_SPAN = re.compile(r"(`+)(?!`)(?:(?!\n\s*\n).)+?(?<!`)\1(?!`)", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
# [text](target) and ![alt](target), not \[text\](target); a target may hold balanced parentheses
INLINE_LINK = re.compile(r"(?<!\\)\]\(\s*(<[^>\n]*>|(?:[^()\s]|\([^()\s]*\))+)")
# [label]: target, alone on its line or followed by a title; anything else after it is prose
LINK_DEFINITION = re.compile(
    r"^ {0,3}\[(?!\^)[^\]\n]+\]:[ \t]*(<[^>\n]*>|\S+)(?=[ \t]*$|[ \t]+[\"'(])", re.M
)
MAIN_URL = re.compile(
    r"https?://github\.com/coreweave/inference-cookbook/(?:blob|tree)/main/([^\s)\"'<>`\]]+)"
)
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def git(root: Path, *args: str, stdin: str = "") -> str:
    # Only the repository's own ignore rules count, not the user's global ones.
    cmd = ["git", "-c", f"core.excludesFile={os.devnull}", *args]
    result = subprocess.run(cmd, cwd=root, input=stdin, capture_output=True, text=True, check=False)
    if result.returncode > 1:  # git check-ignore exits 1 when it ignores none of the paths
        raise SystemExit(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout


def repo_files(root: Path) -> set[str]:
    out = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return {path for path in out.split("\0") if path and (root / path).is_file()}


def with_directories(files: set[str]) -> set[str]:
    paths = set(files) | {"."}
    for path in files:
        while (path := posixpath.dirname(path)) and path not in paths:
            paths.add(path)
    return paths


def linked_paths(root: Path, files: set[str]) -> tuple[list[str], str]:
    lines = (root / LINKED).read_text(encoding="utf-8").splitlines()
    listed = [line.strip().rstrip("/") for line in lines if line.strip()[:1] not in ("", "#")]
    known = with_directories(files)
    problems = [
        f"{path} is missing, but pages outside this repository link it, and GitHub does not "
        "redirect a moved path. Update those links first: keep the path (or a copy) until they "
        f"point to its new location. Then remove its line from {LINKED} in the pull request that "
        "removes it."
        for path in listed
        if path not in known
    ]
    return problems, f"{len(listed)} paths linked from outside the repository exist"


def newlines(match: re.Match[str]) -> str:
    return "\n" * match[0].count("\n")


def prose(text: str) -> str:
    """The Markdown outside fenced code, code spans and comments, with its line breaks kept."""
    lines, fence = [], ""
    for line in text.split("\n"):
        match = FENCE.match(line)
        if fence:
            # only a bare fence of the same kind, at least as long, closes the block
            if match and match[1].startswith(fence) and not match[2].strip():
                fence = ""
            line = ""
        elif match and not (match[1][0] == "`" and "`" in match[2]):  # ```x``` is a code span
            fence, line = match[1], ""
        lines.append(line)
    return COMMENT.sub(newlines, CODE_SPAN.sub(newlines, "\n".join(lines)))


def link_path(source: str, target: str) -> str | None:
    """The repository path that a relative link in source points at; None for any other link."""
    target = re.split(r"[#?]", target, maxsplit=1)[0]
    if not target or SCHEME.match(target) or target.startswith("//"):
        return None
    base = "" if target.startswith("/") else posixpath.dirname(source)
    return posixpath.normpath(posixpath.join(base, unquote(target).lstrip("/")))


def main_url_path(url_path: str) -> str:
    # a bare URL may end in punctuation or emphasis that is not part of it, as GitHub renders it
    url_path = re.split(r"[#?]", url_path.rstrip(".,;:!?*_~"), maxsplit=1)[0]
    return posixpath.normpath(unquote(url_path))


def markdown_links(root: Path, files: set[str]) -> tuple[list[str], str]:
    known = with_directories(files)
    problems, checked = [], 0
    markdown = sorted(path for path in files if path.endswith(".md"))
    for source in markdown:
        text = prose((root / source).read_text(encoding="utf-8"))
        links = [
            (match.start(), target, link_path(source, target))
            for pattern in (INLINE_LINK, LINK_DEFINITION)
            for match in pattern.finditer(text)
            for target in [match[1].strip("<>")]
        ]
        links += [(m.start(), m[0], main_url_path(m[1])) for m in MAIN_URL.finditer(text)]
        for start, target, path in sorted(links, key=lambda link: link[0]):
            if path is None:
                continue
            checked += 1
            if path not in known:
                line = text.count("\n", 0, start) + 1
                outside = path.split("/")[0] == ".."
                where = "is outside the repository" if outside else "does not exist"
                problems.append(f"{source}:{line}: {target} links to {path}, which {where}")
    summary = f"{checked} relative and main-branch links in {len(markdown)} Markdown files resolve"
    return problems, summary


def filled_in_copies(root: Path, files: set[str]) -> tuple[list[str], str]:
    problems, copies = [], {}
    templates = sorted(path for path in files if ".example" in posixpath.basename(path))
    for template in templates:
        name = posixpath.basename(template)
        for suffix, copy_suffix in FILLED_IN:
            if name.endswith(suffix):
                copies.setdefault(name.removesuffix(suffix) + copy_suffix, name)
                break
        else:
            problems.append(
                f"{template} looks like a template, but its filled-in copy has no known name. "
                "Name the template *.example.json or *.env.example, or add its suffix to "
                "FILLED_IN in .github/scripts/check_repo_paths.py and a pattern to .gitignore."
            )
    # Users put a copy next to its template or in the kit (k8s/) that reads it, so every name
    # must be ignored in every directory that holds a template.
    dirs = sorted({posixpath.dirname(template) for template in templates})
    wanted = [posixpath.join(d, copy) for d in dirs for copy in copies]
    out = git(root, "check-ignore", "--no-index", "-z", "--stdin", stdin="\0".join(wanted))
    ignored = set(out.split("\0"))
    for copy, name in sorted(copies.items()):
        missing = [d or "." for d in dirs if posixpath.join(d, copy) not in ignored]
        if missing:
            where = ", ".join(missing)
            if len(missing) == len(dirs):
                where = f"any of the {len(dirs)} directories that hold templates"
            problems.append(
                f"{copy}, the filled-in copy of {name}, is not git-ignored in {where}. Users put "
                "it next to its template or in the kit that reads it, and it holds deployment "
                f"ids, endpoints and bucket names. Make .gitignore match {copy} in any directory."
            )
    out = git(root, "ls-files", "-z", "--cached", "--ignored", "--exclude-standard")
    problems += [
        f"{path} is tracked, but .gitignore ignores it. If it is a filled-in copy, remove it with "
        "git rm --cached and treat what it holds as public. If it belongs in the repository, "
        f"exempt it with a line such as !{posixpath.basename(path)} below the pattern that "
        f"ignores it, which git check-ignore -v --no-index {path} shows."
        for path in out.split("\0")
        if path
    ]
    summary = (
        f"filled-in copies of {len(templates)} templates are git-ignored in {len(dirs)} "
        "directories, and no tracked file is git-ignored"
    )
    return problems, summary


def reuse_glob(path: str) -> re.Pattern[str]:
    """REUSE.toml globbing: * matches within a directory, ** and **/ across directories."""
    regex, i = "", 0
    while i < len(path):
        if path[i] == "\\" and i + 1 < len(path):  # "\\*" in TOML is a literal *
            regex, i = regex + re.escape(path[i + 1]), i + 2
        elif path.startswith("**", i):
            regex, i = regex + ".*", i + (3 if path.startswith("**/", i) else 2)
        elif path[i] == "*":
            regex, i = regex + "[^/]*", i + 1
        else:
            regex, i = regex + re.escape(path[i]), i + 1
    return re.compile(regex)


def reuse_annotations(root: Path, files: set[str]) -> tuple[list[str], str]:
    problems, checked, patches = [], 0, {}
    # A file gets the closest REUSE.toml that matches it, and in it the last annotation that does.
    tomls = (path for path in files if posixpath.basename(path) == "REUSE.toml")
    for toml in sorted(tomls, key=lambda path: (path.count("/"), path)):
        # paths in a REUSE.toml are relative to its directory
        prefix = toml.removesuffix("REUSE.toml")
        covered = [path.removeprefix(prefix) for path in files if path.startswith(prefix)]
        config = tomllib.loads((root / toml).read_text(encoding="utf-8"))
        for annotation in config.get("annotations", []):
            paths = annotation.get("path", [])
            paths = [paths] if isinstance(paths, str) else paths
            owners = annotation.get("SPDX-FileCopyrightText", [])
            owners = "; ".join([owners] if isinstance(owners, str) else owners)
            matched = set()
            for path in paths:
                checked += 1
                glob = reuse_glob(path)
                found = {file for file in covered if glob.fullmatch(file)}
                if not found:
                    problems.append(
                        f"{toml}: no file matches path {path!r} ({owners}). If its files moved, "
                        "update the path in the same pull request; otherwise they fall back to "
                        "the default annotation."
                    )
                matched |= found
            for file in matched:
                if file.endswith(PATCHES):
                    patches[prefix + file] = (toml, owners, "**" in paths)
    for path, (toml, owners, default) in sorted(patches.items()):
        if default:
            problems.append(
                f"{toml}: {path} falls back to the default annotation ({owners}). A patch carries "
                "context lines from the project it modifies, so match it with an annotation that "
                "also names that project's copyright holders, below the default one."
            )
    summary = (
        f"{checked} REUSE.toml annotation paths each match a file, and none of the "
        f"{len(patches)} patches falls back to the default annotation"
    )
    return problems, summary


def main(root: Path) -> int:
    files = repo_files(root)
    failed = False
    for check in (linked_paths, markdown_links, filled_in_copies, reuse_annotations):
        problems, summary = check(root, files)
        print("\n".join(problems) if problems else summary)
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parents[2]))
