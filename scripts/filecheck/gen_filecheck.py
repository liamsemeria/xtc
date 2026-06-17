#!/usr/bin/env python3

import subprocess
import re
import os
import sys

INDENT = 7


def process_file(
    file_path: str,
    in_place: bool,
    regexp: bool,
    diff: bool,
    quiet: bool = False,
    check_exit: bool = False,
) -> bool:
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Handle any comment kind, generally '//' or '#'
    re_prefix = r"^([^ ]+) +RUN:"
    m = re.match(re_prefix, lines[0])
    if not m:
        if not quiet:
            print(f"Error: The first line of the file does not start with '... RUN:'")
        return False

    # prefix contains the comment prefix
    prefix = m.group(1)

    # Extract the command from the first line
    first_line = lines[0].strip()
    command = re.sub(re_prefix, "", first_line).strip()
    command = command.rsplit("|", 1)[0].strip()
    command = re.sub(r"%s\b", file_path, command)
    command = re.sub(r"^not\b", "!", command)  # use ! instead of not

    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if check_exit and result.returncode != 0:
        print(
            f"Error: command failed (exit {result.returncode}) for: {file_path}",
            file=sys.stderr,
        )
        return False

    output = result.stdout + result.stderr

    # Remove trailing newline if present
    if output.endswith("\n"):
        output = output[:-1]

    # Process the output
    output_lines = output.splitlines()
    processed_output = f"{prefix} CHECK:" + INDENT * " " + output_lines[0] + "\n"
    for line in output_lines[1:]:
        processed_output += f"{prefix} CHECK-NEXT:" + (INDENT - 5) * " " + line + "\n"

    if in_place or diff:
        # Remove lines starting with '... CHECK:' or '... CHECK-NEXT:'
        new_lines = [
            line
            for line in lines
            if (
                not line.startswith(f"{prefix} CHECK:")
                and not line.startswith(f"{prefix} CHECK-NEXT:")
            )
        ]
        new_lines.append(processed_output)

        tmp_file = f"{file_path}.tmp"
        with open(tmp_file, "w") as file:
            file.writelines(new_lines)
        if diff:
            try:
                subprocess.run(
                    f"diff -u {file_path} {tmp_file}", shell=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"ERROR: output differs for: {file_path}", file=sys.stderr)
            finally:
                os.remove(tmp_file)
        else:
            os.rename(tmp_file, file_path)
    else:
        print(processed_output)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Filecheck directives.",
        epilog="Apply to a directory: python3 gen_filecheck.py <dir> -i",
    )
    parser.add_argument("src", type=str, help="Source file or directory.")
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Insert the resulting Filecheck directives.",
    )
    parser.add_argument(
        "-r",
        "--regexp",
        action="store_true",
        help="Replace MLIR/LLVM variables by regexps (ignored).",
    )
    parser.add_argument(
        "-d", "--diff", action="store_true", help="Output diff instead of generating."
    )
    parser.add_argument(
        "-e",
        "--check-exit",
        action="store_true",
        help="Abort if the RUN command exits with a non-zero status.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.src):
        for name in sorted(os.listdir(args.src)):
            fpath = os.path.join(args.src, name)
            if os.path.isfile(fpath):
                process_file(
                    fpath,
                    args.in_place,
                    args.regexp,
                    args.diff,
                    quiet=True,
                    check_exit=args.check_exit,
                )
    else:
        process_file(
            args.src, args.in_place, args.regexp, args.diff, check_exit=args.check_exit
        )
