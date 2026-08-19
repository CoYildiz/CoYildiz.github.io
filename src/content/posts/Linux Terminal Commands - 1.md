---
title: Linux Commands - 1
published: 2026-06-02
description: CIL notes
tags:
  - CIL_commands
  - Notes
category: Learning
draft: false
lang: en
---

## Overview

These notes are written primarily for myself. They are not meant to be a complete guide, but rather a personal cheatsheet that I can quickly refer back to when needed.

> I think this notes for users who are know little bit scripts.

I am sharing them in case they are useful for others as well.

I also rewrote this post by a LLM to fix English mistakes and improve readability. I checked it quickly, so there may still be minor issues.

---

## Head and Tail
The `head` command prints the first *n* lines of a file. The `-n` flag defines the number of lines to display.

```bash
head -n 10 foo.txt
```

The `tail` command prints the last _n_ lines of a file.

```bash
tail -n 10 foo.txt
```

---

## More and Less
I usually use the `less` command because it provides all the features of `more` and adds additional functionality.

```bash
cat -n foo.csv | less
```

> Inside `less`, you can:
>
> - Press SPACE to move forward
> - Press `b` to move backward
> - Press `q` to quit

---

## Remove Files and Directories
To remove files or directories:

```bash
rm -r foo
```

The `-r` flag stands for recursive and is required for removing directories.

---

## Copy Files and Directories
To copy a file:

```bash
cp source_file.txt destination/
```

To copy a directory and all its contents, use the `-R` flag:

```bash
cp -R dir new_dir
```

---

## Symbolic Links
Symbolic links can be created using `ln -s`:

```bash
ln -s target_path link_path
ln -s documents/important.txt important.txt
```

You can use either relative or absolute paths depending on the situation.

To see where a symbolic link points, use:

```bash
ls -l
```

---

## Grep and Find
The `grep` command is case-sensitive by default, so it only matches exact patterns.
It is used to search text inside files:

```bash
grep "hello world" world.txt
grep -r "hello world" ~/path/dir/
```

The `find` command is used to search for files and directories by name:

```bash
find some_directory -name hello.txt
find some_directory -name "*.txt"
```

## Permissions
The permissions of an individual file or directory are visually represented as a 10-character string:
```
drwxrwxrwx
```
- `-`: Regular file (e.g. `-rwxrwxrwx`)
- `d`: Directory (e.g. `drwxrwxrwx`)

- The first 3 characters are "owner" permissions. The "owner" is usually just the user who created the file or directory, but it can be manually changed.
- The next 3 characters are "group" permissions. Unix-like systems support groups of users, and each file or directory is assigned to exactly one owning group. Anyone who is not the owner and not a member of that group falls under "others." To be honest, unless you're a system administrator, you won't often worry about groups.
- The last 3 characters are "others" permissions. This is everyone else.

When you're doing programming work on your own local machine, you mostly just care about the "owner" permissions because that's usually you. Here are some full examples:

- `-rwxrwxrwx`: A file where everyone can do everything
- `-rwxr-xr-x`: A file where everyone can read and execute, but only the owner can write
- `drwxr-xr-x`: A directory where everyone can read (`ls` the contents) and execute (`cd` into it), but only the owner can write (modify the contents)
- `drwx------`: A directory where only the owner can read, write and execute

> We can change the permissions by using `chmod`:
```bash
chmod -R u=rwx,g=,o= DIRECTORY
```
- `u` means "user", `g` means "group" , `o` means "others".
> or we can use like this:
```bash
chmod u+x foo.sh # we add a user can execute the file
# ---
chmod -x foo.sh # we use the -x flag that simply remove the executable permission
```

> The `chown` command allows you to change the owner of a file or directory, and it requires root privileges.
```bash
sudo chown -R root dir
```
Here's an explanation of the command:
- `sudo` – Run as the root user
- `chown` – Command to change the owner
- `-R` – "Recursive," meaning also apply the changes to everything inside the directory
- `root` – The name of the new owner
- `dir` – The directory to change the owner of

> [!MORE]
> `shebang`, what is it?

## Summary
I think it is enough for now. I am currently learning the bash scripts with a project. These are the most useful commands that i need to remember. I will add more command later.

> Thanks
