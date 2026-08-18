---
title: Linux Commands - 2
date: 2026-06-05
lastmod: 2026-06-05
description: CIL notes
tags:
  - CIL_commands
  - Notes
categories:
  - Learning
draft: false
---
## Environment Varibles

In linux we can create a variables in our shell:

```bash
name="co"
number=130
command=ls
echo $name
# co
echo $number
# 130
echo $command
# ls
```

>as you see we can add a command as variable to use this:

```bash
$command
# foo.txt foo1.txt foo2.txt
```

these are the local variables if you want to set a variable for your **current shell session**, use the `export` command.

```bash
export NAME="Co"
```

## PATH

> One of my educational source says that THIS IS ONE OF THE MOST IMPORTANT LESSONS

If it weren't for the `PATH`, you'd have to remember the filesystem path of every executable you wanted to run in your shell. Instead of just running `ls`, you'd have to run `/bin/ls` (or whatever the location of the `ls` executable is on your system). That's not very convenient.

The `PATH` variable is a list of directories that your shell will look into when you try to run a command. If you type `ls`, your shell will look in each directory listed in your `PATH` variable for an executable called `ls`. If it finds one, it will just run it. If it doesn't, it will give you an error like: "command not found."

```bash
echo $PATH
```
```bash
/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/var/lib/flatpak/exports/bin:/usr  
/bin/site_perl:/usr/bin/vendor_perl:/usr/bin/core_perl
```

It listed the executables direction on the above.

---

When you try to run it from your terminal in the future for installed a new program on your machine you can get a error command not found because of the program is installed in a directory that's not in your `PATH` variable. And you can add a directory to your `PATH` without overwriting all of the existing directories by using that command example:
```bash
export PATH="$PATH:/relativepath"
```

> [!WARNING]
> You need the use the relative path. And this command will be reset to its default value, when you restart your shell. If you want to stays that `PATH` variable permanently you need to add that command in your `.zshrc / .bashrc `

## Exit Codes
Exit codes simply called "return codes" which is the how programms communicate back whether they ran successfully or not.

0 is the exit code for success.Any other exit code is an error.

You can access the exit code of the last program you ran with the question mark variable `$?`.

```bash
echo $?
# 0 or non-zero
```

Its useful when the you use programs to see worked or not for debugging.


## Standart Error
Usually called `stderr`, is a data stream just like standart output.

You can redirect `stdout` and `stderr` to different places using the `>`(stdout) and `2>`(stderr) operators.

```bash
echo "Hello World" > hello.txt
cat hello.txt
# Hello Worll
```

```bash
cat foo.txt 2> error.txt
cat error.txt
# cat : foo.txt : No such file or directory
```


## Piping

One of the most beautiful things about the shell is that you can pipe the output of one program into the input of another program. With this one simple concept, you can run incredibly powerful automation tasks. Pipe operator is `|`.

```bash
echo "balbla" | read NAME
echo $NAME
# blabla

# -----------
grep "hello" somefile.txt | less
```

## Kill
Sometimes a program does not respond to the `SIGINT`(Which is the key `Ctrl + C`). In that case the best option is to manually kill the program in another terminal session.

```bash
kill <PID>
# for PID, use the command below
ps aux # "process starus command" list the processes running on your machine
```

I saw the efficient way to use `ps aux` command with Piping:

```bash
ps aux | grep foo.sh # foo.sh is the which is the running program in your current terminal state
```

