# Lab: Deep-Dive into Google Colab

> [!NOTE]  
> - We will use [Google Colab](https://colab.research.google.com/) for browser-based Python notebooks and access to cloud compute resources.
> - We will also use [Google AI Studio](https://aistudio.google.com/) for free API experiments, so please register with a **personal** Gmail account. Otherwise, you may not be able to obtain the API key needed later in the course.
> - If you have a dedicated GPU and are interested in a local setup using VS Code, please send me an email.

----

## Overview

Google Colab is more than an online Python editor. A Colab notebook combines several components:

* a **notebook interface** containing Markdown and code cells;
* an **IPython kernel** that executes Python code;
* a temporary **Linux virtual machine (VM)** that provides the operating system, CPU, memory, storage, and possibly a GPU;
* IPython **magics**, such as `%cd`, `%timeit`, and `%run`;
* ordinary Linux commands, which can be executed with `!`;
* optional persistent storage through **Google Drive**.

In this lab, you will progressively investigate how these components interact.

You will learn to:

1. understand the structure of a Colab notebook;
2. distinguish Markdown cells, Python cells, IPython magics, and shell commands;
3. understand the difference between `!` and `%`;
4. investigate processes, environment variables, working directories, and files;
5. create and execute Python code from notebook cells;
6. profile and benchmark Python code;
7. use a GPU and correctly measure asynchronous CUDA computation;
8. document experimental results using Markdown and $\LaTeX$;
9. distinguish temporary VM storage from persistent Google Drive storage.

---

# Task 0 - Understanding the Colab Notebook

Before running commands, become familiar with what a Colab notebook actually contains.

## 0.1 Notebook cells

A Colab notebook is made up of **cells**. The two most important cell types are:

### Markdown cells

Markdown cells contain documentation rather than executable Python code.

For example:

```markdown
# My First Colab Experiment

This notebook compares CPU and GPU computation.

The formula for matrix multiplication is:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$
```

When the cell is rendered, the Markdown becomes formatted text and the equation is displayed using $\LaTeX$.

### Code cells

Code cells are sent to the active Python kernel for execution.

For example:

```python
x = 10
print(x)
```

The output is displayed below the cell.

---

## 0.2 Code cells are stateful

Run:

```python
x = 100
```

Then run the following in a **different cell**:

```python
print(x)
```

You should obtain:

```text
100
```

This happens because both cells normally communicate with the same running Python kernel.

Now create another variable:

```python
message = "Hello from Colab"
```

Then:

```python
print(message)
```

The variable remains available because it is stored in the memory of the Python process.

### Experiment

Run:

```python
import os

print("Python version:")
print(os.sys.version)

print("\nCurrent working directory:")
print(os.getcwd())
```

Record the Python version and working directory in your notebook.

---

## 0.3 The runtime

The **runtime** is the computing environment that executes your notebook.

A simplified model is:

```text
                    Google Colab Notebook
                             |
                             v
                     IPython / Python Kernel
                             |
             +---------------+---------------+
             |                               |
             v                               v
       Python memory                  Linux operating system
       Python variables               files, processes, commands
       imported libraries             CPU, RAM, network, etc.
                                             |
                                             v
                                      Optional GPU
```

The runtime is temporary. If it is disconnected or reset, data stored only inside the runtime can disappear.

This distinction becomes important later when we investigate Google Drive.

---

## 0.4 First observation

Run:

```python
import os
import platform

print("Operating system:", platform.system())
print("OS release:", platform.release())
print("Python version:", platform.python_version())
print("Working directory:", os.getcwd())
```

Then run:

```python
!pwd
```

You will probably notice that Python's:

```python
os.getcwd()
```

and Linux's:

```bash
pwd
```

report the same working directory.

This is our first clue that the Python kernel and the Linux environment are related, but they are **not the same thing**.

---

# Task 1 - Python Kernel, Linux Shell, and the Difference Between `!` and `%`

One of the most important concepts in Colab is understanding what happens when a command begins with `!` or `%`.

## 1.1 The three forms

### Ordinary Python

```python
x = 10
print(x)
```

This is executed by the Python/IPython kernel.

### `!` - shell command

```python
!pwd
!ls
!whoami
```

The `!` tells IPython to execute the following command using a shell.

For example:

```python
!ls -lah
```

asks the underlying Linux environment to execute `ls`.

### `%` - IPython line magic

```python
%cd /tmp
%timeit sum(range(1000))
%run my_script.py
```

The `%` tells IPython that this is an **IPython magic command**.

A cell magic uses `%%`:

```python
%%writefile example.py
print("Hello")
```

The important distinction is therefore:

| Syntax     | Interpreted by        | Typical purpose                     |
| ---------- | --------------------- | ----------------------------------- |
| Python     | Python/IPython kernel | Execute Python code                 |
| `!command` | Shell subprocess      | Execute Linux commands              |
| `%magic`   | IPython               | Control or interact with the kernel |
| `%%magic`  | IPython               | Apply a magic to an entire cell     |

### Guiding question

Whenever you see `!` or `%`, ask:

> **Am I asking the Linux operating system to execute a command, or am I asking IPython to perform an operation?**

---

# Task 1A - Working Directories: The Classic `!cd` vs `%cd` Experiment

Before running anything, predict the result.

### Prediction

What do you expect this to print?

```python
!cd /tmp
```

followed by, in a new cell:

```python
!pwd
```

Will the directory be `/tmp`?

Write your prediction in a Markdown cell before continuing.

---

## Experiment 1

Run:

```python
!cd /tmp
```

Then, in a new cell:

```python
!pwd
```

You should normally see the original working directory, such as:

```text
/content
```

### Why?

The `!cd` command runs in a shell subprocess.

Conceptually:

```text
Python/IPython kernel
        |
        +---- creates shell
                  |
                  +---- cd /tmp
                  |
                  +---- shell exits
```

The shell's working directory disappears when that shell process exits.

---

## Experiment 2 - Use `%cd`

Now run:

```python
%cd /tmp
```

Then:

```python
!pwd
```

You should now see:

```text
/tmp
```

Check Python as well:

```python
import os
print(os.getcwd())
```

It should also report:

```text
/tmp
```

### Why is this different?

`%cd` is an IPython magic. It changes the working directory associated with the active IPython/Python environment.

A later `!` command starts a shell using that current working directory.

The relationship is approximately:

```text
%cd /tmp
     |
     v
IPython kernel working directory = /tmp
     |
     v
new ! shell inherits /tmp
     |
     v
!pwd  ---> /tmp
```

---

# Task 1B - Important Counterexample: `!` Can Change the Filesystem

A common mistake is to conclude:

> "`!` is temporary, therefore everything done with `!` disappears."

That statement is **incorrect**.

The shell process is temporary, but the Linux filesystem is separate from the shell's process memory.

Run:

```python
!mkdir -p my_project
!touch my_project/data.txt
```

Now, in a new cell:

```python
!ls -l my_project
```

You should still see:

```text
data.txt
```

The file exists because creating a file changes the filesystem.

The shell that created the file has disappeared, but the file remains in the VM's filesystem.

### Key distinction

```text
Temporary shell state
        |
        +-- shell variables
        +-- shell working directory
        +-- shell process state
        |
        +---- disappears when shell exits


Filesystem
        |
        +-- files
        +-- directories
        |
        +---- remains during the runtime session
```

This distinction is fundamental.

---

# Task 1C - Shell Variables vs Python Variables

Now investigate environment variables.

## Experiment 1 - Shell variable

Run:

```python
!MY_VARIABLE="Colab"
!echo "Inside this command: $MY_VARIABLE"
```

Now run:

```python
!echo "In a new shell: $MY_VARIABLE"
```

The second command will normally show an empty value.

Why?

Each `!` invocation creates a shell environment. The variable created in one shell does not automatically become a variable in the next shell.

---

## Experiment 2 - Python variable

Compare that with Python:

```python
my_variable = "Colab"
print(my_variable)
```

Then, in another cell:

```python
print(my_variable)
```

The value remains available because it belongs to the persistent Python kernel.

---

## Experiment 3 - Python environment variables

Python can modify the environment inherited by processes that it launches.

Run:

```python
import os

os.environ["COURSE_NAME"] = "Cloud Computing"
print(os.environ["COURSE_NAME"])
```

Then:

```python
!echo $COURSE_NAME
```

The shell should inherit the environment variable from the Python process.

### Important distinction

This:

```python
!export COURSE_NAME="Cloud Computing"
```

does not permanently modify the Python process's environment.

Whereas:

```python
import os
os.environ["COURSE_NAME"] = "Cloud Computing"
```

modifies the environment of the Python process, and subsequently launched processes can inherit it.

---

# Task 1D - A Separate Python Process

The `!` syntax can also launch another Python interpreter.

First create a variable in the notebook:

```python
shared_data = [10, 20, 30]

print("Notebook kernel:", shared_data)
```

Now run:

```python
!python3 -c "print('External Python process started')"
```

This is a different operating-system process.

Try:

```python
!python3 -c "print('shared_data' in globals())"
```

The result should be:

```text
False
```

The external Python interpreter does not automatically have access to variables stored in the notebook kernel.

Compare:

```python
print("Notebook kernel:", shared_data)
```

The notebook kernel still knows about `shared_data`.

---

# Task 1E - `%run` vs `!python`

This is an important comparison.

Create a script:

```python
%%writefile experiment.py

experiment_value = 42
print("Script executed")
```

Now run:

```python
%run experiment.py
```

Then:

```python
print(experiment_value)
```

The value should be available in the notebook namespace.

Now restart the experiment conceptually using:

```python
!python3 experiment.py
```

The script executes in a separate Python process.

The key idea is:

```text
%run
  |
  +---- IPython executes the script through its kernel environment


!python3
  |
  +---- shell starts another Python process
             |
             +---- separate process / separate memory
```

---

# Task 1F - Summary

Complete the following table in your notebook.

| Operation                   | Persists across cells? | Explanation                                |
| --------------------------- | ---------------------: | ------------------------------------------ |
| `!cd /tmp`                  |                      False | Changes only the shell's working directory |
| `%cd /tmp`                  |                      True | Changes the IPython working directory      |
| `!MY_VAR=1`                 |                      False | Shell-local variable                       |
| `my_var = 1`                |                      True | Python variable in kernel memory           |
| `!mkdir test`               |                     True* | Filesystem modification                    |
| `!touch file.txt`           |                     True* | Filesystem modification                    |
| `!export KEY=value`         |                      False | Export applies to that shell process       |
| `os.environ["KEY"]="value"` |                      True | Changes the Python process environment     |
| `!python3 script.py`        |       Separate process | Starts another Python interpreter          |
| `%run script.py`            |         Kernel-visible | Executes through IPython                   |

* Until the runtime is reset, disconnected, or the file is removed.

---

# Task 2 - Working with Files and Python Code

Now that you understand the relationship between the notebook and the filesystem, use IPython to create a Python program from a notebook cell.

## 2.1 Creating a Python file with `%%writefile`

The `%%writefile` cell magic writes the contents of the cell to a file.

Run:

```python
%%writefile math_utils.py

def square_sum(n):
    return sum(i * i for i in range(n))

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

Verify that the file exists:

```python
!ls -l math_utils.py
```

You can also inspect its contents:

```python
!cat math_utils.py
```

---

## 2.2 Execute the file with `%run`

Run:

```python
%run math_utils.py
```

Then:

```python
print(square_sum(10))
print(factorial(5))
```

The functions are now available in the notebook's IPython namespace.

---

## 2.3 Compare `%run` with ordinary Python import

You can also import Python files:

```python
import math_utils

print(math_utils.square_sum(10))
```

The two approaches have different purposes.

| Method            | Purpose                                                |
| ----------------- | ------------------------------------------------------ |
| `%run file.py`    | Execute a script through IPython                       |
| `import module`   | Import a Python module                                 |
| `!python file.py` | Start a separate Python process and execute the script |

---

# Task 3 - Measuring Python Performance

Now that you can create and execute Python code, investigate performance.

## 3.1 Why timing matters

Suppose two implementations produce the same result:

```python
sum(i * i for i in range(1_000_000))
```

and:

```python
import numpy as np
np.sum(np.arange(1_000_000, dtype=np.int64) ** 2)
```

They may have very different execution times.

Rather than relying on a single measurement, IPython provides `%timeit`.

---

## 3.2 `%timeit`

Run:

```python
%timeit sum(i * i for i in range(1_000_000))
```

Then:

```python
import numpy as np

%timeit np.sum(np.arange(1_000_000, dtype=np.int64) ** 2)
```

Record the reported times.

### Question

Which implementation is faster?

Why might NumPy be faster even though both approaches calculate the same mathematical result?

Consider:

* compiled native code;
* vectorized operations;
* Python interpreter overhead;
* memory access;
* optimized numerical libraries.

---

## 3.3 `%time` vs `%timeit`

Compare:

```python
%time sum(i * i for i in range(1_000_000))
```

with:

```python
%timeit sum(i * i for i in range(1_000_000))
```

`%time` performs a timing measurement for an execution.

`%timeit` performs repeated measurements and is generally more appropriate for small pieces of code where you want a more reliable micro-benchmark.

### Question

Why might one execution be less reliable than many repetitions?

---

# Task 4 - Inspecting the Cloud Hardware

So far, we have treated Colab as a Python environment. It is also a virtualized computing environment with CPU, memory, storage, and potentially GPU resources.

## 4.1 Inspect the CPU

Run:

```python
!lscpu | grep -E "Model name|Socket|Thread|NUMA|CPU\(s\)"
```

Also run:

```python
!free -h
```

Record:

* CPU model;
* number of available CPUs;
* available memory.

---

## 4.2 Inspect the operating system

Run:

```python
!cat /etc/os-release
```

Identify the Linux distribution and version.

---

# Task 5 - GPU Acceleration

A GPU is designed to execute many operations in parallel.

For suitable workloads, especially large numerical operations, a GPU can perform calculations much faster than a CPU.

However, a GPU is **not automatically faster for every task**.

Small computations may be dominated by:

* data-transfer overhead;
* kernel-launch overhead;
* synchronization;
* setup costs.

Your goal is to investigate this experimentally.

---

## 5.1 Enable a GPU runtime

In Colab:

**Runtime → Change runtime type**

Select an available GPU under **Hardware accelerator**.

The exact GPU available to you may vary. Do not assume that every student receives the same model.

---

## 5.2 Inspect the GPU

Run:

```python
!nvidia-smi
```

Record:

* GPU model;
* GPU memory;
* CUDA version information, if displayed;
* current GPU memory usage.

---

## 5.3 Check GPU availability from PyTorch

Run:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
```

---

# Task 6 - CPU vs GPU Matrix Multiplication

Now perform a real computational experiment.

## 6.1 Mathematical background

For two square matrices:

$$
A,B \in \mathbb{R}^{n \times n}
$$

matrix multiplication produces:

$$
C = AB
$$

where each element is:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

The standard algorithm performs approximately $\mathcal{O}(n^3)$ arithmetic work.

This makes matrix multiplication a useful workload for demonstrating parallel hardware.

---

## 6.2 Benchmark implementation

Run:

```python
import torch
import time

def benchmark_matmul(n=4000):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Enable a GPU runtime first."
        )

    print(f"Matrix size: {n} x {n}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------
    # CPU
    # -------------------------
    a_cpu = torch.randn(n, n, dtype=torch.float32)
    b_cpu = torch.randn(n, n, dtype=torch.float32)

    # Warm up CPU execution
    _ = torch.matmul(a_cpu, b_cpu)

    start = time.perf_counter()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.perf_counter() - start

    # -------------------------
    # GPU
    # -------------------------
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    # Warm up GPU
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    start = time.perf_counter()

    c_gpu = torch.matmul(a_gpu, b_gpu)

    # GPU operations are asynchronous.
    # Wait for the GPU before stopping the timer.
    torch.cuda.synchronize()

    gpu_time = time.perf_counter() - start

    speedup = cpu_time / gpu_time

    print(f"CPU time: {cpu_time:.4f} s")
    print(f"GPU time: {gpu_time:.4f} s")
    print(f"GPU speedup: {speedup:.2f}x")

    return cpu_time, gpu_time, speedup
```

Run:

```python
cpu_time, gpu_time, speedup = benchmark_matmul(4000)
```

---

# Task 7 - Understanding CUDA Asynchrony

The previous task contains a line that is extremely important:

```python
torch.cuda.synchronize()
```

Why is it necessary?

GPU operations are often launched **asynchronously**.

Conceptually:

```text
CPU
 |
 | launch GPU operation
 v
GPU --------------------> computation
 |
 | CPU can continue
 v
next Python instruction
```

Therefore, this code can produce a misleading measurement:

```python
start = time.perf_counter()

c_gpu = torch.matmul(a_gpu, b_gpu)

gpu_time = time.perf_counter() - start
```

The CPU timer may stop before the GPU has finished the computation.

Instead:

```python
start = time.perf_counter()

c_gpu = torch.matmul(a_gpu, b_gpu)

torch.cuda.synchronize()

gpu_time = time.perf_counter() - start
```

forces the CPU to wait for the GPU operation to complete.

---

## 7.1 Deliberately perform an incorrect benchmark

Run:

```python
start = time.perf_counter()

_ = torch.matmul(a_gpu, b_gpu)

incorrect_time = time.perf_counter() - start

print("Without synchronization:", incorrect_time)
```

Then:

```python
torch.cuda.synchronize()

start = time.perf_counter()

_ = torch.matmul(a_gpu, b_gpu)

torch.cuda.synchronize()

correct_time = time.perf_counter() - start

print("With synchronization:", correct_time)
```

### Questions

1. Are the two measurements significantly different?
2. Why can the first measurement be misleading?
3. What does `torch.cuda.synchronize()` force the CPU to do?
4. Why should GPU benchmarks include a warm-up?

---

# Task 8 - Investigate the Effect of Problem Size

A GPU is not necessarily beneficial for every workload.

Run the benchmark for several matrix sizes:

```python
for size in [500, 1000, 2000, 3000]:
    print("\n" + "=" * 50)
    benchmark_matmul(size)
```

Record the results.

Create a table:

| Matrix size | CPU time | GPU time | GPU speedup |
| ----------: | -------: | -------: | ----------: |
|   500 × 500 |          |          |             |
| 1000 × 1000 |          |          |             |
| 2000 × 2000 |          |          |             |
| 3000 × 3000 |          |          |             |

### Analysis

Answer:

1. Does GPU speedup increase with problem size?
2. Is the GPU always faster?
3. Why might a small matrix not benefit as much?
4. What role might data transfer and kernel-launch overhead play?

---

# Task 9 - Verify CPU and GPU Results

A benchmark should not only measure speed. We should also verify that the computations produce equivalent results.

Run:

```python
difference = torch.max(
    torch.abs(c_cpu - c_gpu.cpu())
)

print("Maximum absolute difference:", difference.item())
```

The difference should be small rather than necessarily exactly zero.

### Question

Why might floating-point computations performed on different hardware produce slightly different results?

Discuss:

* floating-point representation;
* operation order;
* parallel execution;
* numerical precision.

---

# Task 10 - Document the Experiment with Markdown and $\LaTeX$

A scientific experiment should be documented rather than represented only by raw output.

Insert a **Markdown cell** below your benchmark.

Your report must contain the following sections.

## 10.1 Hardware

Record:

* CPU model;
* number of CPUs;
* RAM;
* GPU model;
* GPU memory.

---

## 10.2 Mathematical model

Include:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

Explain that standard matrix multiplication requires $\mathcal{O}(n^3)$ arithmetic work.

---

## 10.3 Benchmark table

Create a table containing your **actual measurements**.

For example:

```markdown
| Matrix size | CPU time (s) | GPU time (s) | Speedup |
|---:|---:|---:|---:|
| 500 × 500 |  |  |  |
| 1000 × 1000 |  |  |  |
| 2000 × 2000 |  |  |  |
| 3000 × 3000 |  |  |  |
```

Do not copy benchmark values from another student's notebook.

---

## 10.4 Explain the speedup

Include a short explanation addressing:

* why GPUs can accelerate matrix multiplication;
* why GPU operations are often asynchronous;
* why synchronization is required for accurate timing;
* why the GPU may not provide the same speedup for small workloads.

Use a blockquote for an important observation:

```markdown
> **Important:** GPU benchmarks must account for asynchronous execution.
> CPU-side timing without synchronization can underestimate the actual
> GPU computation time.
```

---

# Task 11 - Persistent Storage with Google Drive

At this point, we have created files in `/content/`.

But there is an important limitation:

> The Colab runtime is temporary.

Files stored only inside the runtime should be considered temporary.

For data that must survive runtime termination, use persistent storage such as Google Drive.

---

## 11.1 Mount Google Drive

Run:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Follow the authentication instructions provided by Colab.

---

## 11.2 Inspect your Drive

Run:

```python
!ls -lah "/content/drive/MyDrive" | head
```

You should see files and directories from your Google Drive.

---

## 11.3 Compare temporary and persistent storage

Create a temporary file:

```python
!echo "temporary data" > /content/temporary.txt
```

Create a Drive file:

```python
!echo "persistent data" > "/content/drive/MyDrive/colab_test.txt"
```

Verify both:

```python
!cat /content/temporary.txt
!cat "/content/drive/MyDrive/colab_test.txt"
```

### Question

What is the difference between:

```text
/content/
```

and:

```text
/content/drive/MyDrive/
```

Explain why this distinction matters when training machine-learning models or producing experiment results.

---

# Task 12 - Collaboration and Reproducibility

A Colab notebook is also a collaborative document.

## 12.1 Share the notebook

1. Click **Share**.
2. Add your lab partner's university account.
3. Give the required permission, such as **Editor**.
4. Confirm that your partner can open the notebook.

---

## 12.2 Add a comment

Use the notebook's commenting functionality to leave your partner a question or observation about the GPU benchmark.

For example:

> Why do you think the GPU speedup changed when we increased the matrix size?

Your partner should respond to the comment.

---

## 12.3 Reproducibility check

Ask your partner to execute the notebook from the beginning.

Check whether they can reproduce:

* the environment information;
* the Python benchmark;
* the GPU benchmark;
* the generated files;
* the Markdown report.

If something only works because you manually performed an undocumented action, improve the notebook.

---

# Final Reflection

Add a final Markdown cell answering the following questions.

### Question 1

What is the difference between:

```python
print(...)
```

```python
!command
```

and:

```python
%magic
```

### Question 2

Why does:

```python
!cd /tmp
```

not permanently change the notebook's working directory, while:

```python
%cd /tmp
```

does?

### Question 3

Why can:

```python
!touch file.txt
```

create a file that remains available even though the shell process itself terminates?

### Question 4

What is the difference between the Python kernel and a separate process started with:

```python
!python3 script.py
```

### Question 5

Why is:

```python
torch.cuda.synchronize()
```

important when measuring GPU execution time?

### Question 6

Why is a GPU not automatically faster than a CPU for every computation?

### Question 7

Why should important experiment data not be stored only in `/content/`?

---

# Submission Requirements

Your final Colab notebook must contain:

* [ ] Names and student IDs of all group members at the top.
* [ ] An explanation of the Colab notebook and its cell types.
* [ ] A demonstration of Python variables persisting across cells.
* [ ] The `!cd` vs `%cd` experiment.
* [ ] The filesystem persistence counterexample.
* [ ] The shell-variable vs Python-variable experiment.
* [ ] The separate-process experiment using `!python3`.
* [ ] A demonstration of `%%writefile`.
* [ ] A demonstration of `%run`.
* [ ] `%timeit` performance measurements.
* [ ] CPU and operating-system information.
* [ ] GPU hardware information using `!nvidia-smi`.
* [ ] A PyTorch GPU availability check.
* [ ] CPU vs GPU matrix-multiplication measurements.
* [ ] Correct use of `torch.cuda.synchronize()`.
* [ ] Measurements for multiple matrix sizes.
* [ ] Verification that CPU and GPU results are numerically similar.
* [ ] A Markdown report containing a benchmark table.
* [ ] Mathematical notation using inline and block $\LaTeX$.
* [ ] An explanation of GPU speedup and CUDA asynchrony.
* [ ] Google Drive mounted and tested.
* [ ] Evidence of collaboration through notebook comments.
* [ ] Final reflection questions answered.

---
<!-- 
# Key Concepts to Remember

At the end of this lab, you should be able to explain the following model:

```text
                         COLAB NOTEBOOK
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
        Markdown cells                    Code cells
        documentation                    |
                                          v
                                   IPython / Python
                                       KERNEL
                                          |
                 +------------------------+----------------------+
                 |                        |                      |
                 v                        v                      v
             Python code             % magic                %% magic
                 |
                 |
                 +--------------------+
                                      |
                                      v
                              Linux environment
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                    ! commands              Files / processes
                         |
                         v
                   Shell subprocess
                         |
                         +---- shell-local state
                         |
                         +---- Linux commands
                         |
                         +---- filesystem changes
                         
                                      |
                                      v
                              Hardware resources
                         +------------+------------+
                         |                         |
                        CPU                       GPU
                                                   |
                                                   v
                                             CUDA / PyTorch


                         Persistent storage
                                  |
                                  v
                       Google Drive / MyDrive
```

The central idea of the lab is that **a Colab notebook is an interface to several interacting layers**. Understanding those layers allows you to predict what will persist, what will disappear, where code is executing, and how computational resources are being used. 

# Recap  ! vs %

In Jupyter and Google Colab environments, the distinction between **`!`** and **`%`** lies in **where the command is executed** and **how process state is handled**.

---

### 1. `!` (Shell Execution)

* **Execution Context:** Spawns a separate, temporary shell subprocess (such as `/bin/bash`) beneath the operating system.
* **Scope and Lifetime:** The subshell is created, executes the command, and immediately terminates upon completion.
* **State Behavior:**
  * **Process-level state does not persist:** Changes to the subshell's environment (such as environment variables or working directory changes via `!cd`) are discarded when the subprocess exits.
  * **Filesystem state does persist:** Any modifications written to disk (such as `!mkdir`, `!touch`, or `!pip install`) remain stored in the filesystem of the virtual machine.

---

### 2. `%` (IPython Magic Command)

* **Execution Context:** Interpreted and executed directly within the active **IPython kernel process**.
* **Scope and Lifetime:** Runs in the same process that executes your Python code and manages the notebook session.
* **State Behavior:**
  * **Kernel state persists:** Commands that alter session configuration (such as `%cd` modifying the Python working directory or `%env` modifying `os.environ`) directly alter the active runtime for all subsequent cells.
  * **Specialized functionality:** Magics provide development and profiling tools integrated with the Python runtime, such as `%timeit` (micro-benchmarking) or `%%writefile` (cell redirection).

---

### Technical Comparison

| Characteristic | `!command` (Shell Subprocess) | `%magic` (IPython Magic) |
| :--- | :--- | :--- |
| **Interpreter** | Operating system shell (`bash`/`sh`) | IPython kernel runtime |
| **Process Boundary** | External subprocess | In-process (within the kernel) |
| **Working Directory Changes** | Temporary (reverts after cell execution) | Persistent (updates the kernel process) |
| **Environment Variables** | Limited to the subprocess lifespan | Directly modifies Python process environment |
| **Primary Use Cases** | Running CLI tools, managing packages, querying system binaries (`nvidia-smi`, `git`) | Managing notebook environment, directory navigation, code profiling (`%timeit`, `%run`) |

---

### Practical Demonstration

```python
# Shell Command: Does not change the kernel's directory
!cd /tmp
!pwd
# Output: /content (remains in the original directory)

# IPython Magic: Modifies the kernel's working directory
%cd /tmp
!pwd
# Output: /tmp (future shell commands inherit this updated directory)
```

-->
