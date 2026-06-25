#!/bin/bash
#SBATCH --job-name=create_kernelbook
#SBATCH --output=create_kernelbook%j.out
#SBATCH --error=create_kernelbook%j.err
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

uv run python kernel_gen/convert_kernelbook.py --all --out KernelBench/level6 --smoke-test