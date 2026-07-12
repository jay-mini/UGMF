# Day 1 Baseline

- Python: 3.11.5
- PyTorch: 2.5.1
- CUDA: True
- GPU: 0

## Commands

python -m pytest -q

## Result

(brain_dynamic) PS D:\Documents\Doctoral_Research\UGMF> python -m pytest -q
..uu.                                                                                                                                                                         [100%]
3 passed, 2 subtests passed in 21.21s


## Checklist

- [x] 创建分支
- [x] 补齐必要的__init__.py
- [x] pip install -e ".[dev]" 成功
- [x] python -c "import ugm" 成功
- [x] python -m pytest 不再发生导入错误
- [x] 阅读train.py -> objective -> unet -> trainer 主链
- [x] 创建docs/training_flow.md
- [x] 记录修改前的测试错误
- [x] 完善pyproject.toml
- [x] 更新README的Current Progress
- [x] 完成append dims
- [x] 完成extract
- [x] 完成3道数组和哈希题
- [x] 提交Day1修改     
