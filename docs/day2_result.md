# Day 2 CheckList

- [x] DDPM schedule 命名统一;
- [x] schedule 使用 register_buffer
- [x] 实现统一的extract函数;
- [x] q_sample 可独立调用;
- [x] predict_x0_from_noise 可独立调用;
- [x] sampler 不再访问错误的 alpha_bar 属性;
- [x] t=0时不添加随机噪声;
- [x] 完整 reverse loop smoke test 通过;
- [x] 所有测试输出均为finite;
- [x] 完成3个PyTorch手写函数;
- [x] 完成3道算法题. 