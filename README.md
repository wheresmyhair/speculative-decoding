# Speculative Decoding
投机解码  
参考文献: [Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192v2)](https://arxiv.org/abs/2211.17192)  
目前仅在 T5-small (作为draft) 以及 T5-base (作为target) 上进行了代码流程测试。  
目前支持 top-p, argmax, temperature 调整。

# TODO
1. 代码Peer Review
2. 继续学习LLM的采样过程，HF的采样实现代码
3. 适配更多模型
4. 重新整理代码逻辑
   - 目前逻辑比较直接，计划重新设计为面向对象形式
   - 实现拒绝采样后中断还未完成的推理，进一步加速
5. 缓存库？


# Questions
1. 模型生成终止符的原理和其他任何token是一样的吗？目前sampling过程停不下来，只能通过max_length来控制
2. 模型输入（即prompt）应包括终止符吗？
```python
tensor([[13959,  1566,    12,  2968,    10,   571,   625,     1]])
tensor([[13959,  1566,    12,  2968,    10,   571,   625]])
```