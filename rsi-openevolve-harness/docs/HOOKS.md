# 续训策略钩子约定

只在 [policy_initial.py](../harness/policy_initial.py) 的一对 EVOLVE-BLOCK 注释之间改 CONFIG 和五个钩子。钩子就是由固定训练流程在指定时刻调用的函数。先复制策略做单独评估，再交给搜索；不要修改 runner。这里与 [系统提示的钩子约定](../prompts/system_message.md) 一致，补充 runner 实际检查的边界。

## 调用流程与返回约定

runner 加载本阶段起点检查点，保留网络参数、Adam 一阶/二阶状态和步数；沿用输入标准化与回归目标缩放。sample_weights 对整个训练集调用一次；每轮调用 learning_rate；每个 batch（小批量样本）调用 output_grad 和 extra_grad；全部轮次训练完后调用 pick_epoch，恢复选中轮次的权重、优化器状态与元数据。这不是在训练中提前中止。

| 项目 | 输入和含义 | 必须返回或满足 |
| --- | --- | --- |
| `CONFIG["epochs"]` | 本次实际训练轮数，初始 5 | 真正的整数，默认 1..50；上限由运行配置 limits.max_epochs 传入。bool 不算整数 |
| `CONFIG["batch_size"]` | 初始 None，继承检查点的 batch_size | None 或真正的整数 1..65536 |
| `CONFIG["grad_clip"]` | 合并输出梯度与额外梯度后，对总范数裁剪；初始 5.0 | 正的有限 int/float，不能是 bool |
| `learning_rate(epoch, total_epochs, base_lr)` | epoch 从 0 开始，是本次训练的局部轮次；base_lr 来自起点检查点 | (0, 1] 内的有限数值；不允许 bool。推荐返回 float |
| `sample_weights(y, task, num_classes)` | y 是完整训练集目标副本，形状 (n,)；分类为类别下标，回归为原始单位目标 | 可转为 float64 的 (n,) 数组，有限、非负，且总和严格大于 0 |
| `output_grad(out, target, w, task)` | 当前 batch 的输出、目标和权重 | 长度 2 的 tuple：`(loss, grad)`；loss 为有限标量，grad 有限且与 out 同形 |
| `extra_grad(params)` | 参数的只读视图；键 w1、b1、w2、b2 | 恰含这四个键的 dict；每项形状与对应参数相同，数值有限 |
| `pick_epoch(history)` | 每轮训练完成后得到的历史列表 | 整数下标 0..len(history)-1，不能是 bool；0 是本次第一轮，不是起点 |

`task` 取 `classification` 或 `regression`。分类的 out 是未经 softmax 的 logits，形状 (b, C)，target 是 (b,) 类别下标；回归 out 是 (b, 1)，target 是 (b,) **标准化目标**。回归 sample_weights 收到的 y 却是原单位目标，切勿混用。w 是整个训练集权重在当前 batch 上的切片，形状 (b,)；runner 不会自动把权重均值归一化。

`loss` 约定为本 batch 的损失之和；`grad` 是 batch 平均损失对 out 的梯度，因此通常需要除以实际 batch 大小 b。不能把损失之和的梯度直接当作平均梯度，否则梯度量级会变化。分类默认加权交叉熵，回归默认加权平方误差。

参数形状：w1=(输入数, 隐藏数)，b1=(隐藏数,)，w2=(隐藏数, 输出数)，b2=(输出数,)。extra_grad 用于例如 L2 正则的附加梯度，不能原地改 params。输出梯度经固定反向传播后与 extra_grad 相加，再由固定 Adam 更新。

history 每项为 `{"epoch": 累计轮次, "train_loss": 浮点数, "val": {指标名: 值}}`。分类 val 只有 accuracy、macro_f1；回归只有 rmse、mae，没有 nasa_score，也没有各组指标。val 来自挑选集，不是验收集。B 的 history 是按样本合并的 RMSE，而最终评分可能是分组 RMSE 的等权平均。

## 可以改什么，不能改什么

可以在块内调整训练轮数、batch 大小、裁剪阈值、学习率、样本权重、输出损失/梯度、参数正则和轮次选择，也可定义必要的纯计算辅助函数。初始写法分别是固定学习率、全 1 权重、官方损失、零附加梯度、选最后一轮。

保持函数名字、参数列表与返回形状不变；保持两条块标记及块外内容不变。不得修改训练循环、网络结构、特征顺序、类别、标准化、回归缩放、Adam 公式、随机洗牌流程、数据文件、指标和检查点格式。不得读取验收集或直接读写检查点。只能使用 numpy（已有别名 np）和 math，不读取文件、网络、环境或执行外部代码。

静态检查拒绝非 numpy/math 导入、危险内置名、双下划线属性、NumPy 文件读写属性和 global/nonlocal 等；检查点检查还核对元数据、标准化、参数形状、步数/轮次增长。OpenEvolve 的 enforce_evolve_blocks 负责撤回块外修改；直接调用 evaluator 的静态检查并不等同于对照原文件逐字检查块外内容。以上约束是使用契约，不是完整安全边界。

## 示例一：提高少数类样本权重

适用于反馈出现 CLASS_NEVER_PREDICTED 或 CLASS_LOW_RECALL 时。仅替换 sample_weights；回归仍返回全 1，分类按频率倒数加权并归一化到均值 1，避免整体放大梯度。

~~~python
def sample_weights(y, task, num_classes):
    if task != "classification":
        return np.ones(len(y))
    counts = np.bincount(y, minlength=num_classes).astype(float)
    weights = (len(y) / (num_classes * np.maximum(counts, 1.0)))[y]
    return weights / weights.mean()
~~~

这只是可检验假设，不保证收益。检查各类召回率，同时看 accuracy 护栏，避免牺牲大类换少数类分数。

## 示例二：选择挑选集上最好的一轮

适用于 LATE_EPOCH_DEGRADATION。仅替换 pick_epoch；分类选 macro_f1 最大轮，回归选 rmse 最小轮。np.argmax/argmin 遇到并列时选择第一次出现的位置。

~~~python
def pick_epoch(history):
    if "macro_f1" in history[0]["val"]:
        return int(np.argmax([h["val"]["macro_f1"] for h in history]))
    return int(np.argmin([h["val"]["rmse"] for h in history]))
~~~

全部请求轮次仍会执行，最后恢复所选轮次的完整状态；最终 epochs/steps 可能小于训练日志中最后一轮，但必须超过起点。两个示例应分别评测，再判断是否组合，不要将示例直接视为已验收策略。
