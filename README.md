主要工作是把示例中的TextDataset类改成了MyTextDataset类，以及修改了loss的计算方法。

# Dataset

数据集采用personachat数据集，上下文包括persona和 dialogue history, 长度限制在100以内，不足用<pad>填补，超出的部分截去；训练用的对话长度为20，不足用<pad>填补，超出的部分截去。在
transformers/src/transformers/data/datasets/language_modeling.py中修改。

# Loss

只计算response部分的loss，即后20个字符的loss。在transformers/src/transformers/modeling_gpt2.py中修改。

# run_lm.py

用于训练模型的代码。

# generation.py

在transformers中文本生成代码基础上修改得到的生成对话的代码，输入persona和context（或只输入context），生成对话并保存到相应文件中。

# test_generation.py

自己写的生成对话的代码，输入persona和context（或只输入context），生成对话并保存到相应文件中。

# response_*.txt

模型在测试集上生成的对话。

# metrics_*.txt

模型在测试集上的评测结果。
