主要工作是把示例中的TextDataset类改成了MyTextDataset类，以及修改了loss的计算方法。

# Dataset 

数据集采用personachat数据集，上下文包括persona和 dialogue history, 长度限制在100以内，不足用<pad>填补，超出的部分截去；训练用的对话长度为20，不足用<pad>填补，超出的部分截去。在
transformers/src/transformers/data/datasets/language_modeling.py中修改。
# Loss

只计算response部分的loss，即后20个字符的loss。在transformers/src/transformers/modeling_gpt2.py中修改。
