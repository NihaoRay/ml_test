这里seq_to_seq的训练代码与模型的创建代码是分开的，如果训练代码也在一起的话，
在引用这个py的时候，会导致里面的训练代码会被调用，因为训练代码不是封装在类里面。
所以要把训练代码与模型的代码分开。