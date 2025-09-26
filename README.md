# A-Class-of-Adaptive-Stochastic-Gradient-Methods-for-Large-Scale-Optimization
大规模机器学习的一类自适应随机梯度方法：理论与实验

This thesis provides an overview of adaptive stochastic gradient descent methods for large-scale optimization, and elaborates on their background, theoretical properties and practical performance. First, the thesis introduces a class of large-scale optimization problems by presenting the structural risk model in machine learning, and points out the limitations of traditional gradient descent (GD) and stochastic gradient descent (SGD) methods in handling such problems. The thesis then introduces the AdaGrad algorithm, explains its iterative format and motivation, and provides a convergence rate of $O (1/\sqrt{T})$ in convex optimization and a rate of $O (\ln T/\sqrt{T})$ in smooth non-convex optimization. The thesis reviews the sublinear regret of AdaGrad and its scalar form AdaGrad-Norm, and compares them with SGD. Subsequently, the thesis points out the defects of AdaGrad, such as excessive decay of step size, and introduces improved algorithms such as RMSProp, AdaDelta, and Adam. Finally, the thesis utilizes the aforementioned algorithms to train three models: Logistic Regression, Support Vector Machine and Multilayer Perceptron, verifying the superior performance of adaptive stochastic gradient methods on sparse datasets and their consistently excellent performance on various optimization problems.

论文介绍了一系列用于求解大规模优化问题的自适应随机梯度方法，并就其产生背景、理论性质和实验表现几个方面详细展开。首先，论文通过介绍机器学习中的结构风险模型，引入了一类大规模优化问题，并指出传统的梯度下降法（GD）和随机梯度下降法（SGD）在处理此类问题时的不足之处。随后，论文引入AdaGrad算法，介绍其迭代格式与动机，并给出AdaGrad在凸优化问题中 $O (1/\sqrt{T})$ 的收敛速率、光滑非凸优化问题中 $O (\ln T/\sqrt{T})$ 的收敛速率。论文回顾了AdaGrad及其标量形式AdaGrad-Norm的次线性遗憾 $R(T)=O(\sqrt{T})$ ，并将其与SGD作比较。随后，论文指出AdaGrad具有步长过度衰减等缺陷，并引入RMSProp, AdaDelta, Adam等改进算法。最后，论文利用上述算法对Logistic回归、支持向量机、多层感知机三种模型进行训练，验证了自适应随机梯度方法在稀疏数据集上的优越表现，以及在各类优化问题上一致的优秀性能。

部分结果展示：

<img width="1446" height="1182" alt="image" src="https://github.com/user-attachments/assets/f34ac3dc-b351-4356-9bc0-4e4a052990d0" />

<img width="1464" height="1042" alt="image" src="https://github.com/user-attachments/assets/5f5b0529-831f-4750-9fff-a01e95b91714" />

