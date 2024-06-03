# 每个数据集的配置不同，以下是这四个数据集的配置情况。
def get_default_config(data_name):
    if data_name in ['Caltech101-20']:       #101个物体图片的数据集
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation 
                #------------------------------------------------------------------------------------------------
                # 表示这个自编码器子网络有五层，第一层有 1984 个神经元，
                # 第二层、第三层和第四层各有 1024 个神经元，最后一层（也称为潜在表示层或瓶颈层）有 128 个神经元。
                # 最后一层的神经元数量通常是用于特征表示的维度。
                #------------------------------------------------------------------------------------------------
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',                 # 表示在 arch1 子网络的每一层都应用 ReLU 激活函数。
                activations2='relu',                 # 表示在 arch2 子网络的每一层也都应用 ReLU 激活函数。
                batchnorm=True,                      # 批归一化处理,显著提升模型的训练性能和泛化能力
            ),
            training=dict(
                seed=4,                              # 随机种子=4，确保实验的可重复性
                missing_rate=0.5,                    # 缺失率
                start_dual_prediction=100,           # 表示在训练到第100个epoch时开始进行双重预测。
                batch_size=256,                      # batch_size=256 表示每次迭代使用256个样本进行训练
                epoch=500,                           # epoch=500 表示要训练500个轮次。
                lr=1.0e-4,                           # lr=1.0e-4 表示学习率为0.0001。
                # Balanced factors for L_cd, L_pre, and L_rec  文中的其它温度平衡参数
                alpha=9,                        
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['Scene_15']:        #15种场景下的图片数据集
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 64],
                arch2=[784, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=0,
                start_dual_prediction=100,
                epoch=500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 64],
                arch2=[40, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0,
                seed=3,
                start_dual_prediction=100,
                epoch=500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    else:
        raise Exception('Undefined data_name')                # 如果 data_name 不在上述数据集中，函数会抛出异常：
