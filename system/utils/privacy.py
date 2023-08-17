from opacus import PrivacyEngine

MAX_GRAD_NORM = 1.0
DELTA = 1e-5
EPSILON = 50.0

def initialize_dp(model, optimizer, data_loader, dp_sigma):
    """
    dp_sigma: 高斯噪声的标准偏差与添加噪声的函数的 L2 灵敏度之比 (要添加多少噪声)
    """
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def initialize_dp_with_budget(model, optimizer, data_loader, EPOCHS):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    epochs=EPOCHS,
    data_loader=data_loader,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)
    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA