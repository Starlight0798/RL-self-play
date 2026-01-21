from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def get_action(self, obs, mask=None, deterministic=False):
        """
        根据观测返回动作。
        Returns:
            action: 动作 (Tensor or Array)
            info: 算法特定的额外信息 (Dict), 例如 logprobs, values
        """
        pass
    
    @abstractmethod
    def store_transition(self, obs, action, reward, done, mask, info):
        """
        存储一步交互数据到缓冲区。
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        执行算法更新。
        Returns:
            metrics: 训练指标 (Dict)
        """
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
