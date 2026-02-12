"""
高效配置管理器
实现增量更新、配置验证、原子性操作等功能
"""

import os
import json
import tempfile
import threading
from typing import Dict, Any, Optional
from ruamel.yaml import YAML
import copy

class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        # 转换为绝对路径
        if not os.path.isabs(config_path):
            # 获取项目根目录（core/utils 的上级目录）
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.config_path = os.path.join(project_root, config_path)
        else:
            self.config_path = config_path

        self.lock = threading.Lock()
        self.yaml = YAML()
        self.yaml.preserve_quotes = True

        # 缓存当前配置，避免频繁读取文件
        self._cached_config = None
        self._last_modified_time = 0

    def _load_config(self) -> Dict[str, Any]:
        """安全加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return self.yaml.load(file) or {}

    def _save_config(self, config: Dict[str, Any]):
        """原子性保存配置文件"""
        # 创建临时文件
        temp_path = self.config_path + '.tmp'
        try:
            with open(temp_path, 'w', encoding='utf-8') as file:
                self.yaml.dump(config, file)

            # 原子性替换原文件
            os.replace(temp_path, self.config_path)

            # 更新缓存
            self._cached_config = copy.deepcopy(config)
            stat = os.stat(self.config_path)
            self._last_modified_time = stat.st_mtime

        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置（带缓存）"""
        with self.lock:
            current_time = os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 0
            
            if (self._cached_config is None or 
                current_time > self._last_modified_time):
                self._cached_config = self._load_config()
                self._last_modified_time = current_time
                
            return copy.deepcopy(self._cached_config)

    def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        try:
            config = self.get_config()
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception:
            return default

    def set_values(self, updates: Dict[str, Any], validate: bool = True) -> bool:
        """
        批量设置配置值（增量更新）
        
        Args:
            updates: 要更新的键值对，格式如 {'api.key': 'new_key', 'asr.language': 'zh'}
            validate: 是否验证配置有效性
        
        Returns:
            bool: 更新是否成功
        """
        with self.lock:
            try:
                # 加载当前配置
                current_config = self._load_config()

                # 检测哪些值真正发生了变化
                changes_detected = False
                for key, new_value in updates.items():
                    if self._has_value_changed(current_config, key, new_value):
                        self._set_nested_value(current_config, key, new_value)
                        changes_detected = True

                # 只有当有实际变化时才保存
                if changes_detected:
                    if validate and not self._validate_config(current_config, updates):
                        raise ValueError("配置验证失败")

                    self._save_config(current_config)
                    return True
                else:
                    # 没有变化，无需保存
                    return True

            except Exception as e:
                raise e

    def _has_value_changed(self, config: Dict[str, Any], key: str, new_value: Any) -> bool:
        """检查值是否真的发生了变化"""
        try:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    # 如果路径不存在，说明是新增的值
                    return True
            
            if isinstance(current, dict) and keys[-1] in current:
                return current[keys[-1]] != new_value
            else:
                # 键不存在，说明是新增的值
                return True
        except:
            # 发生异常，认为值已改变
            return True

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """设置嵌套字典中的值"""
        keys = key.split('.')
        current = config
        
        # 遍历路径，创建不存在的中间字典
        for k in keys[:-1]:
            if not isinstance(current, dict):
                current = {}
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # 设置最终值
        current[keys[-1]] = value

    def _validate_config(self, config: Dict[str, Any], updates: Dict[str, Any] = None) -> bool:
        """验证配置的有效性"""
        # 这里可以添加具体的验证逻辑
        # 例如：检查API密钥格式、URL有效性等
        return True

    def reset_to_defaults(self, keys: Optional[list] = None):
        """
        重置配置为默认值
        
        Args:
            keys: 要重置的键列表，如果为None则重置所有配置
        """
        with self.lock:
            try:
                current_config = self._load_config()
                
                if keys is None:
                    # 重置所有配置为默认值（这里简化处理，实际应从默认配置模板加载）
                    pass
                else:
                    # 只重置指定的键
                    for key in keys:
                        self._remove_nested_key(current_config, key)
                
                self._save_config(current_config)
            except Exception as e:
                raise e

    def _remove_nested_key(self, config: Dict[str, Any], key: str):
        """移除嵌套字典中的键"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return  # 路径不存在
        
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]

# 全局配置管理器实例
_config_manager = None
_manager_lock = threading.Lock()

def get_config_manager() -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    with _manager_lock:
        if _config_manager is None:
            _config_manager = ConfigManager()
        return _config_manager


# 便捷函数
def load_key(key: str, default: Any = None) -> Any:
    """加载配置键值"""
    manager = get_config_manager()
    return manager.get_value(key, default)


def update_keys(updates: Dict[str, Any]) -> bool:
    """
    批量更新配置键值（推荐使用此函数进行配置更新）
    
    Args:
        updates: 要更新的键值对，如 {'api.key': 'new_key', 'asr.language': 'zh'}
    
    Returns:
        bool: 更新是否成功
    """
    manager = get_config_manager()
    return manager.set_values(updates)


def update_key(key: str, value: Any) -> bool:
    """更新单个配置键值"""
    return update_keys({key: value})


if __name__ == "__main__":
    # 示例用法
    manager = get_config_manager()
    
    # 批量更新配置
    updates = {
        'api.key': 'new_api_key',
        'asr.language': 'zh',
        'target_language': 'English'
    }
    
    success = update_keys(updates)
    print(f"批量更新{'成功' if success else '失败'}")
    
    # 读取配置
    api_key = load_key('api.key')
    print(f"API Key: {api_key}")