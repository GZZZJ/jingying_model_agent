"""
Proc 公共基类：封装各 Proc 的共性初始化逻辑。
Proc01Prepare 不继承此基类（它负责生成 metadata）。
"""
import os
import pickle


class BaseProc:
    """
    公共基类，自动完成:
    1. 加载 metadata.pkl
    2. 创建 Proc 级 cache 目录
    3. 加载人行特征列表
    4. 加载 table_fea_map
    """

    # 子类需覆盖此属性，指定 cache 子目录名
    PROC_CACHE_NAME = ''

    def __init__(self, config):
        self.config = config
        self._load_metadata(config)
        self._create_cache_dir()
        self._load_rh_features()
        self._load_table_fea_map()

    def _load_metadata(self, config):
        """加载 metadata.pkl，不存在时抛出明确错误"""
        metadata_save_path = os.path.join(config['project_path'], 'data', 'metadata.pkl')
        if not os.path.exists(metadata_save_path):
            raise FileNotFoundError(
                f"元数据文件不存在: {metadata_save_path}，请先执行 Proc01Prepare"
            )
        with open(metadata_save_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def _create_cache_dir(self):
        """创建 Proc 级 cache 目录"""
        if self.PROC_CACHE_NAME:
            self.proc_cache_path = os.path.join(
                self.metadata['cache_path'], self.PROC_CACHE_NAME
            )
        else:
            self.proc_cache_path = os.path.join(
                self.metadata['cache_path'], self.__class__.__name__
            )
        os.makedirs(self.proc_cache_path, exist_ok=True)

    def _load_rh_features(self):
        """加载人行特征列表（需要做特殊值替换）"""
        rh_fea_save_path = os.path.join(self.metadata['data_path'], 'rh_fea_list.pkl')
        if os.path.exists(rh_fea_save_path):
            with open(rh_fea_save_path, 'rb') as f:
                self.rh_feature_list = pickle.load(f)
        else:
            self.rh_feature_list = []

    def _load_table_fea_map(self):
        """加载宽表→特征映射"""
        table_fea_map_save_path = os.path.join(self.metadata['data_path'], 'table_fea_map.pkl')
        with open(table_fea_map_save_path, 'rb') as f:
            self.table_fea_map = pickle.load(f)
